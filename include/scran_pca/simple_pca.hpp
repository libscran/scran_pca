#ifndef SCRAN_PCA_SIMPLE_PCA_HPP
#define SCRAN_PCA_SIMPLE_PCA_HPP

#include <vector>
#include <type_traits>
#include <algorithm>
#include <memory>
#include <optional>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "quickstats/quickstats.hpp"
#include "irlba/irlba.hpp"
#include "irlba/parallel.hpp"
#include "irlba_tatami/irlba_tatami.hpp"
#include "Eigen/Dense"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file simple_pca.hpp
 * @brief PCA on a gene-by-cell matrix.
 */

namespace scran_pca {

/**
 * @brief Options for `simple_pca()`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<typename EigenVector_ = Eigen::VectorXd>
struct SimplePcaOptions {
    /**
     * @cond
     */
    SimplePcaOptions() {
        // Avoid throwing an error if too many PCs are requested.
        irlba_options.cap_number = true;
    }
    /**
     * @endcond
     */

    /** 
     * Number of the top principal components (PCs) to compute.
     * Retaining more PCs will capture more biological signal at the cost of increasing noise and compute time.
     * If this is greater than the maximum number of PCs (i.e., the smaller dimension of the input matrix), only the maximum number of PCs will be reported in the results.
     */
    int number = 25;

    /**
     * Should genes be scaled to unit variance?
     * This ensures that each gene contributes equally to the PCA, favoring consistent variation across many genes rather than large variation in a few genes.
     * Genes with zero variance are ignored.
     */
    bool scale = false;

    /**
     * Should the PC matrix be transposed on output?
     * If `true`, the output matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     */
    bool transpose = true;

    /**
     * Whether to realize `tatami::Matrix` objects into an appropriate in-memory format before PCA.
     * This is typically faster but increases memory usage.
     */
    bool realize_matrix = true;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()` and `irlba::parallelize()`.
     * Note that the exact values returned by `simple_pca()` will change slightly with different `num_threads`,
     * due to (deterministic) differences in the order of floating-point summations.
     */
    int num_threads = 1;

    /**
     * Further options to pass to `irlba::compute()`.
     */
    irlba::Options<EigenVector_> irlba_options;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, class EigenVector_>
void compute_row_means_and_variances(const tatami::Matrix<Value_, Index_>& mat, const int num_threads, EigenVector_& center_v, EigenVector_& scale_v) {
    tatami_stats::RssOptions<typename EigenVector_::Scalar> opts;
    opts.num_threads = num_threads;
    opts.mean_placeholder = 0; // rss() emits NaNs if there are no cells, we replace them with zeros to avoid downstream problems with propagation.

    tatami_stats::RssBuffers<typename EigenVector_::Scalar> buffers;
    buffers.mean = center_v.data();
    buffers.rss = scale_v.data();
    tatami_stats::rss(true, mat, buffers, opts);

    const auto ncells = mat.ncol();
    if (ncells > 1) {
        scale_v /= ncells - 1;
    }
}

template<class EigenVector_, class EigenMatrix_>
std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > prepare_deferred_matrix_for_irlba(
    std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > ptr,
    const SimplePcaOptions<EigenVector_>& options,
    const EigenVector_& center_v,
    const EigenVector_& scale_v
) {
    std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > alt;
    alt.reset(new irlba::CenteredMatrix<EigenVector_, EigenMatrix_, I<decltype(ptr)>, I<decltype(&center_v)> >(std::move(ptr), &center_v));
    ptr.swap(alt);

    if (options.scale) {
        alt.reset(new irlba::ScaledMatrix<EigenVector_, EigenMatrix_, I<decltype(ptr)>, I<decltype(&scale_v)> >(std::move(ptr), &scale_v, true, true));
        ptr.swap(alt);
    }

    return ptr;
}

template<class EigenMatrix_, typename Value_, typename Index_, class EigenVector_>
std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > prepare_sparse_matrix_for_irlba(
    const tatami::Matrix<Value_, Index_>& mat, 
    const SimplePcaOptions<EigenVector_>& options,
    EigenVector_& center_v,
    EigenVector_& scale_v,
    typename EigenVector_::Scalar& total_var
) {
    const auto ngenes = mat.nrow();
    sanisizer::resize(center_v, ngenes);
    sanisizer::resize(scale_v, ngenes);
    std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > output;

    if (options.realize_matrix) {
        // 'extracted' contains row-major contents...
        auto extracted = tatami::retrieve_compressed_sparse_contents<Value_, Index_>(
            mat, 
            /* row = */ true, 
            [&]{
                tatami::RetrieveCompressedSparseContentsOptions opt;
                opt.two_pass = false;
                opt.num_threads = options.num_threads;
                return opt;
            }()
        );

        // But we effectively transpose it to CSC with genes in columns.
        const Index_ ncells = mat.ncol();
        const auto sparse_ptr = new irlba::ParallelSparseMatrix<
            EigenVector_,
            EigenMatrix_,
            I<decltype(extracted.value)>,
            I<decltype(extracted.index)>,
            I<decltype(extracted.pointers)>
        >(
            ncells,
            ngenes,
            std::move(extracted.value),
            std::move(extracted.index),
            std::move(extracted.pointers), 
            true,
            options.num_threads
        );
        output.reset(sparse_ptr);

        tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
            const auto& pointers = sparse_ptr->get_pointers();
            const auto& values = sparse_ptr->get_values();
            quickstats::RssWorkspace<typename EigenVector_::Scalar> work;

            for (Index_ g = start, end = start + length; g < end; ++g) {
                const auto offset = pointers[g];
                const auto next_offset = pointers[g + 1]; // increment won't overflow as 'g + 1 <= end'.
                const Index_ num_nonzero = next_offset - offset;
                const auto results = quickstats::rss(ncells, num_nonzero, values.data() + offset, work);
                center_v.coeffRef(g) = results.mean;
                scale_v.coeffRef(g) = results.rss;
            }
        }, ngenes, options.num_threads);

        if (ncells > 1) { 
            // if there are fewer than 2 cells, scale_v will naturally be set to zero.
            scale_v /= ncells - 1;
        } else if (!ncells) {
            // override quickstats::rss()'s setting of the mean to NaN if there are no cells.
            std::fill(center_v.begin(), center_v.end(), 0);
        }

        total_var = process_scale_vector(options.scale, scale_v);

    } else {
        compute_row_means_and_variances(mat, options.num_threads, center_v, scale_v);
        total_var = process_scale_vector(options.scale, scale_v);
        output.reset(
            new irlba_tatami::Transposed<EigenVector_, EigenMatrix_, Value_, Index_, decltype(&mat)>(&mat, options.num_threads)
        ); 
    }

    return prepare_deferred_matrix_for_irlba(std::move(output), options, center_v, scale_v);
}

template<class EigenMatrix_, typename Value_, typename Index_, class EigenVector_>
std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > prepare_dense_matrix_for_irlba(
    const tatami::Matrix<Value_, Index_>& mat, 
    const SimplePcaOptions<EigenVector_>& options,
    EigenVector_& center_v,
    EigenVector_& scale_v,
    typename EigenVector_::Scalar& total_var
) {
    const Index_ ngenes = mat.nrow();
    sanisizer::resize(center_v, ngenes);
    sanisizer::resize(scale_v, ngenes);

    if (options.realize_matrix) {
        // Create a matrix with genes in columns.
        const Index_ ncells = mat.ncol();
        auto emat = std::make_unique<EigenMatrix_>(
            sanisizer::cast<I<decltype(std::declval<EigenMatrix_>().rows())> >(ncells),
            sanisizer::cast<I<decltype(std::declval<EigenMatrix_>().cols())> >(ngenes)
        );

        // By default, Eigen's matrices are column major. In such cases, because we want to do
        // a transposition, we pretend it's row major during the conversion.
        static_assert(!EigenMatrix_::IsRowMajor);
        tatami::convert_to_dense(
            mat,
            /* row_major = */ true,
            emat->data(),
            [&]{
                tatami::ConvertToDenseOptions opt;
                opt.num_threads = options.num_threads;
                return opt;
            }()
        );

        center_v.array() = emat->array().colwise().sum();
        if (ncells) { // if there are no cells, center_v will naturally be set to zero.
            center_v /= ncells;
        }
        emat->array().rowwise() -= center_v.adjoint().array(); // applying it to avoid wasting time with deferred operations inside IRLBA.

        scale_v.array() = emat->array().colwise().squaredNorm();
        if (ncells > 1) { // if there are fewer than 2 cells, scale_v will naturally be set to zero.
            scale_v /= ncells - 1;
        }

        total_var = process_scale_vector(options.scale, scale_v);
        if (options.scale) {
            emat->array().rowwise() /= scale_v.adjoint().array();
        }

        return std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> >( 
            new irlba::SimpleMatrix<EigenVector_, EigenMatrix_, decltype(emat)>(std::move(emat))
        );

    } else {
        compute_row_means_and_variances(mat, options.num_threads, center_v, scale_v);
        total_var = process_scale_vector(options.scale, scale_v);
        std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > output(
            new irlba_tatami::Transposed<EigenVector_, EigenMatrix_, Value_, Index_, decltype(&mat)>(&mat, options.num_threads)
        ); 
        return prepare_deferred_matrix_for_irlba(std::move(output), options, center_v, scale_v);
    }
}
/**
 * @endcond
 */

/**
 * @brief Results of `simple_pca()`.
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<typename EigenMatrix_, typename EigenVector_>
struct SimplePcaResults {
    /**
     * Matrix of principal component scores.
     * By default, each row corresponds to a PC while each column corresponds to a cell in the input matrix.
     * If `SimplePcaOptions::transpose = false`, rows are cells instead.
     *
     * The number of PCs is the smaller of `SimplePcaOptions::number` and `min(NR, NC) - 1`,
     * where `NR` and `NC` are the number of rows and columns, respectively, of the input matrix.
     */
    EigenMatrix_ components;

    /**
     * Variance explained by each PC.
     * Each entry corresponds to a column in `components` and is in decreasing order.
     * The number of PCs is as described for `SimplePcaResults::components`.
     */
    EigenVector_ variance_explained;

    /**
     * Total variance of the dataset (possibly after scaling, if `SimplePcaOptions::scale = true`).
     * This can be used to divide `variance_explained` to obtain the percentage of variance explained.
     */
    typename EigenVector_::Scalar total_variance = 0;

    /**
     * Rotation matrix. 
     * Each row corresponds to a feature while each column corresponds to a PC.
     * The number of PCs is as described for `SimplePcaResults::components`.
     */
    EigenMatrix_ rotation;

    /**
     * Centering vector.
     * Each entry corresponds to a gene (i.e., row of the input matrix) and contains the mean value of that gene.
     * If the input matrix has no cells, the mean is set to zero for all genes.
     */
    EigenVector_ center;

    /**
     * Scaling vector, only returned if `SimplePcaOptions::scale = true`.
     * Each entry corresponds to a gene (i.e., row of the input matrix) and contains the scaling factor used to divide the feature values if `SimplePcaOptions::scale = true`.
     * This is usually the sample standard deviation of that gene.
     * For genes with zero variance, the scaling factor is set to 1 to avoid non-finite values upon scaling.
     * For input matrices with fewer than 2 cells, the scaling factor is set to 1 for all genes. 
     */
    std::optional<EigenVector_> scale;

    /**
     * Metrics for IRLBA, including whether the algorithm converged and the number of iterations/multiplications required. 
     */
    irlba::Metrics metrics;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, typename EigenMatrix_, class EigenVector_, class SubsetFunction_>
void simple_pca_internal(
    const tatami::Matrix<Value_, Index_>& mat,
    const SimplePcaOptions<EigenVector_>& options,
    SimplePcaResults<EigenMatrix_, EigenVector_>& output,
    SubsetFunction_ subset_fun
) {
    irlba::EigenThreadScope t(options.num_threads);

    auto scale = tatami::create_container_of_Index_size<EigenVector_>(mat.nrow());
    std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > ptr;
    if (mat.sparse()) {
        ptr = prepare_sparse_matrix_for_irlba<EigenMatrix_>(mat, options, output.center, scale, output.total_variance);
    } else {
        ptr = prepare_dense_matrix_for_irlba<EigenMatrix_>(mat, options, output.center, scale, output.total_variance);
    }

    output.metrics = irlba::compute(*ptr, options.number, output.components, output.rotation, output.variance_explained, options.irlba_options);

    subset_fun(output.components, output.variance_explained);

    clean_up(mat.ncol(), output.components, output.variance_explained);
    if (options.transpose) {
        output.components.adjointInPlace();
    }

    if (options.scale) {
        output.scale = std::move(scale);
    }
}
/**
 * @endcond
 */

/**
 * Principal components analysis (PCA) for compression and denoising of single-cell expression data.
 *
 * We assume that most variation in the dataset is driven by biological differences between subpopulations that drive coordinated changes across multiple genes in the same pathways.
 * In contrast, technical noise is random and not synchronized across any one axis in the high-dimensional space.
 * This suggests that the earlier principal components (PCs) should be enriched for biological heterogeneity while the later PCs capture random noise.
 *
 * Our aim is to reduce the size of the data and eliminate noise by only using the earlier PCs for downstream cell-based analyses (e.g., neighbor detection, clustering).
 * Most practitioners will keep the first 10-50 PCs, though the exact choice is fairly arbitrary - see `SimplePcaOptions::number` to specify the number of PCs.
 * As we are only interested in the top PCs, we can use approximate algorithms for faster computation, in particular [IRLBA](https://github.com/LTLA/CppIrlba).
 *
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type of the indices.
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 *
 * @param[in] mat The input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param options Further options.
 * @param[out] output On output, the results of the PCA on `mat`.
 * This can be re-used across multiple calls to `simple_pca()`. 
 */
template<typename Value_, typename Index_, typename EigenMatrix_, class EigenVector_>
void simple_pca(const tatami::Matrix<Value_, Index_>& mat, const SimplePcaOptions<EigenVector_>& options, SimplePcaResults<EigenMatrix_, EigenVector_>& output) {
    simple_pca_internal(
        mat,
        options,
        output, 
        [](const EigenMatrix_&, const EigenVector_&) -> void {}
    );
}

/**
 * Overload of `simple_pca()` that allocates memory for the output.
 *
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type of the indices.
 *
 * @param[in] mat The input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param options Further options.
 *
 * @return Results of the PCA.
 */
template<typename EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, typename Value_, typename Index_>
SimplePcaResults<EigenMatrix_, EigenVector_> simple_pca(const tatami::Matrix<Value_, Index_>& mat, const SimplePcaOptions<EigenVector_>& options) {
    SimplePcaResults<EigenMatrix_, EigenVector_> output;
    simple_pca(mat, options, output);
    return output;
}

}

#endif
