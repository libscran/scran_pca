#ifndef SCRAN_PCA_SIMPLE_PCA_HPP
#define SCRAN_PCA_SIMPLE_PCA_HPP

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "irlba/irlba.hpp"
#include "irlba/parallel.hpp"
#include "Eigen/Dense"

#include <vector>
#include <type_traits>
#include <algorithm>

#include "utils.hpp"

/**
 * @file simple_pca.hpp
 * @brief Perform a simple PCA on a gene-by-cell matrix.
 */

namespace scran_pca {

/**
 * @brief Options for `simple_pca()`.
 */
struct SimplePcaOptions {
    /**
     * @cond
     */
    SimplePcaOptions() {
        irlba_options.cap_number = true;
    }
    /**
     * @endcond
     */

    /** 
     * Number of PCs to compute.
     * This should be no greater than the maximum number of PCs, i.e., the smaller dimension of the input matrix, otherwise an error will be thrown.
     * (This error can be avoided by setting `irlba::Options::cap_number = true` in `SimplePcaOptions::irlba_options`, in which case only the maximum number of PCs will be reported in the results.)
     */
    int number = 25;

    /**
     * Should genes be scaled to unit variance?
     * Genes with zero variance are ignored.
     */
    bool scale = false;

    /**
     * Should the PC matrix be transposed on output?
     * If `true`, the output matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     */
    bool transpose = true;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()` and `irlba::parallelize()`.
     */
    int num_threads = 1;

    /**
     * Whether to realize `tatami::Matrix` objects into an appropriate in-memory format before PCA.
     * This is typically faster but increases memory usage.
     */
    bool realize_matrix = true;

    /**
     * Further options to pass to `irlba::compute()`.
     */
    irlba::Options irlba_options;
};

/**
 * @cond
 */
namespace internal {

template<bool sparse_, typename Value_, typename Index_, class EigenVector_>
void compute_row_means_and_variances(const tatami::Matrix<Value_, Index_>& mat, int num_threads, EigenVector_& center_v, EigenVector_& scale_v) {
    if (mat.prefer_rows()) {
        tatami::parallelize([&](size_t, Index_ start, Index_ length) -> void {
            tatami::Options opt;
            opt.sparse_extract_index = false;
            auto ext = tatami::consecutive_extractor<sparse_>(&mat, true, start, length, opt);
            auto ncells = mat.ncol();
            std::vector<Value_> vbuffer(ncells);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                auto results = [&]{
                    if constexpr(sparse_) {
                        auto range = ext->fetch(vbuffer.data(), NULL);
                        return tatami_stats::variances::direct(range.value, range.number, ncells, /* skip_nan = */ false);
                    } else {
                        auto ptr = ext->fetch(vbuffer.data());
                        return tatami_stats::variances::direct(ptr, ncells, /* skip_nan = */ false);
                    }
                }();
                center_v.coeffRef(r) = results.first;
                scale_v.coeffRef(r) = results.second;
            }
        }, mat.nrow(), num_threads);

    } else {
        tatami::parallelize([&](size_t t, Index_ start, Index_ length) -> void {
            tatami::Options opt;
            auto ncells = mat.ncol();
            auto ext = tatami::consecutive_extractor<sparse_>(&mat, false, static_cast<Index_>(0), ncells, start, length, opt);

            typedef typename EigenVector_::Scalar Scalar;
            tatami_stats::LocalOutputBuffer<Scalar> cbuffer(t, start, length, center_v.data());
            tatami_stats::LocalOutputBuffer<Scalar> sbuffer(t, start, length, scale_v.data());

            auto running = [&]{
                if constexpr(sparse_) {
                    return tatami_stats::variances::RunningSparse<Scalar, Value_, Index_>(length, cbuffer.data(), sbuffer.data(), /* skip_nan = */ false, /* subtract = */ start);
                } else {
                    return tatami_stats::variances::RunningDense<Scalar, Value_, Index_>(length, cbuffer.data(), sbuffer.data(), /* skip_nan = */ false);
                }
            }();

            std::vector<Value_> vbuffer(length);
            typename std::conditional<sparse_, std::vector<Index_>, Index_>::type ibuffer(length);
            for (Index_ r = 0; r < ncells; ++r) {
                if constexpr(sparse_) {
                    auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    running.add(range.value, range.index, range.number);
                } else {
                    auto ptr = ext->fetch(vbuffer.data());
                    running.add(ptr);
                }
            }

            running.finish();
            cbuffer.transfer();
            sbuffer.transfer();
        }, mat.nrow(), num_threads);
    }
}

template<class IrlbaMatrix_, class EigenMatrix_, class EigenVector_>
auto run_irlba_deferred(
    const IrlbaMatrix_& mat,
    const SimplePcaOptions& options,
    EigenMatrix_& components, 
    EigenMatrix_& rotation, 
    EigenVector_& variance_explained,
    EigenVector_& center_v,
    EigenVector_& scale_v)
{
    irlba::Centered<IrlbaMatrix_, EigenVector_> centered(mat, center_v);
    if (options.scale) {
        irlba::Scaled<true, decltype(centered), EigenVector_> scaled(centered, scale_v, true);
        return irlba::compute(scaled, options.number, components, rotation, variance_explained, options.irlba_options);
    } else {
        return irlba::compute(centered, options.number, components, rotation, variance_explained, options.irlba_options);
    }
}

template<typename Value_, typename Index_, class EigenMatrix_, class EigenVector_>
void run_sparse(
    const tatami::Matrix<Value_, Index_>& mat, 
    const SimplePcaOptions& options,
    EigenMatrix_& components, 
    EigenMatrix_& rotation, 
    EigenVector_& variance_explained,
    EigenVector_& center_v,
    EigenVector_& scale_v,
    typename EigenVector_::Scalar& total_var,
    bool& converged)
{
    Index_ ngenes = mat.nrow();
    center_v.resize(ngenes);
    scale_v.resize(ngenes);

    if (options.realize_matrix) {
        // 'extracted' contains row-major contents...
        auto extracted = tatami::retrieve_compressed_sparse_contents<Value_, Index_>(
            &mat, 
            /* row = */ true, 
            /* two_pass = */ false, 
            /* threads = */ options.num_threads
        );

        // But we effectively transpose it to CSC with genes in columns.
        Index_ ncells = mat.ncol();
        irlba::ParallelSparseMatrix emat(
            ncells,
            ngenes,
            std::move(extracted.value),
            std::move(extracted.index),
            std::move(extracted.pointers), 
            true,
            options.num_threads
        ); 

        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            const auto& ptrs = emat.get_pointers();
            const auto& values = emat.get_values();
            for (size_t r = start, end = start + length; r < end; ++r) {
                auto offset = ptrs[r];
                Index_ num_nonzero = ptrs[r + 1] - offset;
                auto results = tatami_stats::variances::direct(values.data() + offset, num_nonzero, ncells, /* skip_nan = */ false);
                center_v.coeffRef(r) = results.first;
                scale_v.coeffRef(r) = results.second;
            }
        }, ngenes, options.num_threads);

        total_var = internal::process_scale_vector(options.scale, scale_v);
        auto out = run_irlba_deferred(emat, options, components, rotation, variance_explained, center_v, scale_v);
        converged = out.first;

    } else {
        compute_row_means_and_variances<true>(mat, options.num_threads, center_v, scale_v);
        total_var = internal::process_scale_vector(options.scale, scale_v);
        auto out = run_irlba_deferred(
            internal::TransposedTatamiWrapper<EigenVector_, Value_, Index_>(mat, options.num_threads), 
            options, 
            components, 
            rotation, 
            variance_explained, 
            center_v, 
            scale_v
        );
        converged = out.first;
    }
}

template<typename Value_, typename Index_, class EigenMatrix_, class EigenVector_>
void run_dense(
    const tatami::Matrix<Value_, Index_>& mat, 
    const SimplePcaOptions& options,
    EigenMatrix_& components, 
    EigenMatrix_& rotation, 
    EigenVector_& variance_explained, 
    EigenVector_& center_v,
    EigenVector_& scale_v,
    typename EigenVector_::Scalar& total_var,
    bool& converged)
{
    Index_ ngenes = mat.nrow();
    center_v.resize(ngenes);
    scale_v.resize(ngenes);

    if (options.realize_matrix) {
        // Create a matrix with genes in columns.
        Index_ ncells = mat.ncol();
        EigenMatrix_ emat(ncells, ngenes);

        // If emat is row-major, we want to fill it with columns of 'mat', so row_major = false.
        // If emat is column-major, we want to fill it with rows of 'mat', so row_major = true.
        tatami::convert_to_dense(&mat, /* row_major = */ !emat.IsRowMajor, emat.data(), options.num_threads);

        center_v.array() = emat.array().colwise().sum();
        if (ncells) {
            center_v /= ncells;
        } else {
            std::fill(center_v.begin(), center_v.end(), std::numeric_limits<typename EigenVector_::Scalar>::quiet_NaN());
        }
        emat.array().rowwise() -= center_v.adjoint().array(); // applying it to avoid wasting time with deferred operations inside IRLBA.

        scale_v.array() = emat.array().colwise().squaredNorm();
        if (ncells > 1) {
            scale_v /= ncells - 1;
        } else {
            std::fill(scale_v.begin(), scale_v.end(), std::numeric_limits<typename EigenVector_::Scalar>::quiet_NaN());
        }

        total_var = internal::process_scale_vector(options.scale, scale_v);
        if (options.scale) {
            emat.array().rowwise() /= scale_v.adjoint().array();
        }

        auto out = irlba::compute(emat, options.number, components, rotation, variance_explained, options.irlba_options);
        converged = out.first;

    } else {
        compute_row_means_and_variances<false>(mat, options.num_threads, center_v, scale_v);
        total_var = internal::process_scale_vector(options.scale, scale_v);
        auto out = run_irlba_deferred(
            internal::TransposedTatamiWrapper<EigenVector_, Value_, Index_>(mat, options.num_threads), 
            options, 
            components, 
            rotation, 
            variance_explained, 
            center_v, 
            scale_v
        );
        converged = out.first;
    }
}

}
/**
 * @endcond
 */

/**
 * @brief Results of `simple_pca()`.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<typename EigenMatrix_, typename EigenVector_>
struct SimplePcaResults {
    /**
     * Matrix of principal components.
     * By default, each row corresponds to a PC while each column corresponds to a cell in the input matrix.
     * If `SimplePcaOptions::transpose = false`, rows are cells instead.
     * The number of PCs is determined by `SimplePcaOptions::number`. 
     */
    EigenMatrix_ components;

    /**
     * Variance explained by each PC.
     * Each entry corresponds to a column in `components` and is in decreasing order.
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
     * The number of PCs is determined by `SimplePcaOptions::number`.
     */
    EigenMatrix_ rotation;

    /**
     * Centering vector.
     * Each entry corresponds to a row in the matrix and contains the mean value for that feature.
     */
    EigenVector_ center;

    /**
     * Scaling vector, only returned if `SimplePcaOptions::scale = true`.
     * Each entry corresponds to a row in the matrix and contains the scaling factor used to divide the feature values if `SimplePcaOptions::scale = true`.
     */
    EigenVector_ scale;

    /**
     * Whether the algorithm converged.
     */
    bool converged = false;
};

/**
 * Principal components analysis (PCA) for compression and denoising of single-cell expression data.
 *
 * The premise is that most of the variation in the dataset is driven by biology, as changes in pathway activity drive coordinated changes across multiple genes.
 * In contrast, technical noise is random and not synchronized across any one axis in the high-dimensional space.
 * This suggests that the earlier principal components (PCs) should be enriched for biological heterogeneity while the later PCs capture random noise.
 *
 * Our aim is to reduce the size of the data and reduce noise by only using the earlier PCs for downstream cell-based analyses (e.g., neighbor detection, clustering).
 * Most practitioners will keep the first 10-50 PCs, though the exact choice is fairly arbitrary - see `SimplePcaOptions::number` to specify the number of PCs.
 * As we are only interested in the top PCs, we can use approximate algorithms for faster computation, in particular [IRLBA](https://github.com/LTLA/CppIrlba).
 *
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
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
void simple_pca(const tatami::Matrix<Value_, Index_>& mat, const SimplePcaOptions& options, SimplePcaResults<EigenMatrix_, EigenVector_>& output) {
    irlba::EigenThreadScope t(options.num_threads);

    if (mat.sparse()) {
        internal::run_sparse(mat, options, output.components, output.rotation, output.variance_explained, output.center, output.scale, output.total_variance, output.converged);
    } else {
        internal::run_dense(mat, options, output.components, output.rotation, output.variance_explained, output.center, output.scale, output.total_variance, output.converged);
    }

    internal::clean_up(mat.ncol(), output.components, output.variance_explained);
    if (options.transpose) {
        output.components.adjointInPlace();
    }

    if (!options.scale) {
        output.scale = EigenVector_();
    }
}

/**
 * Overload of `simple_pca()` that allocates memory for the output.
 *
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 *
 * @param[in] mat The input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param options Further options.
 *
 * @return Results of the PCA.
 */
template<typename EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, typename Value_, typename Index_>
SimplePcaResults<EigenMatrix_, EigenVector_> simple_pca(const tatami::Matrix<Value_, Index_>& mat, const SimplePcaOptions& options) {
    SimplePcaResults<EigenMatrix_, EigenVector_> output;
    simple_pca(mat, options, output);
    return output;
}

}

#endif
