#ifndef SCRAN_PCA_BLOCKED_PCA_HPP
#define SCRAN_PCA_BLOCKED_PCA_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <cstddef>
#include <functional>
#include <optional>
#include <cassert>

#include "tatami/tatami.hpp"
#include "irlba/irlba.hpp"
#include "irlba/parallel.hpp"
#include "irlba_tatami/irlba_tatami.hpp"
#include "Eigen/Dense"
#include "scran_blocks/scran_blocks.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file blocked_pca.hpp
 * @brief PCA on residuals after regressing out a blocking factor.
 */

namespace scran_pca {

/**
 * @brief Options for `blocked_pca()`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<typename EigenVector_ = Eigen::VectorXd>
struct BlockedPcaOptions {
    /**
     * @cond
     */
    BlockedPcaOptions() {
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
     * In the presence of a blocking factor, each gene's variance is calculated as a weighted sum of the variances from each block. 
     * Genes with zero variance are ignored.
     */
    bool scale = false;

    /**
     * Should the PC matrix be transposed on output?
     * If `true`, the output matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     */
    bool transpose = true;

    /**
     * Policy for weighting the contribution of blocks of different size.
     *
     * The default of `scran_blocks::WeightPolicy::VARIABLE` is to define equal weights for blocks once they reach a certain size (see `BlockedPcaOptions::variable_block_weight_parameters`).
     * For smaller blocks, the weight is linearly proportional to its size to avoid outsized contributions from very small blocks.
     *
     * Other options include `scran_blocks::WeightPolicy::EQUAL`, where all blocks are equally weighted regardless of size;
     * and `scran_blocks::WeightPolicy::NONE`, where the contribution of each block is proportional to its size.
     */
    scran_blocks::WeightPolicy block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;

    /**
     * Parameters for the variable block weights, including the threshold at which blocks are considered to be large enough to have equal weight.
     * Only used when `BlockedPcaOptions::block_weight_policy = scran_blocks::WeightPolicy::VARIABLE`.
     */
    scran_blocks::VariableWeightParameters variable_block_weight_parameters;

    /**
     * Whether to to center the PC scores for each block at the origin.
     * If `true`, the cells for each block are shifted so that their per-block centroid lies at the origin.
     * This could change the relative positions of cells in different blocks.
     * If `false`, the centroid of all cells is set to the origin, without affecting the relative positions of cells in different blocks.
     */
    bool center_scores_by_block = false;

    /**
     * Whether to realize `tatami::Matrix` objects into an appropriate in-memory format before PCA.
     * This is typically faster but increases memory usage.
     */
    bool realize_matrix = true;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()` and `irlba::parallelize()`.
     * Note that the exact values returned by `blocked_pca()` will change slightly with different `num_threads`,
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
/*****************************************************
 ************* Blocking data structures **************
 *****************************************************/

template<class EigenVector_>
struct BlockingDetails {
    template<typename Index_>
    BlockingDetails(std::size_t num_blocks, Index_ num_cells) : 
        per_element_weight(sanisizer::cast<I<decltype(per_element_weight.size())> >(num_blocks)),
        expanded_weights(tatami::cast_Index_to_container_size<EigenVector_>(num_cells))
    {}

    typedef typename EigenVector_::Scalar Weight;
    std::vector<Weight> per_element_weight;
    Weight total_block_weight = 0;
    EigenVector_ expanded_weights;
};

template<class EigenVector_, typename Index_, typename Block_>
std::optional<BlockingDetails<EigenVector_> > compute_blocking_details(
    const Index_ ncells,
    const Block_* block,
    const std::size_t num_blocks,
    const std::vector<Index_>& block_sizes,
    const scran_blocks::WeightPolicy block_weight_policy, 
    const scran_blocks::VariableWeightParameters& variable_block_weight_parameters
) {
    if (block_weight_policy == scran_blocks::WeightPolicy::NONE) {
        return std::optional<BlockingDetails<EigenVector_> >();
    }

    BlockingDetails<EigenVector_> output(num_blocks, ncells);
    auto& total_weight = output.total_block_weight;
    auto& element_weight = output.per_element_weight;

    for (std::size_t b = 0; b < num_blocks; ++b) {
        const auto bsize = block_sizes[b];

        // Computing effective block weights that also incorporate division by the
        // block size. This avoids having to do the division by block size in the
        // 'compute_blockwise_mean_and_variance*()' functions.
        if (bsize) {
            typename EigenVector_::Scalar block_weight = 1;
            if (block_weight_policy == scran_blocks::WeightPolicy::VARIABLE) {
                block_weight = scran_blocks::compute_variable_weight(bsize, variable_block_weight_parameters);
            }

            element_weight[b] = block_weight / bsize;
            total_weight += block_weight;
        } else {
            element_weight[b] = 0;
        }
    }

    // Setting a placeholder value to avoid problems with division by zero.
    if (total_weight == 0) {
        total_weight = 1; 
    }

    // Expanding them for multiplication in the IRLBA wrappers.
    auto sqrt_weights = element_weight;
    for (auto& s : sqrt_weights) {
        s = std::sqrt(s);
    }

    auto& expanded = output.expanded_weights;
    for (Index_ c = 0; c < ncells; ++c) {
        expanded.coeffRef(c) = sqrt_weights[block[c]];
    }

    return output;
}

/*****************************************************************
 ************ Computing the blockwise mean and variance **********
 *****************************************************************/

template<class IrlbaSparseMatrix_, typename Block_, class Index_, class EigenVector_, class EigenMatrix_>
void compute_blockwise_mean_and_variance_realized_sparse(
    const IrlbaSparseMatrix_& emat, // this should be column-major with genes in the columns.
    const Block_* block,
    const std::size_t num_blocks,
    const std::vector<Index_>& block_sizes, 
    const std::optional<BlockingDetails<EigenVector_> >& block_details,
    EigenMatrix_& centers,
    EigenVector_& variances,
    const int nthreads
) {
    const auto ngenes = emat.cols();
    const auto ncells = emat.rows();
    const auto& values = emat.get_values();
    const auto& indices = emat.get_indices();
    const auto& pointers = emat.get_pointers();
    static_assert(!EigenMatrix_::IsRowMajor);

    assert(sanisizer::is_equal(ngenes, variances.size()));
    assert(sanisizer::is_equal(ngenes, centers.cols()));
    assert(sanisizer::is_equal(num_blocks, centers.rows()));

    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        auto block_zeros = sanisizer::create<std::vector<Index_> >(num_blocks);
        auto block_rss = sanisizer::create<std::vector<typename EigenVector_::Scalar> >(num_blocks);
        auto block_centers = sanisizer::create<std::vector<typename EigenMatrix_::Scalar> >(num_blocks); // use a local copy to avoid false sharing.

        for (I<decltype(start)> g = start, end = start + length; g < end; ++g) {
            const auto offset = pointers[g];
            const auto num_nonzero = pointers[g + 1] - offset; // increment won't overflow as 'g < end' and 'end' is of the same type.

            const auto vptr = values.data() + offset;
            const auto iptr = indices.data() + offset;

            std::fill(block_centers.begin(), block_centers.end(), 0);
            for (I<decltype(num_nonzero)> i = 0; i < num_nonzero; ++i) {
                block_centers[block[iptr[i]]] += vptr[i];
            }
            for (std::size_t b = 0; b < num_blocks; ++b) {
                const auto bsize = block_sizes[b];
                if (bsize) {
                    block_centers[b] /= bsize;
                }
            }

            // Computing the RSS instead of the sample variance.
            // We don't consider the loss of residual d.f. from estimating the block means, as the PCA doesn't either.
            std::copy(block_sizes.begin(), block_sizes.end(), block_zeros.begin());
            std::fill(block_rss.begin(), block_rss.end(), 0);

            for (I<decltype(num_nonzero)> i = 0; i < num_nonzero; ++i) {
                const Block_ curb = block[iptr[i]];
                const auto diff = vptr[i] - block_centers[curb];
                block_rss[curb] += diff * diff;
                --block_zeros[curb];
            }

            typename EigenVector_::Scalar rss = 0; 
            for (std::size_t b = 0; b < num_blocks; ++b) {
                const auto bsize = block_sizes[b];
                if (bsize) {
                    const auto val = block_centers[b];
                    const auto final_rss = block_rss[b] + val * val * block_zeros[b];
                    if (block_details.has_value()) {
                        rss += final_rss * block_details->per_element_weight[b];
                    } else {
                        rss += final_rss;
                    }
                }
            }

            // COMMENT ON DENOMINATOR:
            // If we're not dealing with weights, we compute the actual sample variance for easy interpretation
            // (and to match up with the per-PC calculations in clean_up).
            //
            // If we're dealing with weights, the concept of the sample variance becomes somewhat weird.
            // So, we just use the same denominator for consistency in clean_up_projected.
            // Magnitude doesn't matter when scaling for process_scale_vector anyway.
            //
            // If there are not enough cells, we set the variance to zero so that no scaling is done in process_scale_vector().
            // We don't set this to NaN to avoid problems with propagation.
            if (ncells > 1) {
                variances[g] = rss / (ncells - 1);
            } else {
                variances[g] = 0;
            }

            std::copy(block_centers.begin(), block_centers.end(), centers.data() + sanisizer::product_unsafe<std::size_t>(g, num_blocks));
        }
    }, ngenes, nthreads);
}

template<class EigenMatrix_, typename Block_, class Index_, class EigenVector_>
void compute_blockwise_mean_and_variance_realized_dense(
    const EigenMatrix_& emat, // this should be column-major with genes in the columns.
    const Block_* block, 
    const std::size_t num_blocks,
    const std::vector<Index_>& block_sizes, 
    const std::optional<BlockingDetails<EigenVector_> >& block_details,
    EigenMatrix_& centers,
    EigenVector_& variances,
    const int nthreads
) {
    const auto ngenes = emat.cols();
    const auto ncells = emat.rows();
    static_assert(!EigenMatrix_::IsRowMajor);

    assert(sanisizer::is_equal(ngenes, variances.size()));
    assert(sanisizer::is_equal(ngenes, centers.cols()));
    assert(sanisizer::is_equal(num_blocks, centers.rows()));

    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        auto block_rss = sanisizer::create<std::vector<typename EigenVector_::Scalar> >(num_blocks);
        auto block_centers = sanisizer::create<std::vector<typename EigenMatrix_::Scalar> >(num_blocks); // use a local copy to avoid false sharing.

        for (Index_ g = start, end = start + length; g < end; ++g) {
            const auto values = emat.data() + sanisizer::product_unsafe<std::size_t>(g, ncells);

            std::fill(block_centers.begin(), block_centers.end(), 0);
            for (I<decltype(ncells)> i = 0; i < ncells; ++i) {
                block_centers[block[i]] += values[i];
            }
            for (std::size_t b = 0; b < num_blocks; ++b) {
                const auto bsize = block_sizes[b];
                if (bsize) {
                    block_centers[b] /= bsize;
                }
            }

            // See comments above on why we're computing RSS's.
            std::fill(block_rss.begin(), block_rss.end(), 0);
            for (I<decltype(ncells)> i = 0; i < ncells; ++i) {
                const auto curb = block[i];
                const auto delta = values[i] - block_centers[curb];
                block_rss[curb] += delta * delta;
            }

            typename EigenVector_::Scalar rss = 0; 
            for (std::size_t b = 0; b < num_blocks; ++b) {
                if (block_sizes[b]) {
                    if (block_details.has_value()) {
                        rss += block_rss[b] * block_details->per_element_weight[b];
                    } else {
                        rss += block_rss[b];
                    }
                }
            }

            // See COMMENT ON DENOMINATOR above.
            if (ncells > 1) {
                variances[g] = rss / (ncells - 1);
            } else {
                variances[g] = 0;
            }

            std::copy(block_centers.begin(), block_centers.end(), centers.data() + sanisizer::product_unsafe<std::size_t>(g, num_blocks));
        }
    }, ngenes, nthreads);
}

template<typename Value_, typename Index_, typename Block_, class EigenMatrix_, class EigenVector_>
void compute_blockwise_mean_and_variance_tatami(
    const tatami::Matrix<Value_, Index_>& mat, // this should have genes in the rows!
    const Block_* block, 
    const std::size_t num_blocks,
    const std::vector<Index_>& block_sizes, 
    const std::optional<BlockingDetails<EigenVector_> >& block_details,
    EigenMatrix_& centers,
    EigenVector_& variances,
    const int nthreads
) {
    static_assert(!EigenMatrix_::IsRowMajor); // need this for correct pointer calculations.
    typedef typename EigenMatrix_::Scalar Float;

    const auto ngenes = mat.nrow();
    EigenMatrix_ tmp_mean(
        sanisizer::cast<I<decltype(std::declval<EigenMatrix_>().rows())> >(ngenes),
        sanisizer::cast<I<decltype(std::declval<EigenMatrix_>().cols())> >(num_blocks)
    );

    tatami_stats::GroupRssBuffers<Float> buffers;
    buffers.mean.reserve(num_blocks);
    buffers.rss.reserve(num_blocks);
    auto tmp_rss = sanisizer::create<std::vector<std::vector<Float> > >(num_blocks);

    for (std::size_t b = 0; b < num_blocks; ++b) {
        buffers.mean.push_back(tmp_mean.data() + sanisizer::product_unsafe<std::size_t>(ngenes, b));
        tatami::resize_container_to_Index_size(tmp_rss[b], ngenes);
        buffers.rss.push_back(tmp_rss[b].data());
    }

    tatami_stats::GroupRssOptions<Float> opt;
    opt.num_threads = nthreads;
    opt.mean_placeholder = 0; // avoid NaN propagation in ResidualMatrix.
    tatami_stats::group_rss(true, mat, block, num_blocks, block_sizes.data(), buffers, opt);

    assert(sanisizer::is_equal(variances.size(), ngenes));
    variances.setZero();
    for (std::size_t b = 0; b < num_blocks; ++b) {
        if (block_sizes[b]) {
            const auto& currss = tmp_rss[b];
            if (block_details.has_value()) {
                for (Index_  g = 0; g < ngenes; ++g) {
                    variances.coeffRef(g) += currss[g] * block_details->per_element_weight[b];
                }
            } else {
                for (Index_  g = 0; g < ngenes; ++g) {
                    variances.coeffRef(g) += currss[g];
                }
            }
        }
    }

    centers = tmp_mean.adjoint();

    // See COMMENT ON DENOMINATOR above.
    const auto ncells = mat.ncol();
    if (ncells > 1) {
        for (Index_  g = 0; g < ngenes; ++g) {
            variances.coeffRef(g) /= ncells - 1;
        }
    }
}

/******************************************************************
 ************ Project matrices on their rotation vectors **********
 ******************************************************************/

template<class EigenMatrix_, class EigenVector_>
const EigenMatrix_& scale_rotation_matrix(const EigenMatrix_& rotation, bool scale, const EigenVector_& scale_v, EigenMatrix_& tmp) {
    if (scale) {
        tmp = (rotation.array().colwise() / scale_v.array()).matrix();
        return tmp;
    } else {
        return rotation;
    }
}

template<class EigenVector_, class IrlbaSparseMatrix_, class EigenMatrix_>
inline void project_matrix_realized_sparse(
    const IrlbaSparseMatrix_& emat, // cell in rows, genes in the columns, CSC.
    EigenMatrix_& components, // dims in rows, cells in columns
    const EigenMatrix_& scaled_rotation, // genes in rows, dims in columns
    int nthreads
) {
    const auto rank = scaled_rotation.cols();
    const auto ncells = emat.rows();
    const auto ngenes = emat.cols();

    // Store as transposed for more cache efficiency.
    components.resize(
        sanisizer::cast<I<decltype(components.rows())> >(rank),
        sanisizer::cast<I<decltype(components.cols())> >(ncells)
    );
    components.setZero();

    const auto& values = emat.get_values();
    const auto& indices = emat.get_indices();
    const auto& pointers = emat.get_pointers();

    if (nthreads == 1) {
        auto multipliers = sanisizer::create<EigenVector_>(rank);
        for (I<decltype(ngenes)> g = 0; g < ngenes; ++g) {
            multipliers.noalias() = scaled_rotation.row(g);
            const auto start = pointers[g], end = pointers[g + 1]; // increment is safe as 'g + 1 <= ngenes'.
            for (auto i = start; i < end; ++i) {
                components.col(indices[i]).noalias() += values[i] * multipliers;
            }
        }

    } else {
        // Here, the general strategy is to split the matrix by chunks into genes,
        // perform the matrix multiplication for each chunk,
        // and then sum the per-chunk products to obtain the final product.
        // The exact result of the reduction depends on the number of threads,
        // but this is an acceptable annoyance for greater speed.
        const auto& primary_bounds = emat.get_primary_boundaries();
        auto working = sanisizer::create<std::vector<EigenMatrix_> >(nthreads - 1);

        irlba::parallelize(nthreads, [&](const int t) -> void { 
            EigenMatrix_* ptr;
            if (t == 0) {
                ptr = &components;
            } else {
                auto& mat = working[t - 1];
                mat.resize(components.rows(), components.cols());
                mat.setZero();
                ptr = &mat;
            }

            const auto gstart = primary_bounds[t];
            const auto gend = primary_bounds[t + 1]; // increment is safe as 't + 1 <= nthreads'.
            auto multipliers = sanisizer::create<EigenVector_>(rank);
            for (I<decltype(ngenes)> g = gstart; g < gend; ++g) {
                multipliers.noalias() = scaled_rotation.row(g);
                const auto start = pointers[g], end = pointers[g + 1]; // increment is safe as 'g + 1 <= ngenes'
                for (auto i = start; i < end; ++i) {
                    ptr->col(indices[i]).noalias() += values[i] * multipliers;
                }
            }
        });

        for (auto& w : working) {
            components += w;
        }
    }
}

template<typename Value_, typename Index_, class EigenMatrix_>
void project_matrix_transposed_tatami(
    const tatami::Matrix<Value_, Index_>& mat, // genes in rows, cells in columns
    EigenMatrix_& components,
    const EigenMatrix_& scaled_rotation, // genes in rows, dims in columns
    const int nthreads) 
{
    const auto rank = scaled_rotation.cols();
    const auto ngenes = mat.nrow();
    const auto ncells = mat.ncol();

    // Store as transposed for more cache efficiency.
    // This is a column-major rank x ncells matrix, which makes it a row-major ncells x rank matrix.
    components.resize(
        sanisizer::cast<I<decltype(components.rows())> >(rank),
        sanisizer::cast<I<decltype(components.cols())> >(ncells)
    );

    tatami::DelayedTranspose<Value_, Index_> tmat(tatami::wrap_shared_ptr(&mat));
    static_assert(!EigenMatrix_::IsRowMajor);
    auto get_right = [&](I<decltype(rank)> r) -> auto {
        return scaled_rotation.data() + sanisizer::product_unsafe<std::size_t>(r, ngenes);
    };

    if (tmat.is_sparse()) {
        if (tmat.prefer_rows()) {
            tatami_mult::MultiplySparseRowWithDenseColumnMatrixToRowOutputOptions options;
            options.num_threads = nthreads;
            tatami_mult::multiply_sparse_row_with_dense_column_matrix_to_row_output(tmat, rank, get_right, components.data(), options);
        } else {
            tatami_mult::MultiplySparseColumnWithDenseColumnMatrixToRowOutputOptions options;
            options.num_threads = nthreads;
            tatami_mult::multiply_sparse_column_with_dense_column_matrix_to_row_output(tmat, rank, get_right, components.data(), options);
        }
    } else {
        if (tmat.prefer_rows()) {
            tatami_mult::MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions options;
            options.num_threads = nthreads;
            tatami_mult::multiply_dense_row_with_dense_column_matrix_to_row_output(tmat, rank, get_right, components.data(), options);
        } else {
            tatami_mult::MultiplyDenseColumnWithDenseColumnMatrixToRowOutputOptions options;
            options.num_threads = nthreads;
            tatami_mult::multiply_dense_column_with_dense_column_matrix_to_row_output(tmat, rank, get_right, components.data(), options);
        }
    }
}

template<class EigenMatrix_, class EigenVector_>
void clean_up_projected(EigenMatrix_& projected, EigenVector_& D) {
    // Empirically centering to give nice centered PCs, because we can't
    // guarantee that the projection is centered in this manner.
    for (I<decltype(projected.rows())> i = 0, prows = projected.rows(); i < prows; ++i) {
        projected.row(i).array() -= projected.row(i).sum() / projected.cols();
    }

    // Just dividing by the number of observations - 1 regardless of weighting.
    const typename EigenMatrix_::Scalar denom = projected.cols() - 1;
    if (denom) {
        for (auto& d : D) {
            d = d * d / denom;
        }
    }
}

/*******************************
 ***** Residual wrapper ********
 *******************************/

template<class EigenVector_, class IrlbaMatrix_, typename Block_, class CenterMatrix_>
class ResidualWorkspace final : public irlba::Workspace<EigenVector_> {
public:
    ResidualWorkspace(const IrlbaMatrix_& matrix, const Block_* block, const CenterMatrix_& means) :
        my_work(matrix.new_known_workspace()),
        my_block(block),
        my_means(means),
        my_sub(sanisizer::cast<I<decltype(my_sub.size())> >(my_means.rows()))
    {}

private:
    I<decltype(std::declval<IrlbaMatrix_>().new_known_workspace())> my_work;
    const Block_* my_block;
    const CenterMatrix_& my_means;
    EigenVector_ my_sub;

public:
    void multiply(const EigenVector_& right, EigenVector_& output) {
        my_work->multiply(right, output);

        my_sub.noalias() = my_means * right;
        for (I<decltype(output.size())> i = 0, end = output.size(); i < end; ++i) {
            auto& val = output.coeffRef(i);
            val -= my_sub.coeff(my_block[i]);
        }
    }
};

template<class EigenVector_, class IrlbaMatrix_, typename Block_, class CenterMatrix_>
class ResidualAdjointWorkspace final : public irlba::AdjointWorkspace<EigenVector_> {
public:
    ResidualAdjointWorkspace(const IrlbaMatrix_& matrix, const Block_* block, const CenterMatrix_& means) :
        my_work(matrix.new_known_adjoint_workspace()),
        my_block(block),
        my_means(means),
        my_aggr(sanisizer::cast<I<decltype(my_aggr.size())> >(my_means.rows()))
    {}

private:
    I<decltype(std::declval<IrlbaMatrix_>().new_known_adjoint_workspace())> my_work;
    const Block_* my_block;
    const CenterMatrix_& my_means;
    EigenVector_ my_aggr;

public:
    void multiply(const EigenVector_& right, EigenVector_& output) {
        my_work->multiply(right, output);

        my_aggr.setZero();
        for (I<decltype(right.size())> i = 0, end = right.size(); i < end; ++i) {
            my_aggr.coeffRef(my_block[i]) += right.coeff(i); 
        }

        output.noalias() -= my_means.adjoint() * my_aggr;
    }
};

template<class EigenMatrix_, class IrlbaMatrix_, typename Block_, class CenterMatrix_>
class ResidualRealizeWorkspace final : public irlba::RealizeWorkspace<EigenMatrix_> {
public:
    ResidualRealizeWorkspace(const IrlbaMatrix_& matrix, const Block_* block, const CenterMatrix_& means) :
        my_work(matrix.new_known_realize_workspace()),
        my_block(block),
        my_means(means)
    {}

private:
    I<decltype(std::declval<IrlbaMatrix_>().new_known_realize_workspace())> my_work;
    const Block_* my_block;
    const CenterMatrix_& my_means;

public:
    const EigenMatrix_& realize(EigenMatrix_& buffer) {
        my_work->realize_copy(buffer);
        for (I<decltype(buffer.rows())> i = 0, end = buffer.rows(); i < end; ++i) {
            buffer.row(i) -= my_means.row(my_block[i]);
        }
        return buffer;
    }
};

// This wrapper class mimics multiplication with the residuals,
// i.e., after subtracting the per-block mean from each cell.
template<class EigenVector_, class EigenMatrix_, class IrlbaMatrixPointer_, class Block_, class CenterMatrixPointer_>
class ResidualMatrix final : public irlba::Matrix<EigenVector_, EigenMatrix_>  {
public:
    ResidualMatrix(IrlbaMatrixPointer_ mat, const Block_* block, CenterMatrixPointer_ means) : 
        my_matrix(std::move(mat)),
        my_block(block),
        my_means(std::move(means)) 
    {}

public:
    Eigen::Index rows() const {
        return my_matrix->rows();
    }

    Eigen::Index cols() const {
        return my_matrix->cols();
    }

private:
    IrlbaMatrixPointer_ my_matrix;
    const Block_* my_block;
    CenterMatrixPointer_ my_means;

public:
    std::unique_ptr<irlba::Workspace<EigenVector_> > new_workspace() const {
        return new_known_workspace();
    }

    std::unique_ptr<irlba::AdjointWorkspace<EigenVector_> > new_adjoint_workspace() const {
        return new_known_adjoint_workspace();
    }

    std::unique_ptr<irlba::RealizeWorkspace<EigenMatrix_> > new_realize_workspace() const {
        return new_known_realize_workspace();
    }

public:
    std::unique_ptr<ResidualWorkspace<EigenVector_, decltype(*my_matrix), Block_, decltype(*my_means)> > new_known_workspace() const {
        return std::make_unique<ResidualWorkspace<EigenVector_, decltype(*my_matrix), Block_, decltype(*my_means)> >(*my_matrix, my_block, *my_means);
    }

    std::unique_ptr<ResidualAdjointWorkspace<EigenVector_, decltype(*my_matrix), Block_, decltype(*my_means)> > new_known_adjoint_workspace() const {
        return std::make_unique<ResidualAdjointWorkspace<EigenVector_, decltype(*my_matrix), Block_, decltype(*my_means)> >(*my_matrix, my_block, *my_means);
    }

    std::unique_ptr<ResidualRealizeWorkspace<EigenMatrix_, decltype(*my_matrix), Block_, decltype(*my_means)> > new_known_realize_workspace() const {
        return std::make_unique<ResidualRealizeWorkspace<EigenMatrix_, decltype(*my_matrix), Block_, decltype(*my_means)> >(*my_matrix, my_block, *my_means);
    }
};
/**
 * @endcond
 */

/**
 * @brief Results of `blocked_pca()`.
 *
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<typename EigenMatrix_, typename EigenVector_>
struct BlockedPcaResults {
    /**
     * Matrix of principal component scores.
     * By default, each row corresponds to a PC while each column corresponds to a cell in the input matrix.
     * If `BlockedPcaOptions::transpose = false`, rows are cells instead.
     *
     * The number of PCs is the smaller of `BlockedPcaOptions::number` and `min(NR, NC) - 1`,
     * where `NR` and `NC` are the number of rows and columns, respectively, of the input matrix.
     */
    EigenMatrix_ components;

    /**
     * Variance explained by each PC.
     * Each entry corresponds to a column in `components` and is in decreasing order.
     * The number of PCs is as described for `BlockedPcaResults::components`.
     */
    EigenVector_ variance_explained;

    /**
     * Total variance of the dataset (possibly after scaling, if `BlockedPcaOptions::scale = true`).
     * This can be used to divide `variance_explained` to obtain the percentage of variance explained.
     */
    typename EigenVector_::Scalar total_variance = 0;

    /**
     * Rotation matrix.
     * Each row corresponds to a gene (i.e., row of the input matrix) while each column corresponds to a PC.
     * The number of PCs is as described for `BlockedPcaResults::components`.
     */
    EigenMatrix_ rotation;

    /**
     * Centering matrix.
     * Each row corresponds to a block and each column corresponds to a gene (i.e., row of the input matrix).
     * Each entry contains the mean of a particular gene in the corresponding block.
     * For empty blocks, the mean for all genes is set to zero.
     */
    EigenMatrix_ center;

    /**
     * Scaling vector, only returned if `BlockedPcaOptions::scale = true`.
     * Each entry corresponds to a gene (i.e., row of the input matrix) and contains the scaling factor used to divide that gene's values if `BlockedPcaOptions::scale = true`.
     * This is usually the weighted sum of the per-block sample standard deviation of that gene.
     * For genes with zero variance in all blocks, the scaling factor is set to 1 to avoid non-finite values upon scaling.
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
template<typename Value_, typename Index_, typename Block_, typename EigenMatrix_, class EigenVector_, class SubsetFunction_>
void blocked_pca_internal(
    const tatami::Matrix<Value_, Index_>& mat,
    const Block_* block,
    const std::size_t num_blocks,
    const BlockedPcaOptions<EigenVector_>& options,
    BlockedPcaResults<EigenMatrix_, EigenVector_>& output,
    SubsetFunction_ subset_fun
) {
    irlba::EigenThreadScope t(options.num_threads);
    std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > ptr;
    std::function<void(const EigenMatrix_&)> projector;

    const Index_ ngenes = mat.nrow(), ncells = mat.ncol(); 
    output.center.resize(
        sanisizer::cast<I<decltype(output.center.rows())> >(num_blocks),
        sanisizer::cast<I<decltype(output.center.cols())> >(ngenes)
    );
    auto scale = tatami::create_container_of_Index_size<EigenVector_>(ngenes);

    auto block_sizes = sanisizer::create<std::vector<Index_> >(num_blocks);
    for (Index_ c = 0; c < ncells; ++c) {
        block_sizes[block[c]] += 1;
    }
    auto block_details = compute_blocking_details<EigenVector_>(  
        mat.ncol(),
        block,
        num_blocks,
        block_sizes,
        options.block_weight_policy,
        options.variable_block_weight_parameters
    );

    if (!options.realize_matrix) {
        compute_blockwise_mean_and_variance_tatami(
            mat,
            block,
            num_blocks,
            block_sizes,
            block_details,
            output.center,
            scale,
            options.num_threads
        );
        ptr.reset(new irlba_tatami::Transposed<EigenVector_, EigenMatrix_, Value_, Index_, decltype(&mat)>(&mat, options.num_threads));
        projector = [&](const EigenMatrix_& scaled_rotation) -> void {
            project_matrix_transposed_tatami(mat, output.components, scaled_rotation, options.num_threads);
        };

    } else if (mat.sparse()) {
        // 'extracted' contains row-major contents... but we implicitly transpose it to CSC with genes in columns.
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

        // Storing sparse_ptr in the unique pointer should not invalidate the former,
        // based on a reading of the C++ specification w.r.t. reset();
        // so we can continue to use it for projection.
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
        ptr.reset(sparse_ptr);

        compute_blockwise_mean_and_variance_realized_sparse(
            *sparse_ptr,
            block,
            num_blocks,
            block_sizes,
            block_details,
            output.center,
            scale,
            options.num_threads
        );

        // Make sure to copy sparse_ptr because it doesn't exist outside of this scope.
        projector = [&,sparse_ptr](const EigenMatrix_& scaled_rotation) -> void {
            project_matrix_realized_sparse<EigenVector_>(*sparse_ptr, output.components, scaled_rotation, options.num_threads);
        };

    } else {
        // Perform an implicit transposition by performing a row-major extraction into a column-major transposed matrix.
        auto tmp_ptr = std::make_unique<EigenMatrix_>(
            sanisizer::cast<I<decltype(std::declval<EigenMatrix_>().rows())> >(ncells),
            sanisizer::cast<I<decltype(std::declval<EigenMatrix_>().cols())> >(ngenes)
        ); 
        static_assert(!EigenMatrix_::IsRowMajor);

        tatami::convert_to_dense(
            mat,
            /* row_major = */ true,
            tmp_ptr->data(),
            [&]{
                tatami::ConvertToDenseOptions opt;
                opt.num_threads = options.num_threads;
                return opt;
            }()
        );

        compute_blockwise_mean_and_variance_realized_dense(
            *tmp_ptr,
            block,
            num_blocks,
            block_sizes,
            block_details,
            output.center,
            scale,
            options.num_threads
        );

        const auto dense_ptr = tmp_ptr.get(); // do this before the move.
        ptr.reset(new irlba::SimpleMatrix<EigenVector_, EigenMatrix_, decltype(tmp_ptr)>(std::move(tmp_ptr)));

        // Make sure to copy dense_ptr because it doesn't exist outside of this scope.
        projector = [&,dense_ptr](const EigenMatrix_& scaled_rotation) -> void {
            output.components.noalias() = (*dense_ptr * scaled_rotation).adjoint();
        };
    }

    output.total_variance = process_scale_vector(options.scale, scale);

    std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > alt;
    alt.reset(
        new ResidualMatrix<
            EigenVector_,
            EigenMatrix_,
            I<decltype(ptr)>,
            Block_,
            I<decltype(&(output.center))>
        >(
            std::move(ptr),
            block,
            &(output.center)
        )
    );
    ptr.swap(alt);

    if (options.scale) {
        alt.reset(
            new irlba::ScaledMatrix<
                EigenVector_,
                EigenMatrix_,
                I<decltype(ptr)>,
                I<decltype(&(scale))>
            >(
                std::move(ptr),
                &(scale),
                /* column = */ true,
                /* divide = */ true
            )
        );
        ptr.swap(alt);
    }

    if (block_details.has_value()) {
        alt.reset(
            new irlba::ScaledMatrix<
                EigenVector_,
                EigenMatrix_,
                I<decltype(ptr)>,
                I<decltype(&(block_details->expanded_weights))>
            >(
                std::move(ptr),
                &(block_details->expanded_weights),
                /* column = */ false,
                /* divide = */ false
            )
        );
        ptr.swap(alt);

        output.metrics = irlba::compute(*ptr, options.number, output.components, output.rotation, output.variance_explained, options.irlba_options);
        subset_fun(num_blocks, block_sizes, block_details, output.components, output.variance_explained);

        EigenMatrix_ tmp;
        const auto& scaled_rotation = scale_rotation_matrix(output.rotation, options.scale, scale, tmp);
        projector(scaled_rotation);

        // Subtracting each block's mean from the PCs.
        if (options.center_scores_by_block) {
            EigenMatrix_ centering = (output.center * scaled_rotation).adjoint();
            for (I<decltype(ncells)> c =0 ; c < ncells; ++c) {
                output.components.col(c) -= centering.col(block[c]);
            }
        }

        clean_up_projected(output.components, output.variance_explained);
        if (!options.transpose) {
            output.components.adjointInPlace();
        }

    } else {
        output.metrics = irlba::compute(*ptr, options.number, output.components, output.rotation, output.variance_explained, options.irlba_options);
        subset_fun(num_blocks, block_sizes, block_details, output.components, output.variance_explained);

        if (options.center_scores_by_block) {
            clean_up(mat.ncol(), output.components, output.variance_explained);
            if (options.transpose) {
                output.components.adjointInPlace();
            }

        } else {
            EigenMatrix_ tmp;
            const auto& scaled_rotation = scale_rotation_matrix(output.rotation, options.scale, scale, tmp);
            projector(scaled_rotation);

            clean_up_projected(output.components, output.variance_explained);
            if (!options.transpose) {
                output.components.adjointInPlace();
            }
        }
    }

    if (options.scale) {
        output.scale = std::move(scale);
    }
}
/**
 * @endcond
 */

/**
 * Principal components analysis on residuals, after regressing out a blocking factor across cells.
 *
 * As discussed in `simple_pca()`, we extract the top PCs from a single-cell dataset for downstream cell-based procedures like clustering.
 * In the presence of a blocking factor (e.g., batches, samples), we want to ensure that the PCA is not driven by uninteresting differences between blocks of cells.
 * To achieve this, `blocked_pca()` centers the expression of each gene within each blocking level and uses the residuals for PCA.
 * This ensures that the gene-gene covariance matrix will only contain variation within each batch, 
 * such that the top rotation vectors/principal components capture biological heterogeneity instead of inter-block differences.
 *
 * In addition, `blocked_pca()` will weight each block of cells to control its relative contribution to the PCA.
 * By default, `blocked_pca()` scales the expression values for each block so that each "sufficiently large" block contributes equally to the gene-gene covariance matrix and thus the rotation vectors.
 * This ensures that the definition of the axes of maximum variance are not dominated by the largest block, potentially masking interesting variation in the smaller blocks.
 * (See `BlockedPcaOptions::block_weight_policy` for the choice of weighting scheme.)
 *
 * The PC scores themselves are computed by projecting each cell's expression profile onto the subspace defined by the rotation vectors,
 * and then centering them according to `BlockedPcaOptions::center_scores_by_block`.
 * The interpretation of these scores depends on the choice of centering mode:
 *
 * - If `false` (the default), the dataset is globally shifted so that the centroid across all cells lies at the origin.
 *   This does not explicitly remove differences between blocks.
 *   Any differences in expression that are not orthogonal to the rotation vectors will still manifest in the PC scores.
 *   In this mode, blocking only reduces the impact of inter-block differences on the identification of the rotation vectors.
 * - If `true`, the scores are centered within each block, i.e., each block of cells is centered at the origin.
 *   Without weighting, this is equivalent to the PC scores that would be obtained from PCA on the residuals.
 *   This represents a low-dimensional space where inter-block differences have been "corrected",
 *   assuming that all blocks have the same subpopulation composition and the inter-block differences are consistent for all cell subpopulations.
 *
 * We default to `false` as the assumptions mentioned above for `true` are usually too strong.
 * Per-block centering can distort the differences between blocks when these assumptions are violated, even in the absence of any differences between blocks.
 * Global centering avoids any distortion while mitigating the impact of uninteresting inter-block differences on the scores.
 * Any remaining differences can be corrected by processing the scores with more sophisticated batch correction methods like [MNN correction](https://github.com/libscran/mnncorrect). 
 *
 * Internally, `blocked_pca()` defers the residual calculation until the matrix multiplication steps within [IRLBA](https://github.com/LTLA/CppIrlba).
 * This yields the same results as the naive calculation of residuals but is much faster as it can take advantage of efficient sparse operations.
 *
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type of the indices.
 * @tparam Block_ Integer type of the blocking factor.
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 *
 * @param[in] mat Input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param[in] block Pointer to an array of length equal to the number of cells, 
 * containing the block assignment for each cell. 
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options.
 * @param[out] output On output, the results of the PCA on the residuals. 
 * This can be re-used across multiple calls to `blocked_pca()`. 
 */
template<typename Value_, typename Index_, typename Block_, typename EigenMatrix_, class EigenVector_>
void blocked_pca(
    const tatami::Matrix<Value_, Index_>& mat,
    const Block_* block,
    const std::size_t num_blocks,
    const BlockedPcaOptions<EigenVector_>& options,
    BlockedPcaResults<EigenMatrix_, EigenVector_>& output
) {
    blocked_pca_internal<Value_, Index_, Block_, EigenMatrix_, EigenVector_>(
        mat,
        block,
        num_blocks,
        options,
        output,
        [&](
            const std::size_t, 
            const std::vector<Index_>&,
            const std::optional<BlockingDetails<EigenVector_> >&,
            const EigenMatrix_&,
            const EigenVector_&
        ) -> void {}
    );
}

/**
 * Overload of `blocked_pca()` that allocates memory for the output.
 *
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type of the indices.
 * @tparam Block_ Integer type of the blocking factor.
 *
 * @param[in] mat Input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param[in] block Pointer to an array of length equal to the number of cells, 
 * containing the block assignment for each cell. 
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options.
 *
 * @return Results of the PCA on the residuals. 
 */
template<typename EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, typename Value_, typename Index_, typename Block_>
BlockedPcaResults<EigenMatrix_, EigenVector_> blocked_pca(
    const tatami::Matrix<Value_, Index_>& mat,
    const Block_* block,
    const std::size_t num_blocks,
    const BlockedPcaOptions<EigenVector_>& options
) {
    BlockedPcaResults<EigenMatrix_, EigenVector_> output;
    blocked_pca(mat, block, num_blocks, options, output);
    return output;
}

}

#endif
