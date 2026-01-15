#ifndef SCRAN_PCA_BLOCKED_PCA_HPP
#define SCRAN_PCA_BLOCKED_PCA_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <cstddef>
#include <functional>

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
 */
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
     * Whether to compute the principal components from the residuals.
     * If `false`, only the rotation vector is computed from the residuals and the original expression values are projected onto the new axes. 
     * This avoids strong assumptions about the nature of the differences between blocks as discussed in `blocked_pca()`.
     */
    bool components_from_residuals = true;

    /**
     * Whether to realize `tatami::Matrix` objects into an appropriate in-memory format before PCA.
     * This is typically faster but increases memory usage.
     */
    bool realize_matrix = true;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()` and `irlba::parallelize()`.
     */
    int num_threads = 1;

    /**
     * Further options to pass to `irlba::compute()`.
     */
    irlba::Options<Eigen::VectorXd> irlba_options;
};

/**
 * @cond
 */
/*****************************************************
 ************* Blocking data structures **************
 *****************************************************/

template<typename Index_, class EigenVector_>
struct BlockingDetails {
    std::vector<Index_> block_size;

    bool weighted = false;
    typedef typename EigenVector_::Scalar Weight;

    // The below should only be used if weighted = true.
    std::vector<Weight> per_element_weight;
    Weight total_block_weight = 0;
    EigenVector_ expanded_weights;
};

template<class EigenVector_, typename Index_, typename Block_>
BlockingDetails<Index_, EigenVector_> compute_blocking_details(
    const Index_ ncells,
    const Block_* block,
    const scran_blocks::WeightPolicy block_weight_policy, 
    const scran_blocks::VariableWeightParameters& variable_block_weight_parameters) 
{
    BlockingDetails<Index_, EigenVector_> output;
    output.block_size = tatami_stats::tabulate_groups(block, ncells);
    if (block_weight_policy == scran_blocks::WeightPolicy::NONE) {
        return output;
    }

    const auto& block_size = output.block_size;
    const auto nblocks = block_size.size();
    output.weighted = true;
    auto& total_weight = output.total_block_weight;
    auto& element_weight = output.per_element_weight;
    sanisizer::resize(element_weight, nblocks);

    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        const auto bsize = block_size[b];

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
    sanisizer::resize(expanded, ncells);
    for (Index_ c = 0; c < ncells; ++c) {
        expanded.coeffRef(c) = sqrt_weights[block[c]];
    }

    return output;
}

/*****************************************************************
 ************ Computing the blockwise mean and variance **********
 *****************************************************************/

template<typename Num_, typename Value_, typename Index_, typename Block_, typename EigenVector_, typename Float_>
void compute_sparse_mean_and_variance_blocked(
    const Num_ num_nonzero, 
    const Value_* values, 
    const Index_* indices, 
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details,
    Float_* centers,
    Float_& variance,
    std::vector<Index_>& block_copy,
    const Num_ num_all)
{
    const auto& block_size = block_details.block_size;
    const auto nblocks = block_size.size();

    std::fill_n(centers, nblocks, 0);
    for (Num_ i = 0; i < num_nonzero; ++i) {
        centers[block[indices[i]]] += values[i];
    }
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        auto bsize = block_size[b];
        if (bsize) {
            centers[b] /= bsize;
        }
    }

    // Computing the variance from the sum of squared differences.
    // This is technically not the correct variance estimate if we
    // were to consider the loss of residual d.f. from estimating
    // the block means, but it's what the PCA sees, so whatever.
    variance = 0;
    std::copy(block_size.begin(), block_size.end(), block_copy.begin());

    if (block_details.weighted) {
        for (Num_ i = 0; i < num_nonzero; ++i) {
            const Block_ curb = block[indices[i]];
            const auto diff = values[i] - centers[curb];
            variance += diff * diff * block_details.per_element_weight[curb];
            --block_copy[curb];
        }
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            const auto val = centers[b];
            variance += val * val * block_copy[b] * block_details.per_element_weight[b];
        }
    } else {
        for (Num_ i = 0; i < num_nonzero; ++i) {
            const Block_ curb = block[indices[i]];
            const auto diff = values[i] - centers[curb];
            variance += diff * diff;
            --block_copy[curb];
        }
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            const auto val = centers[b];
            variance += val * val * block_copy[b];
        }
    }

    // COMMENT ON DENOMINATOR:
    // If we're not dealing with weights, we compute the actual sample
    // variance for easy interpretation (and to match up with the
    // per-PC calculations in clean_up).
    //
    // If we're dealing with weights, the concept of the sample variance
    // becomes somewhat weird, but we just use the same denominator for
    // consistency in clean_up_projected. Magnitude doesn't matter when
    // scaling for process_scale_vector anyway.
    variance /= num_all - 1;
}

template<class IrlbaSparseMatrix_, typename Block_, class Index_, class EigenVector_, class EigenMatrix_>
void compute_blockwise_mean_and_variance_realized_sparse(
    const IrlbaSparseMatrix_& emat, // this should be column-major with genes in the columns.
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details,
    EigenMatrix_& centers,
    EigenVector_& variances,
    const int nthreads) 
{
    const auto ngenes = emat.cols();
    tatami::parallelize([&](const int, const I<decltype(ngenes)> start, const I<decltype(ngenes)> length) -> void {
        const auto ncells = emat.rows();
        const auto& values = emat.get_values();
        const auto& indices = emat.get_indices();
        const auto& pointers = emat.get_pointers();

        const auto nblocks = block_details.block_size.size();
        static_assert(!EigenMatrix_::IsRowMajor);
        auto block_copy = sanisizer::create<std::vector<Index_> >(nblocks);

        for (I<decltype(start)> g = start, end = start + length; g < end; ++g) {
            const auto offset = pointers[g];
            const auto next_offset = pointers[g + 1]; // increment won't overflow as 'g < end' and 'end' is of the same type. 
            compute_sparse_mean_and_variance_blocked(
                static_cast<I<decltype(ncells)> >(next_offset - offset),
                values.data() + offset,
                indices.data() + offset,
                block,
                block_details,
                centers.data() + sanisizer::product_unsafe<std::size_t>(g, nblocks),
                variances[g],
                block_copy,
                ncells
            );
        }
    }, ngenes, nthreads);
}

template<typename Num_, typename Value_, typename Block_, typename Index_, typename EigenVector_, typename Float_>
void compute_dense_mean_and_variance_blocked(
    const Num_ number, 
    const Value_* values, 
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details,
    Float_* centers,
    Float_& variance) 
{
    const auto& block_size = block_details.block_size;
    const auto nblocks = block_size.size();
    std::fill_n(centers, nblocks, 0);
    for (Num_ i = 0; i < number; ++i) {
        centers[block[i]] += values[i];
    }
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        const auto& bsize = block_size[b];
        if (bsize) {
            centers[b] /= bsize;
        }
    }

    variance = 0;

    if (block_details.weighted) {
        for (Num_ i = 0; i < number; ++i) {
            const auto curb = block[i];
            const auto delta = values[i] - centers[curb];
            variance += delta * delta * block_details.per_element_weight[curb];
        }
    } else {
        for (Num_ i = 0; i < number; ++i) {
            const auto curb = block[i];
            const auto delta = values[i] - centers[curb];
            variance += delta * delta;
        }
    }

    variance /= number - 1; // See COMMENT ON DENOMINATOR above.
}

template<class EigenMatrix_, typename Block_, class Index_, class EigenVector_>
void compute_blockwise_mean_and_variance_realized_dense(
    const EigenMatrix_& emat, // this should be column-major with genes in the columns.
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details,
    EigenMatrix_& centers,
    EigenVector_& variances,
    const int nthreads) 
{
    const auto ngenes = emat.cols();
    tatami::parallelize([&](const int, const I<decltype(ngenes)> start, const I<decltype(ngenes)> length) -> void {
        const auto ncells = emat.rows();
        static_assert(!EigenMatrix_::IsRowMajor);
        const auto nblocks = block_details.block_size.size();
        for (I<decltype(start)> g = start, end = start + length; g < end; ++g) {
            compute_dense_mean_and_variance_blocked(
                ncells,
                emat.data() + sanisizer::product_unsafe<std::size_t>(g, ncells),
                block,
                block_details,
                centers.data() + sanisizer::product_unsafe<std::size_t>(g, nblocks),
                variances[g]
            );
        }
    }, ngenes, nthreads);
}

template<typename Value_, typename Index_, typename Block_, class EigenMatrix_, class EigenVector_>
void compute_blockwise_mean_and_variance_tatami(
    const tatami::Matrix<Value_, Index_>& mat, // this should have genes in the rows!
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details,
    EigenMatrix_& centers,
    EigenVector_& variances,
    const int nthreads) 
{
    const auto& block_size = block_details.block_size;
    const auto nblocks = block_size.size();
    const Index_ ngenes = mat.nrow();
    const Index_ ncells = mat.ncol();

    if (mat.prefer_rows()) {
        tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
            static_assert(!EigenMatrix_::IsRowMajor);
            auto block_copy = sanisizer::create<std::vector<Index_> >(nblocks);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(ncells);

            if (mat.is_sparse()) {
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(ncells);
                auto ext = tatami::consecutive_extractor<true>(mat, true, start, length);
                for (Index_ g = start, end = start + length; g < end; ++g) {
                    auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    compute_sparse_mean_and_variance_blocked(
                        range.number,
                        range.value,
                        range.index,
                        block,
                        block_details,
                        centers.data() + sanisizer::product_unsafe<std::size_t>(g, nblocks),
                        variances[g],
                        block_copy,
                        ncells
                    );
                }

            } else {
                auto ext = tatami::consecutive_extractor<false>(mat, true, start, length);
                for (Index_ g = start, end = start + length; g < end; ++g) {
                    auto ptr = ext->fetch(vbuffer.data());
                    compute_dense_mean_and_variance_blocked(
                        ncells,
                        ptr,
                        block,
                        block_details,
                        centers.data() + sanisizer::product_unsafe<std::size_t>(g, nblocks),
                        variances[g]
                    );
                }
            }
        }, ngenes, nthreads);

    } else {
        typedef typename EigenVector_::Scalar Scalar;
        std::vector<std::pair<I<decltype(nblocks)>, Scalar> > block_multipliers;
        block_multipliers.reserve(nblocks);

        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            const auto bsize = block_size[b];
            if (bsize > 1) { // skipping blocks with NaN variances.
                Scalar mult = bsize - 1; // need to convert variances back into sum of squared differences.
                if (block_details.weighted) {
                    mult *= block_details.per_element_weight[b];
                }
                block_multipliers.emplace_back(b, mult);
            }
        }

        tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
            std::vector<std::vector<Scalar> > re_centers, re_variances;
            re_centers.reserve(nblocks);
            re_variances.reserve(nblocks);
            for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                re_centers.emplace_back(length);
                re_variances.emplace_back(length);
            }

            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);

            if (mat.is_sparse()) {
                std::vector<tatami_stats::variances::RunningSparse<Scalar, Value_, Index_> > running;
                running.reserve(nblocks);
                for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                    running.emplace_back(length, re_centers[b].data(), re_variances[b].data(), /* skip_nan = */ false, /* subtract = */ start);
                }

                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);
                auto ext = tatami::consecutive_extractor<true>(mat, false, static_cast<Index_>(0), ncells, start, length);
                for (Index_ c = 0; c < ncells; ++c) {
                    const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    running[block[c]].add(range.value, range.index, range.number);
                }

                for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                    running[b].finish();
                }

            } else {
                std::vector<tatami_stats::variances::RunningDense<Scalar, Value_, Index_> > running;
                running.reserve(nblocks);
                for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                    running.emplace_back(length, re_centers[b].data(), re_variances[b].data(), /* skip_nan = */ false);
                }

                auto ext = tatami::consecutive_extractor<false>(mat, false, static_cast<Index_>(0), ncells, start, length);
                for (Index_ c = 0; c < ncells; ++c) {
                    auto ptr = ext->fetch(vbuffer.data());
                    running[block[c]].add(ptr);
                }

                for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                    running[b].finish();
                }
            }

            static_assert(!EigenMatrix_::IsRowMajor);
            for (Index_ i = 0; i < length; ++i) {
                auto mptr = centers.data() + sanisizer::product_unsafe<std::size_t>(start + i, nblocks);
                for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                    mptr[b] = re_centers[b][i];
                }

                auto& my_var = variances[start + i];
                my_var = 0;
                for (const auto& bm : block_multipliers) {
                    my_var += re_variances[bm.first][i] * bm.second;
                }
                my_var /= ncells - 1; // See COMMENT ON DENOMINATOR above.
            }
        }, ngenes, nthreads);
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

template<class IrlbaSparseMatrix_, class EigenMatrix_>
inline void project_matrix_realized_sparse(
    const IrlbaSparseMatrix_& emat, // cell in rows, genes in the columns, CSC.
    EigenMatrix_& components, // dims in rows, cells in columns
    const EigenMatrix_& scaled_rotation, // genes in rows, dims in columns
    int nthreads) 
{
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

    if (nthreads == 1) {
        const auto& pointers = emat.get_pointers();
        auto multipliers = sanisizer::create<Eigen::VectorXd>(rank);
        for (I<decltype(ngenes)> g = 0; g < ngenes; ++g) {
            multipliers.noalias() = scaled_rotation.row(g);
            const auto start = pointers[g], end = pointers[g + 1]; // increment is safe as 'g + 1 <= ngenes'.
            for (auto i = start; i < end; ++i) {
                components.col(indices[i]).noalias() += values[i] * multipliers;
            }
        }

    } else {
        const auto& row_nonzero_bounds = emat.get_secondary_nonzero_boundaries();
        irlba::parallelize(nthreads, [&](const int t) -> void { 
            const auto& starts = row_nonzero_bounds[t];
            const auto& ends = row_nonzero_bounds[t + 1]; // increment is safe as 't + 1 <= nthreads'.
            auto multipliers = sanisizer::create<Eigen::VectorXd>(rank);

            for (I<decltype(ngenes)> g = 0; g < ngenes; ++g) {
                multipliers.noalias() = scaled_rotation.row(g);
                const auto start = starts[g], end = ends[g];
                for (auto i = start; i < end; ++i) {
                    components.col(indices[i]).noalias() += values[i] * multipliers;
                }
            }
        });
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
    typedef typename EigenMatrix_::Scalar Scalar;

    // Store as transposed for more cache efficiency.
    components.resize(
        sanisizer::cast<I<decltype(components.rows())> >(rank),
        sanisizer::cast<I<decltype(components.cols())> >(ncells)
    );

    if (mat.prefer_rows()) {
        tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
            static_assert(!EigenMatrix_::IsRowMajor);
            const auto vptr = scaled_rotation.data();
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);

            std::vector<std::vector<Scalar> > local_buffers; // create separate buffers to avoid false sharing.
            local_buffers.reserve(rank);
            for (I<decltype(rank)> r = 0; r < rank; ++r) {
                local_buffers.emplace_back(tatami::cast_Index_to_container_size<I<decltype(local_buffers.front())> >(length));
            }

            if (mat.is_sparse()) {
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);
                auto ext = tatami::consecutive_extractor<true>(mat, true, static_cast<Index_>(0), ngenes, start, length);
                for (Index_ g = 0; g < ngenes; ++g) {
                    const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    for (I<decltype(rank)> r = 0; r < rank; ++r) {
                        const auto mult = vptr[sanisizer::nd_offset<std::size_t>(g, ngenes, r)];
                        auto& local_buffer = local_buffers[r];
                        for (Index_ i = 0; i < range.number; ++i) {
                            local_buffer[range.index[i] - start] += range.value[i] * mult;
                        }
                    }
                }

            } else {
                auto ext = tatami::consecutive_extractor<false>(mat, true, static_cast<Index_>(0), ngenes, start, length);
                for (Index_ g = 0; g < ngenes; ++g) {
                    const auto ptr = ext->fetch(vbuffer.data());
                    for (I<decltype(rank)> r = 0; r < rank; ++r) {
                        const auto mult = vptr[sanisizer::nd_offset<std::size_t>(g, ngenes, r)];
                        auto& local_buffer = local_buffers[r];
                        for (Index_ i = 0; i < length; ++i) {
                            local_buffer[i] += ptr[i] * mult;
                        }
                    }
                }
            }

            for (I<decltype(rank)> r = 0; r < rank; ++r) {
                for (Index_ c = 0; c < length; ++c) {
                    components.coeffRef(r, c + start) = local_buffers[r][c];
                }
            }

        }, ncells, nthreads);

    } else {
        tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
            static_assert(!EigenMatrix_::IsRowMajor);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(ngenes);

            if (mat.is_sparse()) {
                std::vector<Index_> ibuffer(ngenes);
                auto ext = tatami::consecutive_extractor<true>(mat, false, start, length);

                for (Index_ c = start, end = start + length; c < end; ++c) {
                    const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    static_assert(!EigenMatrix_::IsRowMajor);
                    for (I<decltype(rank)> r = 0; r < rank; ++r) {
                        auto& output = components.coeffRef(r, c);
                        output = 0;
                        const auto rotptr = scaled_rotation.data() + sanisizer::product_unsafe<std::size_t>(r, ngenes);
                        for (Index_ i = 0; i < range.number; ++i) {
                            output += rotptr[range.index[i]] * range.value[i];
                        }
                    }
                }

            } else {
                auto ext = tatami::consecutive_extractor<false>(mat, false, start, length);
                for (Index_ c = start, end = start + length; c < end; ++c) {
                    const auto ptr = ext->fetch(vbuffer.data()); 
                    static_assert(!EigenMatrix_::IsRowMajor);
                    for (I<decltype(rank)> r = 0; r < rank; ++r) {
                        const auto rotptr = scaled_rotation.data() + sanisizer::product_unsafe<std::size_t>(r, ngenes);
                        components.coeffRef(r, c) = std::inner_product(rotptr, rotptr + ngenes, ptr, static_cast<Scalar>(0));
                    }
                }
            }
        }, ncells, nthreads);
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
    for (auto& d : D) {
        d = d * d / denom;
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
     * Each row corresponds to a gene while each column corresponds to a PC.
     * The number of PCs is as described for `BlockedPcaResults::components`.
     */
    EigenMatrix_ rotation;

    /**
     * Centering matrix.
     * Each row corresponds to a block and each column corresponds to a gene.
     * Each entry contains the mean for a particular gene in the corresponding block.
     */
    EigenMatrix_ center;

    /**
     * Scaling vector, only returned if `BlockedPcaOptions::scale = true`.
     * Each entry corresponds to a row in the input matrix and contains the scaling factor used to divide that gene's values if `BlockedPcaOptions::scale = true`.
     */
    EigenVector_ scale;

    /**
     * Whether the algorithm converged.
     */
    bool converged = false;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, typename Block_, typename EigenMatrix_, class EigenVector_, class SubsetFunction_>
void blocked_pca_internal(
    const tatami::Matrix<Value_, Index_>& mat,
    const Block_* block,
    const BlockedPcaOptions& options,
    BlockedPcaResults<EigenMatrix_, EigenVector_>& output,
    SubsetFunction_ subset_fun
) {
    irlba::EigenThreadScope t(options.num_threads);
    const auto block_details = compute_blocking_details<EigenVector_>(mat.ncol(), block, options.block_weight_policy, options.variable_block_weight_parameters);

    const Index_ ngenes = mat.nrow(), ncells = mat.ncol(); 
    const auto nblocks = block_details.block_size.size();
    output.center.resize(
        sanisizer::cast<I<decltype(output.center.rows())> >(nblocks),
        sanisizer::cast<I<decltype(output.center.cols())> >(ngenes)
    );
    sanisizer::resize(output.scale, ngenes);

    std::unique_ptr<irlba::Matrix<EigenVector_, EigenMatrix_> > ptr;
    std::function<void(const EigenMatrix_&)> projector;

    if (!options.realize_matrix) {
        ptr.reset(new irlba_tatami::Transposed<EigenVector_, EigenMatrix_, Value_, Index_, decltype(&mat)>(&mat, options.num_threads));
        compute_blockwise_mean_and_variance_tatami(mat, block, block_details, output.center, output.scale, options.num_threads);

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

        compute_blockwise_mean_and_variance_realized_sparse(*sparse_ptr, block, block_details, output.center, output.scale, options.num_threads);

        // Make sure to copy sparse_ptr because it doesn't exist outside of this scope.
        projector = [&,sparse_ptr](const EigenMatrix_& scaled_rotation) -> void {
            project_matrix_realized_sparse(*sparse_ptr, output.components, scaled_rotation, options.num_threads);
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

        compute_blockwise_mean_and_variance_realized_dense(*tmp_ptr, block, block_details, output.center, output.scale, options.num_threads);
        const auto dense_ptr = tmp_ptr.get(); // do this before the move.
        ptr.reset(new irlba::SimpleMatrix<EigenVector_, EigenMatrix_, decltype(tmp_ptr)>(std::move(tmp_ptr)));

        // Make sure to copy dense_ptr because it doesn't exist outside of this scope.
        projector = [&,dense_ptr](const EigenMatrix_& scaled_rotation) -> void {
            output.components.noalias() = (*dense_ptr * scaled_rotation).adjoint();
        };
    }

    output.total_variance = process_scale_vector(options.scale, output.scale);

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
                I<decltype(&(output.scale))>
            >(
                std::move(ptr),
                &(output.scale),
                /* column = */ true,
                /* divide = */ true
            )
        );
        ptr.swap(alt);
    }

    if (block_details.weighted) {
        alt.reset(
            new irlba::ScaledMatrix<
                EigenVector_,
                EigenMatrix_,
                I<decltype(ptr)>,
                I<decltype(&(block_details.expanded_weights))>
            >(
                std::move(ptr),
                &(block_details.expanded_weights),
                /* column = */ false,
                /* divide = */ false
            )
        );
        ptr.swap(alt);

        auto out = irlba::compute(*ptr, options.number, output.components, output.rotation, output.variance_explained, options.irlba_options);
        output.converged = out.first;

        subset_fun(block_details, output.components, output.variance_explained);

        EigenMatrix_ tmp;
        const auto& scaled_rotation = scale_rotation_matrix(output.rotation, options.scale, output.scale, tmp);
        projector(scaled_rotation);

        // Subtracting each block's mean from the PCs.
        if (options.components_from_residuals) {
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
        const auto out = irlba::compute(*ptr, options.number, output.components, output.rotation, output.variance_explained, options.irlba_options);
        output.converged = out.first;

        subset_fun(block_details, output.components, output.variance_explained);

        if (options.components_from_residuals) {
            clean_up(mat.ncol(), output.components, output.variance_explained);
            if (options.transpose) {
                output.components.adjointInPlace();
            }

        } else {
            EigenMatrix_ tmp;
            const auto& scaled_rotation = scale_rotation_matrix(output.rotation, options.scale, output.scale, tmp);
            projector(scaled_rotation);

            clean_up_projected(output.components, output.variance_explained);
            if (!options.transpose) {
                output.components.adjointInPlace();
            }
        }
    }

    if (!options.scale) {
        output.scale = EigenVector_();
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
 * To achieve this, `blocked_pca()` centers the expression of each gene in each blocking level and uses the residuals for PCA.
 * This means that the gene-gene covariance matrix will only contain variation within each batch, 
 * ensuring that the top rotation vectors/principal components capture biological heterogeneity instead of inter-block differences.
 *
 * The `BlockedPcaOptions::components_from_residuals` option determines exactly how the PC scores are calculated:
 *
 * - If `true` (the default), the PC scores are computed from the matrix of residuals.
 *   This yields a low-dimensional space where inter-block differences have been removed,
 *   assuming that all blocks have the same subpopulation composition and the inter-block differences are consistent for all cell subpopulations.
 *   Under these assumptions, we could use these components for downstream analysis without any concern for block-wise effects.
 * - If `false`, the rotation vectors are first computed from the matrix of residuals.
 *   To obtain PC scores, each cell is then projected onto the associated subspace using its original expression values.
 *   This approach ensures that inter-block differences do not contribute to the PCA but does not attempt to explicitly remove them.
 * 
 * In complex datasets, the assumptions mentioned above for `true` do not hold,
 * and more sophisticated batch correction methods like [MNN correction](https://github.com/libscran/mnncorrect) are required.
 * Some of these methods accept a low-dimensional embedding of cells that can be created as described above with `BlockedPcaOptions::components_from_residuals = false`. 
 *
 * `blocked_pca()` will weight the contribution from blocks of cells so that each block contributes more or less equally to the PCA.
 * This ensures that the definition of the axes of maximum variance are not dominated by the largest block, potentially masking interesting variation in the smaller blocks.
 * `blocked_pca()` scales the expression values for each block so that each "sufficiently large" block contributes equally to the gene-gene covariance matrix and thus the rotation vectors.
 * (See `BlockedPcaOptions::block_weight_policy` for the choice of weighting scheme.)
 * The vector of residuals for each cell - or the original expression values, if `BlockedPcaOptions::components_from_residuals = false` -
 * is then projected to the subspace defined by these rotation vectors to obtain that cell's PC scores.
 *
 * Internally, `blocked_pca()` defers the residual calculation until the matrix multiplication steps within [IRLBA](https://github.com/LTLA/CppIrlba).
 * This yields the same results as the naive calculation of residuals but is much faster as it can take advantage of efficient sparse operations.
 *
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 * @tparam Block_ Integer type for the blocking factor.
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
    const BlockedPcaOptions& options,
    BlockedPcaResults<EigenMatrix_, EigenVector_>& output
) {
    blocked_pca_internal<Value_, Index_, Block_, EigenMatrix_, EigenVector_>(
        mat,
        block,
        options,
        output,
        [&](const BlockingDetails<Index_, EigenVector_>&, const EigenMatrix_&, const EigenVector_&) -> void {}
    );
}

/**
 * Overload of `blocked_pca()` that allocates memory for the output.
 *
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 * @tparam Block_ Integer type for the blocking factor.
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
BlockedPcaResults<EigenMatrix_, EigenVector_> blocked_pca(const tatami::Matrix<Value_, Index_>& mat, const Block_* block, const BlockedPcaOptions& options) {
    BlockedPcaResults<EigenMatrix_, EigenVector_> output;
    blocked_pca(mat, block, options, output);
    return output;
}

}

#endif
