#ifndef SCRAN_PCA_BLOCKED_PCA_HPP
#define SCRAN_PCA_BLOCKED_PCA_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "irlba/irlba.hpp"
#include "irlba/parallel.hpp"
#include "Eigen/Dense"
#include "scran_blocks/scran_blocks.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file blocked_pca.hpp
 *
 * @brief Perform PCA on residuals after regressing out a blocking factor.
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
        irlba_options.cap_number = true;
    }
    /**
     * @endcond
     */

    /**
     * Number of principal components (PCs) to compute.
     * This should be no greater than the maximum number of PCs, i.e., the smaller dimension of the input matrix, otherwise an error will be thrown.
     * (This error can be avoided by setting `irlba::Options::cap_number = true` in `BlockedPcaOptions::irlba_options`, in which case only the maximum number of PCs will be reported in the results.)
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
     * Policy to use for weighting batches of different size.
     */
    scran_blocks::WeightPolicy block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;

    /**
     * Parameters for the variable block weights.
     * Only used when `BlockedPcaOptions::block_weight_policy = scran_blocks::WeightPolicy::VARIABLE`.
     */
    scran_blocks::VariableWeightParameters variable_block_weight_parameters;

    /**
     * Compute the principal components from the residuals.
     * If false, only the rotation vector is computed from the residuals,
     * and the original expression values are projected onto the new axes. 
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
    irlba::Options irlba_options;
};

/**
 * @cond
 */
namespace internal {

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
    Index_ ncells,
    const Block_* block,
    scran_blocks::WeightPolicy block_weight_policy, 
    const scran_blocks::VariableWeightParameters& variable_block_weight_parameters) 
{
    BlockingDetails<Index_, EigenVector_> output;
    output.block_size = tatami_stats::tabulate_groups(block, ncells);
    if (block_weight_policy == scran_blocks::WeightPolicy::NONE) {
        return output;
    }

    const auto& block_size = output.block_size;
    auto nblocks = block_size.size();
    output.weighted = true;
    auto& total_weight = output.total_block_weight;
    auto& element_weight = output.per_element_weight;
    element_weight.resize(sanisizer::cast<decltype(element_weight.size())>(nblocks));

    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        auto bsize = block_size[b];

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
    expanded.resize(sanisizer::cast<decltype(expanded.size())>(ncells));
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
    Num_ num_nonzero, 
    const Value_* values, 
    const Index_* indices, 
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details,
    Float_* centers,
    Float_& variance,
    std::vector<Index_>& block_copy,
    Num_ num_all)
{
    const auto& block_size = block_details.block_size;
    auto nblocks = block_size.size();

    std::fill_n(centers, nblocks, 0);
    for (Num_ i = 0; i < num_nonzero; ++i) {
        centers[block[indices[i]]] += values[i];
    }
    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
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
            Block_ curb = block[indices[i]];
            auto diff = values[i] - centers[curb];
            variance += diff * diff * block_details.per_element_weight[curb];
            --block_copy[curb];
        }
        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
            auto val = centers[b];
            variance += val * val * block_copy[b] * block_details.per_element_weight[b];
        }
    } else {
        for (Num_ i = 0; i < num_nonzero; ++i) {
            Block_ curb = block[indices[i]];
            auto diff = values[i] - centers[curb];
            variance += diff * diff;
            --block_copy[curb];
        }
        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
            auto val = centers[b];
            variance += val * val * block_copy[b];
        }
    }

    // COMMENT ON DENOMINATOR:
    // If we're not dealing with weights, we compute the actual sample
    // variance for easy interpretation (and to match up with the
    // per-PC calculations in internal::clean_up).
    //
    // If we're dealing with weights, the concept of the sample variance
    // becomes somewhat weird, but we just use the same denominator for
    // consistency in clean_up_projected. Magnitude doesn't matter when
    // scaling for internal::process_scale_vector anyway.
    variance /= num_all - 1;
}

template<class IrlbaSparseMatrix_, typename Block_, class Index_, class EigenVector_, class EigenMatrix_>
void compute_blockwise_mean_and_variance_realized_sparse(
    const IrlbaSparseMatrix_& emat, // this should be column-major with genes in the columns.
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details,
    EigenMatrix_& centers,
    EigenVector_& variances,
    int nthreads) 
{
    auto ngenes = emat.cols();
    tatami::parallelize([&](int, decltype(ngenes) start, decltype(ngenes) length) -> void {
        auto ncells = emat.rows();
        const auto& values = emat.get_values();
        const auto& indices = emat.get_indices();
        const auto& pointers = emat.get_pointers();

        auto nblocks = block_details.block_size.size();
        static_assert(!EigenMatrix_::IsRowMajor);
        auto block_copy = sanisizer::create<std::vector<Index_> >(nblocks);

        for (auto g = start, end = start + length; g < end; ++g) {
            auto offset = pointers[g];
            auto next_offset = pointers[g + 1]; // increment won't overflow as 'g < end' and 'end' is of the same type. 
            compute_sparse_mean_and_variance_blocked(
                static_cast<decltype(ncells)>(next_offset - offset),
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
    Num_ number, 
    const Value_* values, 
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details,
    Float_* centers,
    Float_& variance) 
{
    const auto& block_size = block_details.block_size;
    auto nblocks = block_size.size();
    std::fill_n(centers, nblocks, 0);
    for (Num_ i = 0; i < number; ++i) {
        centers[block[i]] += values[i];
    }
    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        const auto& bsize = block_size[b];
        if (bsize) {
            centers[b] /= bsize;
        }
    }

    variance = 0;

    if (block_details.weighted) {
        for (Num_ i = 0; i < number; ++i) {
            auto curb = block[i];
            auto delta = values[i] - centers[curb];
            variance += delta * delta * block_details.per_element_weight[curb];
        }
    } else {
        for (Num_ i = 0; i < number; ++i) {
            auto curb = block[i];
            auto delta = values[i] - centers[curb];
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
    int nthreads) 
{
    auto ngenes = emat.cols();
    tatami::parallelize([&](int, decltype(ngenes) start, decltype(ngenes) length) -> void {
        auto ncells = emat.rows();
        static_assert(!EigenMatrix_::IsRowMajor);
        auto nblocks = block_details.block_size.size();
        for (auto g = start, end = start + length; g < end; ++g) {
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
    int nthreads) 
{
    const auto& block_size = block_details.block_size;
    auto nblocks = block_size.size();
    Index_ ngenes = mat.nrow();
    Index_ ncells = mat.ncol();

    if (mat.prefer_rows()) {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
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

        std::vector<std::pair<decltype(nblocks), Scalar> > block_multipliers;
        block_multipliers.reserve(nblocks);
        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
            auto bsize = block_size[b];
            if (bsize > 1) { // skipping blocks with NaN variances.
                Scalar mult = bsize - 1; // need to convert variances back into sum of squared differences.
                if (block_details.weighted) {
                    mult *= block_details.per_element_weight[b];
                }
                block_multipliers.emplace_back(b, mult);
            }
        }

        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            std::vector<std::vector<Scalar> > re_centers, re_variances;
            re_centers.reserve(nblocks);
            re_variances.reserve(nblocks);
            for (decltype(nblocks) b = 0; b < nblocks; ++b) {
                re_centers.emplace_back(length);
                re_variances.emplace_back(length);
            }

            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);

            if (mat.is_sparse()) {
                std::vector<tatami_stats::variances::RunningSparse<Scalar, Value_, Index_> > running;
                running.reserve(nblocks);
                for (decltype(nblocks) b = 0; b < nblocks; ++b) {
                    running.emplace_back(length, re_centers[b].data(), re_variances[b].data(), /* skip_nan = */ false, /* subtract = */ start);
                }

                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);
                auto ext = tatami::consecutive_extractor<true>(mat, false, static_cast<Index_>(0), ncells, start, length);
                for (Index_ c = 0; c < ncells; ++c) {
                    auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    running[block[c]].add(range.value, range.index, range.number);
                }

                for (decltype(nblocks) b = 0; b < nblocks; ++b) {
                    running[b].finish();
                }

            } else {
                std::vector<tatami_stats::variances::RunningDense<Scalar, Value_, Index_> > running;
                running.reserve(nblocks);
                for (decltype(nblocks) b = 0; b < nblocks; ++b) {
                    running.emplace_back(length, re_centers[b].data(), re_variances[b].data(), /* skip_nan = */ false);
                }

                auto ext = tatami::consecutive_extractor<false>(mat, false, static_cast<Index_>(0), ncells, start, length);
                for (Index_ c = 0; c < ncells; ++c) {
                    auto ptr = ext->fetch(vbuffer.data());
                    running[block[c]].add(ptr);
                }

                for (decltype(nblocks) b = 0; b < nblocks; ++b) {
                    running[b].finish();
                }
            }

            static_assert(!EigenMatrix_::IsRowMajor);
            for (Index_ i = 0; i < length; ++i) {
                auto mptr = centers.data() + sanisizer::product_unsafe<std::size_t>(start + i, nblocks);
                for (decltype(nblocks) b = 0; b < nblocks; ++b) {
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
    auto rank = scaled_rotation.cols();
    auto ncells = emat.rows();
    auto ngenes = emat.cols();

    // Store as transposed for more cache efficiency.
    components.resize(
        sanisizer::cast<decltype(components.rows())>(rank),
        sanisizer::cast<decltype(components.cols())>(ncells)
    );
    components.setZero();

    const auto& values = emat.get_values();
    const auto& indices = emat.get_indices();

    if (nthreads == 1) {
        const auto& pointers = emat.get_pointers();
        auto multipliers = sanisizer::create<Eigen::VectorXd>(rank);
        for (decltype(ngenes) g = 0; g < ngenes; ++g) {
            multipliers.noalias() = scaled_rotation.row(g);
            auto start = pointers[g], end = pointers[g + 1]; // increment is safe as 'g + 1 <= ngenes'.
            for (auto i = start; i < end; ++i) {
                components.col(indices[i]).noalias() += values[i] * multipliers;
            }
        }

    } else {
        const auto& row_nonzero_starts = emat.get_secondary_nonzero_starts();
        irlba::parallelize(nthreads, [&](int t) -> void { 
            const auto& starts = row_nonzero_starts[t];
            const auto& ends = row_nonzero_starts[t + 1]; // increment is safe as 't + 1 <= nthreads'.
            auto multipliers = sanisizer::create<Eigen::VectorXd>(rank);

            for (decltype(ngenes) g = 0; g < ngenes; ++g) {
                multipliers.noalias() = scaled_rotation.row(g);
                auto start = starts[g], end = ends[g];
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
    int nthreads) 
{
    auto rank = scaled_rotation.cols();
    auto ngenes = mat.nrow();
    auto ncells = mat.ncol();
    typedef typename EigenMatrix_::Scalar Scalar;

    // Store as transposed for more cache efficiency.
    components.resize(
        sanisizer::cast<decltype(components.rows())>(rank),
        sanisizer::cast<decltype(components.cols())>(ncells)
    );

    if (mat.prefer_rows()) {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            static_assert(!EigenMatrix_::IsRowMajor);
            auto vptr = scaled_rotation.data();
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);

            std::vector<std::vector<Scalar> > local_buffers; // create separate buffers to avoid false sharing.
            local_buffers.reserve(rank);
            for (decltype(rank) r = 0; r < rank; ++r) {
                local_buffers.emplace_back(tatami::cast_Index_to_container_size<decltype(local_buffers.front())>(length));
            }

            if (mat.is_sparse()) {
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);
                auto ext = tatami::consecutive_extractor<true>(mat, true, static_cast<Index_>(0), ngenes, start, length);
                for (Index_ g = 0; g < ngenes; ++g) {
                    auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    for (decltype(rank) r = 0; r < rank; ++r) {
                        auto mult = vptr[sanisizer::nd_offset<std::size_t>(g, ngenes, r)];
                        auto& local_buffer = local_buffers[r];
                        for (Index_ i = 0; i < range.number; ++i) {
                            local_buffer[range.index[i] - start] += range.value[i] * mult;
                        }
                    }
                }

            } else {
                auto ext = tatami::consecutive_extractor<false>(mat, true, static_cast<Index_>(0), ngenes, start, length);
                for (Index_ g = 0; g < ngenes; ++g) {
                    auto ptr = ext->fetch(vbuffer.data());
                    for (decltype(rank) r = 0; r < rank; ++r) {
                        auto mult = vptr[sanisizer::nd_offset<std::size_t>(g, ngenes, r)];
                        auto& local_buffer = local_buffers[r];
                        for (Index_ i = 0; i < length; ++i) {
                            local_buffer[i] += ptr[i] * mult;
                        }
                    }
                }
            }

            for (decltype(rank) r = 0; r < rank; ++r) {
                for (Index_ c = 0; c < length; ++c) {
                    components.coeffRef(r, c + start) = local_buffers[r][c];
                }
            }

        }, ncells, nthreads);

    } else {
        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            static_assert(!EigenMatrix_::IsRowMajor);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(ngenes);

            if (mat.is_sparse()) {
                std::vector<Index_> ibuffer(ngenes);
                auto ext = tatami::consecutive_extractor<true>(mat, false, start, length);

                for (Index_ c = start, end = start + length; c < end; ++c) {
                    auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    static_assert(!EigenMatrix_::IsRowMajor);
                    for (decltype(rank) r = 0; r < rank; ++r) {
                        auto& output = components.coeffRef(r, c);
                        output = 0;
                        auto rotptr = scaled_rotation.data() + sanisizer::product_unsafe<std::size_t>(r, ngenes);
                        for (Index_ i = 0; i < range.number; ++i) {
                            output += rotptr[range.index[i]] * range.value[i];
                        }
                    }
                }

            } else {
                auto ext = tatami::consecutive_extractor<false>(mat, false, start, length);
                for (Index_ c = start, end = start + length; c < end; ++c) {
                    auto ptr = ext->fetch(vbuffer.data()); 
                    static_assert(!EigenMatrix_::IsRowMajor);
                    for (decltype(rank) r = 0; r < rank; ++r) {
                        auto rotptr = scaled_rotation.data() + sanisizer::product_unsafe<std::size_t>(r, ngenes);
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
    for (decltype(projected.rows()) i = 0, prows = projected.rows(); i < prows; ++i) {
        projected.row(i).array() -= projected.row(i).sum() / projected.cols();
    }

    // Just dividing by the number of observations - 1 regardless of weighting.
    typename EigenMatrix_::Scalar denom = projected.cols() - 1;
    for (auto& d : D) {
        d = d * d / denom;
    }
}

/*******************************
 ***** Residual wrapper ********
 *******************************/

// This wrapper class mimics multiplication with the residuals,
// i.e., after subtracting the per-block mean from each cell.
template<class Matrix_, typename Block_, class EigenMatrix_, class EigenVector_>
class ResidualWrapper {
public:
    ResidualWrapper(const Matrix_& mat, const Block_* block, const EigenMatrix_& means) : my_mat(mat), my_block(block), my_means(means) {}

public:
    Eigen::Index rows() const { return my_mat.rows(); }
    Eigen::Index cols() const { return my_mat.cols(); }

public:
    struct Workspace {
        template<typename NumBlocks_>
        Workspace(NumBlocks_ nblocks, irlba::WrappedWorkspace<Matrix_> c) :
            sub(sanisizer::cast<decltype(sub.size())>(nblocks)),
            child(std::move(c))
        {}

        EigenVector_ sub;
        EigenVector_ holding;
        irlba::WrappedWorkspace<Matrix_> child;
    };

    Workspace workspace() const {
        return Workspace(my_means.rows(), irlba::wrapped_workspace(my_mat));
    }

    template<class Right_>
    void multiply(const Right_& rhs, Workspace& work, EigenVector_& output) const {
        const auto& realized_rhs = [&]() -> const auto& {
            if constexpr(std::is_same<Right_, EigenVector_>::value) {
                return rhs;
            } else {
                work.holding.noalias() = rhs;
                return work.holding;
            }
        }();

        irlba::wrapped_multiply(my_mat, realized_rhs, work.child, output);

        work.sub.noalias() = my_means * realized_rhs;
        for (decltype(output.size()) i = 0, end = output.size(); i < end; ++i) {
            auto& val = output.coeffRef(i);
            val -= work.sub.coeff(my_block[i]);
        }
    }

public:
    struct AdjointWorkspace {
        template<typename NumBlocks_>
        AdjointWorkspace(NumBlocks_ nblocks, irlba::WrappedAdjointWorkspace<Matrix_> c) :
            aggr(sanisizer::cast<decltype(aggr.size())>(nblocks)),
            child(std::move(c))
        {}

        EigenVector_ aggr;
        EigenVector_ holding;
        irlba::WrappedAdjointWorkspace<Matrix_> child;
    };

    AdjointWorkspace adjoint_workspace() const {
        return AdjointWorkspace(my_means.rows(), irlba::wrapped_adjoint_workspace(my_mat));
    }

    template<class Right_>
    void adjoint_multiply(const Right_& rhs, AdjointWorkspace& work, EigenVector_& output) const {
        const auto& realized_rhs = [&]() {
            if constexpr(std::is_same<Right_, EigenVector_>::value) {
                return rhs;
            } else {
                work.holding.noalias() = rhs;
                return work.holding;
            }
        }();

        irlba::wrapped_adjoint_multiply(my_mat, realized_rhs, work.child, output);

        work.aggr.setZero();
        for (decltype(realized_rhs.size()) i = 0, end = realized_rhs.size(); i < end; ++i) {
            work.aggr.coeffRef(my_block[i]) += realized_rhs.coeff(i); 
        }

        output.noalias() -= my_means.adjoint() * work.aggr;
    }

public:
    template<class EigenMatrix2_>
    EigenMatrix2_ realize() const {
        EigenMatrix2_ output = irlba::wrapped_realize<EigenMatrix2_>(my_mat);
        for (decltype(output.rows()) i = 0, end = output.rows(); i < end; ++i) {
            output.row(i) -= my_means.row(my_block[i]);
        }
        return output;
    }

private:
    const Matrix_& my_mat;
    const Block_* my_block;
    const EigenMatrix_& my_means;
};

/**************************
 ***** Dispatchers ********
 **************************/

template<bool realize_matrix_, bool sparse_, typename Value_, typename Index_, typename Block_, class EigenMatrix_, class EigenVector_>
void run_blocked(
    const tatami::Matrix<Value_, Index_>& mat, 
    const Block_* block, 
    const BlockingDetails<Index_, EigenVector_>& block_details, 
    const BlockedPcaOptions& options,
    EigenMatrix_& components, 
    EigenMatrix_& rotation, 
    EigenVector_& variance_explained, 
    EigenMatrix_& center_m,
    EigenVector_& scale_v,
    typename EigenVector_::Scalar& total_var,
    bool& converged)
{
    Index_ ngenes = mat.nrow(), ncells = mat.ncol(); 

    auto emat = [&]{
        if constexpr(!realize_matrix_) {
            return internal::TransposedTatamiWrapper<EigenVector_, Value_, Index_>(mat, options.num_threads);

        } else if constexpr(sparse_) {
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
            return irlba::ParallelSparseMatrix(ncells, ngenes, std::move(extracted.value), std::move(extracted.index), std::move(extracted.pointers), true, options.num_threads); 

        } else {
            // Perform an implicit transposition by performing a row-major extraction into a column-major transposed matrix.
            EigenMatrix_ emat(
                sanisizer::cast<decltype(std::declval<EigenMatrix_>().rows())>(ncells),
                sanisizer::cast<decltype(std::declval<EigenMatrix_>().rows())>(ngenes)
            ); 
            static_assert(!EigenMatrix_::IsRowMajor);
            tatami::convert_to_dense(
                mat,
                /* row_major = */ true,
                emat.data(),
                [&]{
                    tatami::ConvertToDenseOptions opt;
                    opt.num_threads = options.num_threads;
                    return opt;
                }()
            );
            return emat;
        }
    }();

    auto nblocks = block_details.block_size.size();
    center_m.resize(
        sanisizer::cast<decltype(center_m.rows())>(nblocks),
        sanisizer::cast<decltype(center_m.cols())>(ngenes)
    );
    scale_v.resize(sanisizer::cast<decltype(scale_v.size())>(ngenes));

    if constexpr(!realize_matrix_) {
        compute_blockwise_mean_and_variance_tatami(mat, block, block_details, center_m, scale_v, options.num_threads);
    } else if constexpr(sparse_) {
        compute_blockwise_mean_and_variance_realized_sparse(emat, block, block_details, center_m, scale_v, options.num_threads);
    } else {
        compute_blockwise_mean_and_variance_realized_dense(emat, block, block_details, center_m, scale_v, options.num_threads);
    }
    total_var = internal::process_scale_vector(options.scale, scale_v);

    ResidualWrapper<decltype(emat), Block_, EigenMatrix_, EigenVector_> centered(emat, block, center_m);

    if (block_details.weighted) {
        if (options.scale) {
            irlba::Scaled<true, decltype(centered), EigenVector_> scaled(centered, scale_v, /* divide = */ true);
            irlba::Scaled<false, decltype(scaled), EigenVector_> weighted(scaled, block_details.expanded_weights, /* divide = */ false);
            auto out = irlba::compute(weighted, options.number, components, rotation, variance_explained, options.irlba_options);
            converged = out.first;
        } else {
            irlba::Scaled<false, decltype(centered), EigenVector_> weighted(centered, block_details.expanded_weights, /* divide = */ false);
            auto out = irlba::compute(weighted, options.number, components, rotation, variance_explained, options.irlba_options);
            converged = out.first;
        }

        EigenMatrix_ tmp;
        const auto& scaled_rotation = scale_rotation_matrix(rotation, options.scale, scale_v, tmp);

        // This transposes 'components' to be a NDIM * NCELLS matrix.
        if constexpr(!realize_matrix_) {
            project_matrix_transposed_tatami(mat, components, scaled_rotation, options.num_threads);
        } else if constexpr(sparse_) {
            project_matrix_realized_sparse(emat, components, scaled_rotation, options.num_threads);
        } else {
            components.noalias() = (emat * scaled_rotation).adjoint();
        }

        // Subtracting each block's mean from the PCs.
        if (options.components_from_residuals) {
            EigenMatrix_ centering = (center_m * scaled_rotation).adjoint();
            for (decltype(ncells) c =0 ; c < ncells; ++c) {
                components.col(c) -= centering.col(block[c]);
            }
        }

        clean_up_projected(components, variance_explained);
        if (!options.transpose) {
            components.adjointInPlace();
        }

    } else {
        if (options.scale) {
            irlba::Scaled<true, decltype(centered), EigenVector_> scaled(centered, scale_v, /* divide = */ true);
            auto out = irlba::compute(scaled, options.number, components, rotation, variance_explained, options.irlba_options);
            converged = out.first;
        } else {
            auto out = irlba::compute(centered, options.number, components, rotation, variance_explained, options.irlba_options);
            converged = out.first;
        }

        if (options.components_from_residuals) {
            internal::clean_up(mat.ncol(), components, variance_explained);
            if (options.transpose) {
                components.adjointInPlace();
            }
        } else {
            EigenMatrix_ tmp;
            const auto& scaled_rotation = scale_rotation_matrix(rotation, options.scale, scale_v, tmp);

            // This transposes 'components' to be a NDIM * NCELLS matrix.
            if constexpr(!realize_matrix_) {
                project_matrix_transposed_tatami(mat, components, scaled_rotation, options.num_threads);
            } else if constexpr(sparse_) {
                project_matrix_realized_sparse(emat, components, scaled_rotation, options.num_threads);
            } else {
                components.noalias() = (emat * scaled_rotation).adjoint();
            }

            clean_up_projected(components, variance_explained);
            if (!options.transpose) {
                components.adjointInPlace();
            }
        }
    }
}

}
/**
 * @endcond
 */

/**
 * @brief Results of `blocked_pca()`.
 *
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<typename EigenMatrix_, typename EigenVector_>
struct BlockedPcaResults {
    /**
     * Matrix of principal components.
     * By default, each row corresponds to a PC while each column corresponds to a cell in the input matrix.
     * If `BlockedPcaOptions::transpose = false`, rows are cells instead.
     * The number of PCs is determined by `BlockedPcaOptions::number`. 
     */
    EigenMatrix_ components;

    /**
     * Variance explained by each PC.
     * Each entry corresponds to a column in `components` and is in decreasing order.
     * The length of the vector is determined by `BlockedPcaOptions::number`. 
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
     * The number of PCs is determined by `BlockedPcaOptions::number`. 
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
 * As mentioned in `simple_pca()`, it is desirable to obtain the top PCs for downstream cell-based analyses.
 * However, in the presence of a blocking factor (e.g., batches, samples), we want to ensure that the PCA is not driven by uninteresting differences between blocks.
 * To achieve this, `blocked_pca()` centers the expression of each gene in each blocking level and uses the residuals for PCA.
 * The gene-gene covariance matrix will thus focus on variation within each batch, 
 * ensuring that the top rotation vectors/principal components capture biological heterogeneity instead of inter-block differences.
 * Internally, `blocked_pca()` defers the residual calculation until the matrix multiplication steps within [IRLBA](https://github.com/LTLA/CppIrlba).
 * This yields the same results as the naive calculation of residuals but is much faster as it can take advantage of efficient sparse operations.
 *
 * By default, the principal components are computed from the (conceptual) matrix of residuals.
 * This yields a low-dimensional space where all inter-block differences have been removed,
 * assuming that all blocks have the same composition and the inter-block differences are consistent for all cell subpopulations.
 * Under these assumptions, we could use these components for downstream analysis without any concern for block-wise effects.
 * In practice, these assumptions do not hold and more sophisticated batch correction methods like [MNN correction](https://github.com/LTLA/CppMnnCorrect) are required.
 * Some of these methods accept a low-dimensional embedding of cells as input, which can be created by `blocked_pca()` with `BlockedPcaOptions::components_from_residuals = false`.
 * In this mode, only the rotation vectors are computed from the residuals.
 * The original expression values for each cell are then projected onto the associated subspace to obtain PC coordinates that can be used for further batch correction.
 * This approach aims to avoid any strong assumptions about the nature of inter-block differences,
 * while still leveraging the benefits of blocking to focus on intra-block biology.
 *
 * If one batch has many more cells than the others, it will dominate the PCA by driving the axes of maximum variance. 
 * This may mask interesting aspects of variation in the smaller batches.
 * To mitigate this, we scale each batch in inverse proportion to its size (see `BlockedPcaOptions::block_weight_policy`).
 * This ensures that each batch contributes equally to the (conceptual) gene-gene covariance matrix and thus the rotation vectors.
 * The vector of residuals for each cell (or the original expression values, if `BlockedPcaOptions::components_from_residuals = false`) 
 * is then projected to the subspace defined by these rotation vectors to obtain that cell's PC coordinates.
 *
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 * @tparam Block_ Integer type for the blocking factor.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
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
void blocked_pca(const tatami::Matrix<Value_, Index_>& mat, const Block_* block, const BlockedPcaOptions& options, BlockedPcaResults<EigenMatrix_, EigenVector_>& output) {
    irlba::EigenThreadScope t(options.num_threads);
    auto bdetails = internal::compute_blocking_details<EigenVector_>(mat.ncol(), block, options.block_weight_policy, options.variable_block_weight_parameters);

    EigenMatrix_& components = output.components;
    EigenMatrix_& rotation = output.rotation;
    EigenVector_& variance_explained = output.variance_explained;
    EigenMatrix_& center_m = output.center;
    EigenVector_& scale_v = output.scale;
    auto& total_var = output.total_variance;
    bool& converged = output.converged;

    if (mat.sparse()) {
        if (options.realize_matrix) {
            internal::run_blocked<true, true>(mat, block, bdetails, options, components, rotation, variance_explained, center_m, scale_v, total_var, converged);
        } else {
            internal::run_blocked<false, true>(mat, block, bdetails, options, components, rotation, variance_explained, center_m, scale_v, total_var, converged);
        }
    } else {
        if (options.realize_matrix) {
            internal::run_blocked<true, false>(mat, block, bdetails, options, components, rotation, variance_explained, center_m, scale_v, total_var, converged);
        } else {
            internal::run_blocked<false, false>(mat, block, bdetails, options, components, rotation, variance_explained, center_m, scale_v, total_var, converged);
        }
    }

    if (!options.scale) {
        output.scale = EigenVector_();
    }
}

/**
 * Overload of `blocked_pca()` that allocates memory for the output.
 *
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
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
