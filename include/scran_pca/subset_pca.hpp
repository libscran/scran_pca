#ifndef SCRAN_PCA_SUBSET_HPP
#define SCRAN_PCA_SUBSET_HPP

#include <vector>
#include <optional>
#include <cstddef>

#include "simple_pca.hpp"
#include "blocked_pca.hpp"
#include "utils.hpp"

#include "sanisizer/sanisizer.hpp"
#include "tatami_mult/tatami_mult.hpp"
#include "tatami/tatami.hpp"
#include "Eigen/Dense"

/**
 * @file subset_pca.hpp
 * @brief PCA on a subset of features.
 */

namespace scran_pca {

/**
 * @cond
 */
template<typename Index_, class SubsetVector_>
std::vector<Index_> invert_subset(const Index_ total, const SubsetVector_& subset) {
    std::vector<Index_> output;
    output.reserve(total - subset.size());
    const auto end = subset.size(); 
    I<decltype(end)> pos = 0;
    for (Index_ i = 0; i < total; ++i) {
        if (pos != end && sanisizer::is_equal(subset[pos], i)) {
            ++pos;
            continue;
        }
        output.push_back(i);
    }
    return output;
}

template<typename Value_, typename Index_, typename EigenMatrix_, typename Scalar_>
void multiply_by_right_singular_vectors(
    const tatami::Matrix<Value_, Index_>& mat,
    const EigenMatrix_& rhs_vectors,
    std::vector<Scalar_>& output,
    std::vector<Scalar_*>& out_ptrs,
    int num_threads
) {
    const auto num_features = mat.nrow();
    const auto num_cells = mat.ncol();
    const auto rank = rhs_vectors.cols();
    static_assert(!EigenMatrix_::IsRowMajor);

    output.resize(sanisizer::product<I<decltype(output.size())> >(num_features, rank));
    sanisizer::resize(out_ptrs, rank);
    auto rhs_ptrs = sanisizer::create<std::vector<const typename EigenMatrix_::Scalar*> >(rank);
    for (I<decltype(rank)> r = 0; r < rank; ++r) {
        rhs_ptrs[r] = rhs_vectors.data() + sanisizer::product_unsafe<std::size_t>(r, num_cells);
        out_ptrs[r] = output.data() + sanisizer::product_unsafe<std::size_t>(r, num_features);
    }

    tatami_mult::Options opt;
    opt.num_threads = num_threads;
    tatami_mult::multiply(mat, rhs_ptrs, out_ptrs, opt);
}

template<class SubsetVector_, class EigenVector_>
void expand_into_vector(const SubsetVector_& subset, const EigenVector_& source, EigenVector_& dest) {
    const auto nsub = subset.size();
    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        dest.coeffRef(subset[s]) = source.coeff(s);
    }
}

template<class SubsetVector_, class EigenMatrix_>
void expand_into_matrix_rows(const SubsetVector_& subset, const EigenMatrix_& source, EigenMatrix_& dest) {
    const auto nsub = subset.size();

    // This access pattern should be a little more cache-friendly for the
    // default column-major storage of Eigen::MatrixXd's.
    const auto cols = dest.cols();
    for (I<decltype(cols)> c = 0; c < cols; ++c) {
        for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
            dest.coeffRef(subset[s], c) = source.coeff(s, c);
        }
    }
}

template<class SubsetVector_, class EigenMatrix_>
void expand_into_matrix_columns(const SubsetVector_& subset, const EigenMatrix_& source, EigenMatrix_& dest) {
    const auto nsub = subset.size();
    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        dest.col(subset[s]) = source.col(s);
    }
}
/**
 * @endcond
 */

/**
 * Options for `subset_pca()`.
 * These are identical to those for `simple_pca()`.
 */
typedef SimplePcaOptions SubsetPcaOptions;

/**
 * Results of `subset_pca()`.
 *
 * These are mostly the same as the results for `simple_pca()`.
 * The only difference is that the number of PCs is the smaller of `SimplePcaOptions::number` and `min(subset.size(), NC) - 1`,
 * where `subset` is the subset vector and `NC` is the number of columns of the input matrix.
 *
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<typename EigenMatrix_, class EigenVector_>
using SubsetPcaResults = SimplePcaResults<EigenMatrix_, EigenVector_>;

/**
 * Principal components analysis on a subset of features. 
 *
 * This function performs PCA on a subset of features (e.g., from highly variable genes) in the input matrix.
 * The results are almost equivalent to subsetting the input matrix and then running `simple_pca()`.
 * However, `subset_pca()` will also populate the rotation matrix, centering vector and scaling vector for features outside of the subset.
 * For the rotation matrix, this is done by projecting the unused features into the low-dimensional space defined by the PCs.
 * The goal is to allow callers to create a low-rank approximation of the entire input matrix, even if only a subset of the features are relevant to the PCA.
 *
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 * @tparam SubsetVector_ Container of the row indices.
 * Should support `[]`, `size()` and copy construction. 
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 *
 * @param[in] mat The input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param subset Vector of indices for rows to be used in the PCA.
 * This should be sorted and unique.
 * @param options Further options.
 * @param[out] output On output, the results of the PCA on `mat`.
 * This can be re-used across multiple calls to `subset_pca()`. 
 */
template<typename Value_, typename Index_, typename SubsetVector_, typename EigenMatrix_, class EigenVector_>
void subset_pca(
    const tatami::Matrix<Value_, Index_>& mat,
    const SubsetVector_& subset,
    const SubsetPcaOptions& options,
    SubsetPcaResults<EigenMatrix_, EigenVector_>& output
) {
    const auto full_size = mat.nrow();
    auto final_center = sanisizer::create<EigenVector_>(full_size);
    auto final_scale = sanisizer::create<EigenVector_>(full_size);
    EigenMatrix_ final_rotation;

    // Don't move subset into the constructor, we'll need it later.
    tatami::DelayedSubsetSortedUnique<Value_, Index_, SubsetVector_> sub_mat(tatami::wrap_shared_ptr(&mat), subset, true);

    simple_pca_internal(
        sub_mat,
        options,
        output,
        [&](const EigenMatrix_& rhs_vectors, const EigenVector_& sing_vals) -> void {
            const auto inv_subset = invert_subset(mat.nrow(), subset);
            // Don't move inv_subset into the constructor, we'll need it later.
            tatami::DelayedSubsetSortedUnique<Value_, Index_, I<decltype(inv_subset)> > inv_mat(tatami::wrap_shared_ptr(&mat), inv_subset, true);

            const auto num_inv = inv_mat.nrow();
            auto inv_center = sanisizer::create<EigenVector_>(num_inv);
            auto inv_scale = sanisizer::create<EigenVector_>(num_inv);
            if (inv_mat.sparse()) {
                compute_row_means_and_variances<true>(inv_mat, options.num_threads, inv_center, inv_scale);
            } else {
                compute_row_means_and_variances<false>(inv_mat, options.num_threads, inv_center, inv_scale);
            }
            process_scale_vector(options.scale, inv_scale);

            std::vector<typename EigenVector_::Scalar> product;
            std::vector<typename EigenVector_::Scalar*> product_ptrs;
            multiply_by_right_singular_vectors(
                inv_mat,
                rhs_vectors,
                product,
                product_ptrs,
                options.num_threads
            );

            const auto rank = rhs_vectors.cols();
            final_rotation.resize(sanisizer::cast<Eigen::Index>(full_size), rank);
            for (I<decltype(rank)> r = 0; r < rank; ++r) {
                const auto curshift = rhs_vectors.col(r).sum();
                const auto varexp = sing_vals.coeff(r);
                const auto optr = product_ptrs[r];
                const auto compute = [&](I<decltype(num_inv)> i) -> typename EigenVector_::Scalar {
                    return (optr[i] - curshift * inv_center.coeff(i)) / varexp;
                };

                if (!options.scale) {
                    for (I<decltype(num_inv)> i = 0; i < num_inv; ++i) {
                        final_rotation.coeffRef(inv_subset[i], r) = compute(i);
                    }
                } else {
                    for (I<decltype(num_inv)> i = 0; i < num_inv; ++i) {
                        final_rotation.coeffRef(inv_subset[i], r) = compute(i) / inv_scale.coeff(i);
                    }
                }
            }

            expand_into_vector(inv_subset, inv_center, final_center);
            if (options.scale) {
                expand_into_vector(inv_subset, inv_scale, final_scale);
            }
        }
    );

    expand_into_vector(subset, output.center, final_center);
    output.center.swap(final_center);

    if (options.scale) {
        expand_into_vector(subset, output.scale, final_scale);
        output.scale.swap(final_scale);
    }

    expand_into_matrix_rows(subset, output.rotation, final_rotation);
    output.rotation.swap(final_rotation);
}

/**
 * Overload of `subset_pca()` that allocates memory for the output.
 *
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 * @tparam SubsetVector_ Container of the row indices.
 * Should support `[]`, `size()` and copy construction. 
 *
 * @param[in] mat The input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param subset Vector of indices for rows to be used in the PCA.
 * This should be sorted and unique.
 * @param options Further options.
 *
 * @return Results of the subsetted PCA.
 */
template<typename EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, typename Value_, typename Index_, class SubsetVector_>
SubsetPcaResults<EigenMatrix_, EigenVector_> subset_pca(
    const tatami::Matrix<Value_, Index_>& mat,
    const SubsetVector_& subset,
    const SubsetPcaOptions& options
) {
    SubsetPcaResults<EigenMatrix_, EigenVector_> output;
    subset_pca(mat, subset, options, output);
    return output;
}

/**
 * Options for `subset_pca_blocked()`.
 * These are identical to the options for `blocked_pca()`.
 */
typedef BlockedPcaOptions SubsetPcaBlockedOptions;

/**
 * Results of `subset_pca_blocked()`.
 *
 * These are mostly the same as the results for `blocked_pca()`.
 * The only difference is that the number of PCs is the smaller of `BlockedPcaOptions::number` and `min(subset.size(), NC) - 1`,
 * where `subset` is the subset vector and `NC` is the number of columns of the input matrix.
 *
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 */
template<typename EigenMatrix_, class EigenVector_>
using SubsetPcaBlockedResults = BlockedPcaResults<EigenMatrix_, EigenVector_>;

/**
 * Principal components analysis on a subset of features in the input matrix, with blocking.
 *
 * This function performs PCA on a subset of interesting features (e.g., from highly variable genes) while accounting for a blocking factor.
 * The results are almost equivalent to subsetting the input matrix and then running `blocked_pca()`.
 * However, `subset_pca_blocked()` will also populate the rotation matrix, centering matrix and scaling vector for features outside of the subset.
 * For the rotation matrix, this is done by projecting the unused features into the low-dimensional space defined by the top PCs.
 * The goal is to allow callers to create a low-rank approximation of the entire input matrix, even if only a subset of the features are relevant to the PCA.
 *
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 * @tparam SubsetVector_ Container of the row indices.
 * Should support `[]`, `size()` and copy construction. 
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 *
 * @param[in] mat The input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param subset Vector of indices for rows to be used in the PCA.
 * This should be sorted and unique.
 * @param[in] block Pointer to an array of length equal to the number of cells, 
 * containing the block assignment for each cell. 
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options.
 * @param[out] output On output, the results of the PCA on `mat`.
 * This can be re-used across multiple calls to `subset_pca_blocked()`. 
 */
template<typename Value_, typename Index_, class SubsetVector_, typename Block_, typename EigenMatrix_, class EigenVector_>
void subset_pca_blocked(
    const tatami::Matrix<Value_, Index_>& mat,
    const SubsetVector_& subset,
    const Block_* block, 
    const SubsetPcaBlockedOptions& options,
    SubsetPcaBlockedResults<EigenMatrix_, EigenVector_>& output
) {
    const auto full_size = mat.nrow();
    EigenMatrix_ final_center;
    auto final_scale = sanisizer::create<EigenVector_>(full_size);
    EigenMatrix_ final_rotation;

    // Don't move subset into the constructor, we'll need it later.
    tatami::DelayedSubsetSortedUnique<Value_, Index_, SubsetVector_> sub_mat(tatami::wrap_shared_ptr(&mat), subset, true);

    blocked_pca_internal<Value_, Index_, Block_, EigenMatrix_, EigenVector_>(
        sub_mat,
        block,
        options,
        output,
        [&](
            const BlockingDetails<Index_, EigenVector_>& block_details,
            const EigenMatrix_& rhs_vectors,
            const EigenVector_& sing_vals
        ) -> void {
            final_center.resize(
                sanisizer::cast<I<decltype(final_center.rows())> >(block_details.block_size.size()),
                sanisizer::cast<I<decltype(final_center.cols())> >(full_size)
            );
            final_rotation.resize(
                sanisizer::cast<I<decltype(final_rotation.rows())> >(full_size),
                rhs_vectors.cols()
            ); 

            auto inv_subset = invert_subset(mat.nrow(), subset);
            // Don't move inv_subset into the constructor, we'll need it later.
            tatami::DelayedSubsetSortedUnique<Value_, Index_, I<decltype(inv_subset)> > inv_mat(tatami::wrap_shared_ptr(&mat), inv_subset, true);

            const auto num_cells = inv_mat.ncol();
            const auto num_inv = inv_mat.nrow();
            const auto num_blocks = block_details.block_size.size();
            EigenMatrix_ inv_center(
                sanisizer::cast<Eigen::Index>(num_blocks),
                sanisizer::cast<Eigen::Index>(num_inv)
            );
            auto inv_scale = sanisizer::create<EigenVector_>(num_inv);
            compute_blockwise_mean_and_variance_tatami(inv_mat, block, block_details, inv_center, inv_scale, options.num_threads);
            process_scale_vector(options.scale, inv_scale);

            // Need to adjust the RHS singular vector matrix to mimic weighting of the input matrix.
            const EigenMatrix_* rhs_ptr = NULL;
            std::optional<EigenMatrix_> weighted_rhs;
            if (block_details.weighted) {
                weighted_rhs = rhs_vectors;
                weighted_rhs->array().colwise() *= block_details.expanded_weights.array();
                rhs_ptr = &(*weighted_rhs);
            } else {
                rhs_ptr = &rhs_vectors;
            }

            std::vector<typename EigenVector_::Scalar> product;
            std::vector<typename EigenVector_::Scalar*> out_ptrs;
            multiply_by_right_singular_vectors(
                inv_mat,
                *rhs_ptr,
                product,
                out_ptrs,
                options.num_threads
            );

            const auto rank = rhs_vectors.cols();
            auto shift_buffer = sanisizer::create<EigenVector_>(num_blocks);
            for (I<decltype(rank)> r = 0; r < rank; ++r) {
                std::fill(shift_buffer.begin(), shift_buffer.end(), 0);
                for (I<decltype(num_cells)> i = 0; i < num_cells; ++i) {
                    shift_buffer.coeffRef(block[i]) += rhs_vectors.coeff(i, r);
                }

                const auto varexp = sing_vals.coeff(r);
                const auto optr = out_ptrs[r];
                const auto compute = [&](I<decltype(num_inv)> i) -> typename EigenVector_::Scalar {
                    typename EigenVector_::Scalar curshift = 0;
                    for (I<decltype(num_blocks)> b = 0; b < num_blocks; ++b) {
                        curshift += shift_buffer.coeff(b) * inv_center.coeff(b, i);
                    }
                    return (optr[i] - curshift) / varexp;
                };

                if (options.scale) {
                    for (I<decltype(num_inv)> i = 0; i < num_inv; ++i) {
                        final_rotation.coeffRef(inv_subset[i], r) = compute(i) / inv_scale.coeff(i);
                    }
                } else {
                    for (I<decltype(num_inv)> i = 0; i < num_inv; ++i) {
                        final_rotation.coeffRef(inv_subset[i], r) = compute(i);
                    }
                }
            }

            expand_into_matrix_columns(inv_subset, inv_center, final_center);
            if (options.scale) {
                expand_into_vector(inv_subset, inv_scale, final_scale);
            }
        }
    );

    expand_into_matrix_columns(subset, output.center, final_center);
    output.center.swap(final_center);

    if (options.scale) {
        expand_into_vector(subset, output.scale, final_scale);
        output.scale.swap(final_scale);
    }

    expand_into_matrix_rows(subset, output.rotation, final_rotation);
    output.rotation.swap(final_rotation);
}

/**
 * Overload of `subset_pca_blocked()` that allocates memory for the output.
 *
 * @tparam EigenMatrix_ A floating-point column-major `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 * @tparam Value_ Type of the matrix data.
 * @tparam Index_ Integer type for the indices.
 * @tparam SubsetVector_ Container of the row indices.
 * Should support `[]`, `size()` and copy construction. 
 * @tparam Block_ Integer type for the blocking factor.
 *
 * @param[in] mat Input matrix.
 * Columns should contain cells while rows should contain genes.
 * Matrix entries are typically log-expression values.
 * @param subset Vector of indices for rows to be used in the PCA.
 * This should be sorted and unique.
 * @param[in] block Pointer to an array of length equal to the number of cells, 
 * containing the block assignment for each cell. 
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options.
 *
 * @return Results of the blocked and subsetted PCA. 
 */
template<typename EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, typename Value_, typename Index_, class SubsetVector_, typename Block_>
SubsetPcaBlockedResults<EigenMatrix_, EigenVector_>  subset_pca_blocked(
    const tatami::Matrix<Value_, Index_>& mat,
    const SubsetVector_& subset,
    const Block_* block, 
    const SubsetPcaBlockedOptions& options
) {
    SubsetPcaBlockedResults<EigenMatrix_, EigenVector_> output;
    subset_pca_blocked(mat, subset, block, options, output);
    return output;
}

}

#endif
