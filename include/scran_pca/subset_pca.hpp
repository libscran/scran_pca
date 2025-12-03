#ifndef SCRAN_PCA_SUBSET_HPP
#define SCRAN_PCA_SUBSET_HPP

#include <vector>

#include "simple_pca.hpp"
#include "blocked_pca.hpp"
#include "utils.hpp"

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"

namespace scran_pca {

/**
 * @cond
 */
template<typename Index_>
std::vector<Index_> invert_subset(const Index_ total, const std::vector<Index_>& subset) {
    std::vector<Index_> output;
    output.reserve(total - subset.size());
    auto ptr = subset.begin(), end = subset.end();
    for (Index_ i = 0; i < total; ++i) {
        if (ptr != end && *ptr == i) {
            ++ptr;
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

template<typename Index_, class EigenVector_>
void expand_into_vector(const std::vector<Index_>& subset, const EigenVector_& source, EigenVector_& dest) {
    const auto nsub = subset.size();
    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        dest.coeffRef(subset[s]) = source.coeff(s);
    }
}

template<typename Index_, class EigenMatrix_>
void expand_into_matrix_rows(const std::vector<Index_>& subset, const EigenMatrix_& source, EigenMatrix_& dest) {
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

template<typename Index_, class EigenMatrix_>
void expand_into_matrix_columns(const std::vector<Index_>& subset, const EigenMatrix_& source, EigenMatrix_& dest) {
    const auto nsub = subset.size();
    for (I<decltype(nsub)> s = 0; s < nsub; ++s) {
        dest.col(subset[s]) = source.col(s);
    }
}
/**
 * @endcond
 */

typedef SimplePcaOptions SubsetPcaOptions;

template<typename EigenMatrix_, class EigenVector_>
using SubsetPcaResults = SimplePcaResults<EigenMatrix_, EigenVector_>;

template<typename Value_, typename Index_, typename EigenMatrix_, class EigenVector_>
void subset_pca(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Index_>& subset,
    const SubsetPcaOptions& options,
    SubsetPcaResults<EigenMatrix_, EigenVector_>& output
) {
    const auto full_size = mat.nrow();
    auto final_center = sanisizer::create<EigenVector_>(full_size);
    auto final_scale = sanisizer::create<EigenVector_>(full_size);
    EigenMatrix_ final_rotation;

    auto sub_mat = tatami::make_DelayedSubset(tatami::wrap_shared_ptr(&mat), subset, true); // don't move subset, we'll need it later.
    simple_pca_internal(
        *sub_mat,
        options,
        output,
        [&](const EigenMatrix_& rhs_vectors, const EigenVector_& sing_vals) -> void {
            auto inv_subset = invert_subset(mat.nrow(), subset);
            auto inv_mat = tatami::make_DelayedSubset(tatami::wrap_shared_ptr(&mat), inv_subset, true); // don't move inv_subset, we'll need it later.

            const auto num_inv = inv_mat->nrow();
            auto inv_center = sanisizer::create<EigenVector_>(num_inv);
            auto inv_scale = sanisizer::create<EigenVector_>(num_inv);
            if (inv_mat->sparse()) {
                compute_row_means_and_variances<true>(*inv_mat, options.num_threads, inv_center, inv_scale);
            } else {
                compute_row_means_and_variances<false>(*inv_mat, options.num_threads, inv_center, inv_scale);
            }
            process_scale_vector(options.scale, inv_scale);

            std::vector<typename EigenVector_::Scalar> product;
            std::vector<typename EigenVector_::Scalar*> product_ptrs;
            multiply_by_right_singular_vectors(
                *inv_mat,
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

template<typename EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, typename Value_, typename Index_>
SubsetPcaResults<EigenMatrix_, EigenVector_> subset_pca(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Index_>& subset,
    const SubsetPcaOptions& options
) {
    SubsetPcaResults<EigenMatrix_, EigenVector_> output;
    subset_pca(mat, subset, options, output);
    return output;
}

typedef BlockedPcaOptions SubsetPcaBlockedOptions;

template<typename EigenMatrix_, class EigenVector_>
using SubsetPcaBlockedResults = BlockedPcaResults<EigenMatrix_, EigenVector_>;

template<typename Value_, typename Index_, typename Block_, typename EigenMatrix_, class EigenVector_>
void subset_pca_blocked(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Index_>& subset,
    const Block_* block, 
    const SubsetPcaBlockedOptions& options,
    SubsetPcaBlockedResults<EigenMatrix_, EigenVector_>& output
) {
    const auto full_size = mat.nrow();
    EigenMatrix_ final_center;
    auto final_scale = sanisizer::create<EigenVector_>(full_size);
    EigenMatrix_ final_rotation;

    auto submat = tatami::make_DelayedSubset(tatami::wrap_shared_ptr(&mat), subset, true); // don't move the subset, we'll need it later.
    blocked_pca_internal<Value_, Index_, Block_, EigenMatrix_, EigenVector_>(
        *submat,
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
            auto inv_mat = tatami::make_DelayedSubset(tatami::wrap_shared_ptr(&mat), inv_subset, true); // don't move inv_subset, we'll need it later.

            const auto num_cells = inv_mat->ncol();
            const auto num_inv = inv_mat->nrow();
            const auto num_blocks = block_details.block_size.size();
            EigenMatrix_ inv_center(
                sanisizer::cast<Eigen::Index>(num_blocks),
                sanisizer::cast<Eigen::Index>(num_inv)
            );
            auto inv_scale = sanisizer::create<EigenVector_>(num_inv);
            compute_blockwise_mean_and_variance_tatami(*inv_mat, block, block_details, inv_center, inv_scale, options.num_threads);
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
                *inv_mat,
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
                const auto compute = [&](I<decltype(num_inv)> i) -> void {
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

template<typename EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, typename Value_, typename Index_, typename Block_>
SubsetPcaBlockedResults<EigenMatrix_, EigenVector_>  subset_pca_blocked(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Index_>& subset,
    const Block_* block, 
    const SubsetPcaBlockedOptions& options
) {
    SubsetPcaBlockedResults<EigenMatrix_, EigenVector_> output;
    subset_pca_blocked(mat, subset, block, options, output);
    return output;
}

}

#endif
