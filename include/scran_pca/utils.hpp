#ifndef SCRAN_PCA_UTILS_HPP
#define SCRAN_PCA_UTILS_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"

namespace scran_pca {

namespace internal {

template<class EigenVector_>
auto process_scale_vector(bool scale, EigenVector_& scale_v) {
    typedef typename EigenVector_::Scalar Scalar;
    if (scale) {
        Scalar total_var = 0;
        for (auto& s : scale_v) {
            if (s) {
                s = std::sqrt(s);
                ++total_var;
            } else {
                s = 1; // avoid division by zero.
            }
        }
        return total_var;
    } else {
        return std::accumulate(scale_v.begin(), scale_v.end(), static_cast<Scalar>(0.0));
    }
}

template<typename NumObs_, class EigenMatrix_, class EigenVector_>
void clean_up(NumObs_ num_obs, EigenMatrix_& U, EigenVector_& D) {
    typename EigenVector_::Scalar denom = num_obs - 1;
    U.array().rowwise() *= D.adjoint().array();
    for (auto& d : D) {
        d = d * d / denom;
    }
}

template<class EigenVector_, typename Value_, typename Index_>
class TransposedTatamiWrapper {
public:
    TransposedTatamiWrapper(const tatami::Matrix<Value_, Index_>& mat, int num_threads) : 
        my_mat(mat), 
        my_nrow(mat.nrow()),
        my_ncol(mat.ncol()),
        my_is_sparse(mat.is_sparse()),
        my_prefer_rows(mat.prefer_rows()),
        my_num_threads(num_threads)
    {
        // Check that these casts are safe.
        sanisizer::cast<Eigen::Index>(my_nrow);
        sanisizer::cast<Eigen::Index>(my_ncol);
    }

public:
    Eigen::Index rows() const {
        return my_ncol; // transposed, remember.
    }

    Eigen::Index cols() const {
        return my_nrow;
    }

private:
    const tatami::Matrix<Value_, Index_>& my_mat;
    Index_ my_nrow, my_ncol;
    bool my_is_sparse;
    bool my_prefer_rows;
    int my_num_threads;
    typedef typename EigenVector_::Scalar Scalar;

public:
    struct Workspace {
        std::vector<std::vector<Value_> > vbuffers;
        std::vector<std::vector<Index_> > ibuffers;
        EigenVector_ holding;
    };

    Workspace workspace() const {
        Workspace output;
        output.vbuffers.resize(sanisizer::cast<decltype(output.vbuffers.size())>(my_num_threads));
        if (my_is_sparse) {
            output.ibuffers.resize(sanisizer::cast<decltype(output.ibuffers.size())>(my_num_threads));
        }

        return output;
    }

    typedef Workspace AdjointWorkspace;

    AdjointWorkspace adjoint_workspace() const {
        return workspace();
    }

private:
    template<class Right_>
    void inner_multiply(const Right_& rhs, bool transposed, Workspace& work, EigenVector_& out) const {
        const auto& realized_rhs = [&]() -> const auto& {
            if constexpr(std::is_same<Right_, EigenVector_>::value) {
                return rhs;
            } else {
                work.holding = rhs;
                return work.holding;
            }
        }();

        auto resultdim = (transposed ? my_ncol : my_nrow);
        auto otherdim = (transposed ? my_nrow : my_ncol);

        tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
            auto& vbuffer = work.vbuffers[t];

            if (my_prefer_rows != transposed) {
                tatami::resize_container_to_Index_size(vbuffer, otherdim);

                if (my_is_sparse) {
                    auto& ibuffer = work.ibuffers[t];
                    tatami::resize_container_to_Index_size(ibuffer, otherdim);
                    auto ext = tatami::consecutive_extractor<true>(my_mat, my_prefer_rows, start, length);

                    for (Index_ r = start, end = start + length; r < end; ++r) {
                        auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                        Scalar prod = 0;
                        for (Index_ i = 0; i < range.number; ++i) {
                            prod += realized_rhs[range.index[i]] * range.value[i];
                        }
                        out[r] = prod;
                    }

                } else {
                    auto ext = tatami::consecutive_extractor<false>(my_mat, my_prefer_rows, start, length);
                    for (Index_ r = start, end = start + length; r < end; ++r) {
                        auto ptr = ext->fetch(vbuffer.data());
                        out[r] = std::inner_product(realized_rhs.begin(), realized_rhs.end(), ptr, static_cast<Scalar>(0));
                    }
                }

            } else {
                tatami::resize_container_to_Index_size(vbuffer, length);

                if (my_is_sparse) {
                    auto& ibuffer = work.ibuffers[t];
                    tatami::resize_container_to_Index_size(ibuffer, length);
                    auto ext = tatami::consecutive_extractor<true>(my_mat, my_prefer_rows, static_cast<Index_>(0), otherdim, start, length);
                    tatami_stats::LocalOutputBuffer<Scalar> buffer(t, start, length, out.data());
                    auto bdata = buffer.data();
                    for (Index_ c = 0; c < otherdim; ++c) {
                        auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                        auto mult = realized_rhs[c];
                        for (Index_ i = 0; i < range.number; ++i) {
                            bdata[range.index[i] - start] += mult * range.value[i];
                        }
                    }
                    buffer.transfer();

                } else {
                    auto ext = tatami::consecutive_extractor<false>(my_mat, my_prefer_rows, static_cast<Index_>(0), otherdim, start, length);
                    tatami_stats::LocalOutputBuffer<Scalar> buffer(t, start, length, out.data());
                    auto bdata = buffer.data();
                    for (Index_ c = 0; c < otherdim; ++c) {
                        auto ptr = ext->fetch(vbuffer.data());
                        auto mult = realized_rhs[c];
                        for (Index_ r = 0; r < length; ++r) {
                            bdata[r] += mult * ptr[r];
                        }
                    }
                    buffer.transfer();
                }
            }

        }, resultdim, my_num_threads);
    }

public:
    template<class Right_>
    void multiply(const Right_& rhs, Workspace& work, EigenVector_& out) const {
        inner_multiply(rhs, true, work, out); // mimicking a transposed matrix, remember!
    }

    template<class Right_>
    void adjoint_multiply(const Right_& rhs, Workspace& work, EigenVector_& out) const {
        inner_multiply(rhs, false, work, out);
    }

    template<class EigenMatrix_>
    EigenMatrix_ realize() const {
        // Copying into a transposed matrix.
        EigenMatrix_ emat(
            sanisizer::cast<decltype(std::declval<EigenMatrix_>().rows())>(my_ncol),
            sanisizer::cast<decltype(std::declval<EigenMatrix_>().cols())>(my_nrow)
        );
        tatami::convert_to_dense(
            my_mat,
            !emat.IsRowMajor,
            emat.data(),
            [&]{
                tatami::ConvertToDenseOptions opt;
                opt.num_threads = my_num_threads;
                return opt;
            }()
        );
        return emat;
    }
};

}

}

#endif
