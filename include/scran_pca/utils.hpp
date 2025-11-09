#ifndef SCRAN_PCA_UTILS_HPP
#define SCRAN_PCA_UTILS_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <memory>

#include "tatami/tatami.hpp"
#include "tatami_mult/tatami_mult.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "irlba/irlba.hpp"

namespace scran_pca {

template<typename Input_>
using I = typename std::remove_cv<typename std::remove_reference<Input_>::type>::type;

template<class EigenVector_>
auto process_scale_vector(const bool scale, EigenVector_& scale_v) {
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
void clean_up(const NumObs_ num_obs, EigenMatrix_& U, EigenVector_& D) {
    typename EigenVector_::Scalar denom = num_obs - 1;
    U.array().rowwise() *= D.adjoint().array();
    for (auto& d : D) {
        d = d * d / denom;
    }
}

template<typename Value_, typename Index_>
class TransposedTatamiWrapperCore {
public:
    TransposedTatamiWrapperCore(const tatami::Matrix<Value_, Index_>& mat, int num_threads) : 
        my_mat(mat), 
        my_nrow(mat.nrow()),
        my_ncol(mat.ncol()),
        my_is_sparse(mat.is_sparse()),
        my_prefer_rows(mat.prefer_rows()),
        my_num_threads(num_threads)
    {}

private:
    const tatami::Matrix<Value_, Index_>& my_mat;
    Index_ my_nrow, my_ncol;
    bool my_is_sparse;
    bool my_prefer_rows;
    int my_num_threads;

public:
    const tatami::Matrix<Value_, Index_>& get_matrix() const {
        return my_mat;
    }

    int get_num_threads() const {
        return my_num_threads;
    }

    Index_ get_nrow() const {
        return my_nrow;
    }

    Index_ get_ncol() const {
        return my_ncol;
    }

public:
    template<class EigenVector_>
    void inner_multiply(const EigenVector_& right, bool transposed, EigenVector_& out) const {
        tatami_mult::Options opt;
        opt.num_threads = my_num_threads;
        if (!transposed) {
            tatami_mult::multiply(my_mat, right.data(), out.data(), opt);
        } else {
            tatami_mult::multiply(right.data(), my_mat, out.data(), opt);
        }
    }
};

template<class EigenVector_, typename Value_, typename Index_>
class TransposedTatamiWrapperWorkspace final : public irlba::Workspace<EigenVector_> {
public:
    TransposedTatamiWrapperWorkspace(const TransposedTatamiWrapperCore<Value_, Index_>& core) : my_core(core) {}

private:
    const TransposedTatamiWrapperCore<Value_, Index_>& my_core;

public:
    void multiply(const EigenVector_& rhs, EigenVector_& out) {
        my_core.inner_multiply(
            rhs,
            true, // mimicking a transposed matrix, remember!
            out
        );
    }
};

template<class EigenVector_, typename Value_, typename Index_>
class TransposedTatamiWrapperAdjointWorkspace final : public irlba::AdjointWorkspace<EigenVector_> {
public:
    TransposedTatamiWrapperAdjointWorkspace(const TransposedTatamiWrapperCore<Value_, Index_>& core) : my_core(core) {}

private:
    const TransposedTatamiWrapperCore<Value_, Index_>& my_core;

public:
    void multiply(const EigenVector_& rhs, EigenVector_& out) {
        my_core.inner_multiply(
            rhs,
            false, // mimicking a transposed matrix, remember!
            out
        );
    }
};

template<class EigenMatrix_, typename Value_, typename Index_>
class TransposedTatamiWrapperRealizeWorkspace final : public irlba::RealizeWorkspace<EigenMatrix_> {
public:
    TransposedTatamiWrapperRealizeWorkspace(const TransposedTatamiWrapperCore<Value_, Index_>& core) :
        my_core(core)
    {}

private:
    const TransposedTatamiWrapperCore<Value_, Index_>& my_core;

public:
    const EigenMatrix_& realize(EigenMatrix_& buffer) {
        // Copying into a transposed matrix, hence the switch of the ncol/nrow order.
        // Both values can be cast to Eigen::Index, as we checked this in the TransposedTatamiWrapperMatrix constructor.
        buffer.resize(my_core.get_ncol(), my_core.get_nrow());

        tatami::convert_to_dense(
            my_core.get_matrix(),
            !buffer.IsRowMajor,
            buffer.data(),
            [&]{
                tatami::ConvertToDenseOptions opt;
                opt.num_threads = my_core.get_num_threads();
                return opt;
            }()
        );

        return buffer;
    }
};

template<class EigenVector_, class EigenMatrix_, typename Value_, typename Index_>
class TransposedTatamiWrapperMatrix final : public irlba::Matrix<EigenVector_, EigenMatrix_> {
public:
    TransposedTatamiWrapperMatrix(const tatami::Matrix<Value_, Index_>& mat, int num_threads) : 
        my_core(mat, num_threads)
    {
        // Check that these casts are safe.
        sanisizer::cast<Eigen::Index>(my_core.get_nrow());
        sanisizer::cast<Eigen::Index>(my_core.get_ncol());
    }

public:
    Eigen::Index rows() const {
        return my_core.get_ncol(); // transposed, remember.
    }

    Eigen::Index cols() const {
        return my_core.get_nrow();
    }

private:
    TransposedTatamiWrapperCore<Value_, Index_> my_core;

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
    std::unique_ptr<TransposedTatamiWrapperWorkspace<EigenVector_, Value_, Index_> > new_known_workspace() const {
        return std::make_unique<TransposedTatamiWrapperWorkspace<EigenVector_, Value_, Index_> >(my_core);
    }

    std::unique_ptr<TransposedTatamiWrapperAdjointWorkspace<EigenVector_, Value_, Index_> > new_known_adjoint_workspace() const {
        return std::make_unique<TransposedTatamiWrapperAdjointWorkspace<EigenVector_, Value_, Index_> >(my_core);
    }

    std::unique_ptr<TransposedTatamiWrapperRealizeWorkspace<EigenMatrix_, Value_, Index_> > new_known_realize_workspace() const {
        return std::make_unique<TransposedTatamiWrapperRealizeWorkspace<EigenMatrix_, Value_, Index_> >(my_core);
    }
};

}

#endif
