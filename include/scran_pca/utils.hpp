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

}

#endif
