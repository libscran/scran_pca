#ifndef COMPARE_PCS_H
#define COMPARE_PCS_H

#include <gtest/gtest.h>

#include <vector>

#include "scran_tests/scran_tests.hpp"

#include "Eigen/Dense"
#include "tatami/tatami.hpp"

inline void are_pcs_centered(const Eigen::MatrixXd& pcs, double tol = 1e-8) {
    int ndims = pcs.rows(), ncells = pcs.cols();
    for (int r = 0; r < ndims; ++r) {
        auto ptr = pcs.data() + r;

        double mean = 0;
        for (int c = 0; c < ncells; ++c, ptr += ndims) {
            mean += *ptr;
        }
        mean /= ncells;

        EXPECT_LT(std::abs(mean), tol);
    }
}

inline void expect_equal_pcs(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8, bool relative = true) {
    int ndims = left.rows(), ncells = left.cols();
    ASSERT_EQ(ncells, right.cols());
    ASSERT_EQ(ndims, right.rows());

    for (int i = 0; i < ndims; ++i) {
        for (int j = 0; j < ncells; ++j) {
            auto aleft = std::abs(left(i, j));
            auto aright = std::abs(right(i, j));
            if (relative) {
                scran_tests::compare_almost_equal(aleft, aright, tol);
            } else if (std::abs(aleft - aright) > tol) {
                EXPECT_TRUE(false) << "mismatch in almost-equal floats (expected " << aleft << ", got " << aright << ")";
            }
        }

        // PCs should average to zero.
        EXPECT_LT(std::abs(left.row(i).sum()), tol);
        EXPECT_LT(std::abs(right.row(i).sum()), tol);
    }
}

inline void expect_equal_rotation(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
    int ndims = left.rows(), ngenes = left.cols();
    ASSERT_EQ(ngenes, right.cols());
    ASSERT_EQ(ndims, right.rows());

    for (int i = 0; i < ndims; ++i) {
        for (int j = 0; j < ngenes; ++j) {
            auto aleft = std::abs(left(i, j));
            auto aright = std::abs(right(i, j));
            scran_tests::compare_almost_equal(aleft, aright, tol);
        }
    }
}

inline void expect_equal_vectors(const Eigen::VectorXd& left, const Eigen::VectorXd& right, double tol=1e-8) {
    int n = left.size();
    ASSERT_EQ(n, right.size());
    for (int i = 0; i < n; ++i) {
        scran_tests::compare_almost_equal(left[i], right[i], tol);
    }
}

inline void compare_almost_equal(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
    int ndims = left.rows(), ngenes = left.cols();
    ASSERT_EQ(ngenes, right.cols());
    ASSERT_EQ(ndims, right.rows());

    for (int c = 0; c < ngenes; ++c) {
        for (int r = 0; r < ndims; ++r) {
            scran_tests::compare_almost_equal(left(r, c), right(r, c), tol);
        }
    }
}

#endif

