#ifndef COMPARE_PCS_H
#define COMPARE_PCS_H

#include <gtest/gtest.h>

#include <vector>

#include "scran_tests/scran_tests.hpp"

#include "Eigen/Dense"
#include "tatami/tatami.hpp"

inline void are_pcs_centered(const Eigen::MatrixXd& pcs, double tol = 1e-8) {
    const Eigen::VectorXd means = pcs.rowwise().mean();
    ASSERT_EQ(means.size(), pcs.rows());
    for (auto m : means) {
        EXPECT_LT(std::abs(m), tol);
    }
}

inline void expect_equal_pcs(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
    int ndims = left.rows(), ncells = left.cols();
    ASSERT_EQ(ncells, right.cols());
    ASSERT_EQ(ndims, right.rows());

    scran_tests::CompareAlmostEqualParameters params;
    params.relative_tolerance = tol;

    for (int i = 0; i < ndims; ++i) {
        for (int j = 0; j < ncells; ++j) {
            auto aleft = std::abs(left(i, j));
            auto aright = std::abs(right(i, j));
            scran_tests::compare_almost_equal(aleft, aright, params);
        }

        // PCs should average to zero.
        EXPECT_LT(std::abs(left.row(i).sum()), tol);
        EXPECT_LT(std::abs(right.row(i).sum()), tol);
    }
}

inline void expect_equal_rotation(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
    int ndims = left.cols(), ngenes = left.rows();
    ASSERT_EQ(ngenes, right.rows());
    ASSERT_EQ(ndims, right.cols());

    scran_tests::CompareAlmostEqualParameters params;
    params.relative_tolerance = tol;

    for (int i = 0; i < ndims; ++i) {
        for (int j = 0; j < ngenes; ++j) {
            auto aleft = std::abs(left(j, i));
            auto aright = std::abs(right(j, i));
            scran_tests::compare_almost_equal(aleft, aright, params);
        }

        scran_tests::compare_almost_equal(
            std::abs(left.col(i).sum()),
            std::abs(right.col(i).sum()),
            params
        );
    }
}

inline void expect_equal_vectors(const Eigen::VectorXd& left, const Eigen::VectorXd& right, double tol=1e-8) {
    int n = left.size();
    ASSERT_EQ(n, right.size());

    scran_tests::CompareAlmostEqualParameters params;
    params.relative_tolerance = tol;

    for (int i = 0; i < n; ++i) {
        scran_tests::compare_almost_equal(left[i], right[i], params);
    }
}

inline void expect_equal_matrices(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
    int ndims = left.rows(), ngenes = left.cols();
    ASSERT_EQ(ngenes, right.cols());
    ASSERT_EQ(ndims, right.rows());

    scran_tests::CompareAlmostEqualParameters params;
    params.relative_tolerance = tol;

    for (int c = 0; c < ngenes; ++c) {
        for (int r = 0; r < ndims; ++r) {
            scran_tests::compare_almost_equal(left(r, c), right(r, c), params);
        }
    }
}

#endif

