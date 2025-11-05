#include <gtest/gtest.h>

#include "compare_pcs.h"
#include "utils.h"

#include "scran_tests/scran_tests.hpp"
#include "tatami/tatami.hpp"

#include "scran_pca/utils.hpp"

class TransposedTatamiWrapperTest : public ::testing::TestWithParam<int> {};

TEST_P(TransposedTatamiWrapperTest, DenseColumn) {
    std::size_t NR = 30, NC = 10;
    auto nthreads = GetParam();
    auto thing = simulate_dense_matrix(NR, NC, /* seed = */ nthreads + 1000);

    tatami::DenseMatrix<double, int, tatami::ArrayView<double> > mat(NR, NC, tatami::ArrayView<double>(thing.data(), NC * NR), false);
    scran_pca::TransposedTatamiWrapperMatrix<Eigen::VectorXd, Eigen::MatrixXd, double, int> wrapped(mat, nthreads);
    EXPECT_EQ(wrapped.rows(), NC);
    EXPECT_EQ(wrapped.cols(), NR);

    Eigen::MatrixXd realized;
    auto realizer = wrapped.new_realize_workspace();
    realizer->realize_copy(realized);

    // Checking that the reference matches up.
    for (std::size_t c = 0; c < NC; ++c) {
        Eigen::VectorXd refcol = thing.col(c);
        Eigen::VectorXd obsrow = realized.row(c);
        EXPECT_EQ(refcol, obsrow);
    }

    // Trying in the normal orientation.
    {
        Eigen::VectorXd rhs = simulate_vector(NR, /* seed = */ nthreads + 2000);
        Eigen::VectorXd prod1(NC);
        auto wrk = wrapped.new_workspace();
        wrk->multiply(rhs, prod1);
        Eigen::VectorXd prod2 = thing.adjoint() * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        Eigen::VectorXd rhs = simulate_vector(NC, /* seed = */ nthreads + 3000);
        Eigen::VectorXd tprod1(NR);
        auto wrk = wrapped.new_adjoint_workspace();
        wrk->multiply(rhs, tprod1);
        Eigen::VectorXd tprod2 = thing * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

TEST_P(TransposedTatamiWrapperTest, DenseRow) {
    std::size_t NR = 30, NC = 10;
    auto nthreads = GetParam();
    auto thing = simulate_dense_matrix(NR, NC, /* seed = */ nthreads + 1000);

    tatami::DenseMatrix<double, int, tatami::ArrayView<double> > mat(NC, NR, tatami::ArrayView<double>(thing.data(), NC * NR), true);
    scran_pca::TransposedTatamiWrapperMatrix<Eigen::VectorXd, Eigen::MatrixXd, double, int> wrapped(mat, GetParam());
    EXPECT_EQ(wrapped.rows(), NR);
    EXPECT_EQ(wrapped.cols(), NC);

    Eigen::MatrixXd realized;
    auto realizer = wrapped.new_realize_workspace();
    realizer->realize_copy(realized);

    // Checking that the reference matches up.
    for (std::size_t c = 0; c < NC; ++c) {
        Eigen::VectorXd refcol = thing.col(c);
        Eigen::VectorXd obscol = realized.col(c);
        EXPECT_EQ(refcol, obscol);
    }

    // Trying in the normal orientation.
    {
        Eigen::VectorXd rhs = simulate_vector(NC, /* seed = */ nthreads + 2000);
        Eigen::VectorXd prod1(NR);
        auto wrk = wrapped.new_workspace();
        wrk->multiply(rhs, prod1);
        Eigen::VectorXd prod2 = thing * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        Eigen::VectorXd rhs = simulate_vector(NR, /* seed = */ nthreads + 3000);
        Eigen::VectorXd tprod1(NC);
        auto wrk = wrapped.new_adjoint_workspace();
        wrk->multiply(rhs, tprod1);
        Eigen::VectorXd tprod2 = thing.adjoint() * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

TEST_P(TransposedTatamiWrapperTest, SparseColumn) {
    std::size_t NR = 60, NC = 50;
    auto nthreads = GetParam();
    auto sim = simulate_sparse_triplets(NR, NC, /* seed = */ nthreads + 1000);

    auto thing = sparse_to_dense(NR, NC, sim);
    tatami::CompressedSparseMatrix<double, int, decltype(sim.values), decltype(sim.indices), decltype(sim.ptrs)> mat(NR, NC, sim.values, sim.indices, sim.ptrs, false);
    scran_pca::TransposedTatamiWrapperMatrix<Eigen::VectorXd, Eigen::MatrixXd, double, int> wrapped(mat, GetParam());
    EXPECT_EQ(wrapped.rows(), NC);
    EXPECT_EQ(wrapped.cols(), NR);

    Eigen::MatrixXd realized;
    auto realizer = wrapped.new_realize_workspace();
    realizer->realize_copy(realized);

    // Checking that the reference matches up.
    for (std::size_t c = 0; c < NC; ++c) {
        Eigen::VectorXd refcol = thing.col(c);
        Eigen::VectorXd obsrow = realized.row(c);
        EXPECT_EQ(refcol, obsrow);
    }

    // Trying in the normal orientation.
    {
        Eigen::VectorXd rhs = simulate_vector(NR, /* seed = */ nthreads + 2000);
        Eigen::VectorXd prod1(NC);
        auto wrk = wrapped.new_workspace();
        wrk->multiply(rhs, prod1);
        Eigen::VectorXd prod2 = thing.adjoint() * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        Eigen::VectorXd rhs = simulate_vector(NC, /* seed = */ nthreads + 3000);
        Eigen::VectorXd tprod1(NR);
        auto wrk = wrapped.new_adjoint_workspace();
        wrk->multiply(rhs, tprod1);
        Eigen::VectorXd tprod2 = thing * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

TEST_P(TransposedTatamiWrapperTest, SparseRow) {
    std::size_t NR = 60, NC = 50;
    auto nthreads = GetParam();
    auto sim = simulate_sparse_triplets(NR, NC, /* seed = */ nthreads + 1000);

    auto thing = sparse_to_dense(NR, NC, sim);
    tatami::CompressedSparseMatrix<double, int, decltype(sim.values), decltype(sim.indices), decltype(sim.ptrs)> mat(NC, NR, sim.values, sim.indices, sim.ptrs, true);
    scran_pca::TransposedTatamiWrapperMatrix<Eigen::VectorXd, Eigen::MatrixXd, double, int> wrapped(mat, GetParam());
    EXPECT_EQ(wrapped.rows(), NR);
    EXPECT_EQ(wrapped.cols(), NC);

    Eigen::MatrixXd realized;
    auto realizer = wrapped.new_realize_workspace();
    realizer->realize_copy(realized);

    // Checking that the reference matches up.
    for (std::size_t c = 0; c < NC; ++c) {
        Eigen::VectorXd refcol = thing.col(c);
        Eigen::VectorXd obscol = realized.col(c);
        EXPECT_EQ(refcol, obscol);
    }

    // Trying in the normal orientation.
    {
        Eigen::VectorXd rhs = simulate_vector(NC, /* seed = */ nthreads + 2000);
        Eigen::VectorXd prod1(NR);
        auto wrk = wrapped.new_workspace();
        wrk->multiply(rhs, prod1);
        Eigen::VectorXd prod2 = thing * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        Eigen::VectorXd rhs = simulate_vector(NR, /* seed = */ nthreads + 3000);
        Eigen::VectorXd tprod1(NC);
        auto wrk = wrapped.new_adjoint_workspace();
        wrk->multiply(rhs, tprod1);
        Eigen::VectorXd tprod2 = thing.adjoint() * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Utils,
    TransposedTatamiWrapperTest,
    ::testing::Values(1, 3)
);
