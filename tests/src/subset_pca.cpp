#include <gtest/gtest.h>

#include "compare_pcs.h"
#include "utils.h"

#include "scran_tests/scran_tests.hpp"
#include "tatami/tatami.hpp"

#include "scran_pca/subset_pca.hpp"

TEST(InvertSubset, Basic) {
    {
        std::vector<int> sub{ 0, 1, 4 };
        auto out = scran_pca::invert_subset(5, sub);
        std::vector<int> expected { 2, 3 };
        EXPECT_EQ(out, expected);
    }

    {
        std::vector<unsigned> sub{ 2, 3 };
        auto out = scran_pca::invert_subset(5, sub);
        std::vector<int> expected { 0, 1, 4 };
        EXPECT_EQ(out, expected);
    }

    {
        std::vector<unsigned> sub;
        auto out = scran_pca::invert_subset(5, sub);
        std::vector<int> expected { 0, 1, 2, 3, 4 };
        EXPECT_EQ(out, expected);
    }

    {
        std::vector<int> sub{0, 1, 2, 3, 4};
        auto out = scran_pca::invert_subset(5, sub);
        EXPECT_TRUE(out.empty());
    }
}

TEST(Expand, Vector) {
    Eigen::VectorXd foo(10);
    for (int i = 0; i < 10; ++i) {
        foo.coeffRef(i) = (i + 1) * 10;
    }

    std::vector<int> sub{ 1, 3, 7, 8 };
    Eigen::VectorXd bar(sub.size());
    for (int i = 0; i < 4; ++i) {
        bar[i] = (sub[i] + 1) * 10;
    }

    auto copy = foo;
    for (auto s : sub) {
        copy.coeffRef(s) = -100;
    }
    scran_pca::expand_into_vector(sub, bar, copy);
    expect_equal_vectors(copy, foo);
}

TEST(Expand, MatrixRow) {
    Eigen::MatrixXd foo(10, 3);
    for (int r = 0; r < 10; ++r) {
        int val = (r + 1) * 10;
        for (int c = 0; c < 3; ++c) {
            foo.coeffRef(r, c) = val + c;
        }
    }

    std::vector<int> sub{ 2, 5, 6, 9 };
    Eigen::MatrixXd bar(sub.size(), 3);
    for (int i = 0; i < 4; ++i) {
        int val = (sub[i] + 1) * 10;
        for (int c = 0; c < 3; ++c) {
            bar.coeffRef(i, c) = val + c;
        }
    }

    auto copy = foo;
    for (auto s : sub) {
        for (int c = 0; c < 3; ++c) {
            copy.coeffRef(s, c) = -100;
        }
    }
    scran_pca::expand_into_matrix_rows(sub, bar, copy);
    expect_equal_matrices(copy, foo);
}

TEST(Expand, MatrixColumn) {
    Eigen::MatrixXd foo(3, 10);
    for (int c = 0; c < 10; ++c) {
        int val = (c + 1) * 10;
        for (int r = 0; r < 3; ++r) {
            foo.coeffRef(r, c) = val + r;
        }
    }

    std::vector<int> sub{ 0, 1, 3, 7, 8 };
    Eigen::MatrixXd bar(3, sub.size());
    for (int i = 0; i < 5; ++i) {
        int val = (sub[i] + 1) * 10;
        for (int r = 0; r < 3; ++r) {
            bar.coeffRef(r, i) = val + r;
        }
    }

    auto copy = foo;
    for (auto s : sub) {
        for (int r = 0; r < 3; ++r) {
            copy.coeffRef(r, s) = -100;
        }
    }
    scran_pca::expand_into_matrix_columns(sub, bar, copy);
    expect_equal_matrices(copy, foo);
}

/**************************************************************/

class SubsetPcaTestCore {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    static void assemble() {
        if (dense_row) {
            return;
        }

        size_t nr = 199, nc = 165;
        auto vec = scran_tests::simulate_vector(nr * nc, [&]() {
            scran_tests::SimulateVectorParameters sparams;
            sparams.density = 0.1;
            sparams.lower = -10;
            sparams.upper = 10;
            sparams.seed = 2025;
            return sparams;
        }());

        dense_row.reset(new tatami::DenseRowMatrix<double, int>(nr, nc, std::move(vec)));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_compressed_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_compressed_sparse(dense_row.get(), false);
    }
};

/**************************************************************/

class SubsetPcaBasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int> >, public SubsetPcaTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(SubsetPcaBasicTest, Basic) {
    auto param = GetParam();
    bool scale = std::get<0>(param);
    int rank = std::get<1>(param);
    int threads = std::get<2>(param);

    scran_pca::SimplePcaOptions opt;
    opt.scale = scale;
    opt.number = rank;
    opt.num_threads = threads;
    // We tightened the tolerances so that the rotation matrix comparisons are more accurate,
    // as otherwise the rotation matrix is not quite equal to the matrix * components product.
    opt.irlba_options.convergence_tolerance = 1e-12; 
    auto ref = scran_pca::simple_pca(*dense_row, opt);

    const auto NR = dense_row->nrow();
    std::vector<int> first_sub(NR);
    std::iota(first_sub.begin(), first_sub.end(), 0);

    scran_pca::SubsetPcaOptions sub_opt;
    sub_opt.scale = scale;
    sub_opt.number = rank;
    sub_opt.num_threads = threads;
    sub_opt.irlba_options.convergence_tolerance = opt.irlba_options.convergence_tolerance;

    for (int i = 0; i < 4; ++i) {
        std::shared_ptr<tatami::NumericMatrix> ptr; 
        if (i == 1) {
            ptr = dense_row;
        } else if (i == 2) {
            ptr = dense_column;
        } else if (i == 3) {
            ptr = sparse_row;
        } else {
            ptr = sparse_column;
        }
        tatami::DelayedBind<double, int> doubled(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ ptr, ptr }, true);
        auto subsetted = scran_pca::subset_pca(doubled, first_sub, opt);

        expect_equal_pcs(ref.components, subsetted.components);
        expect_equal_vectors(ref.variance_explained, subsetted.variance_explained);
        expect_equal_rotation(ref.rotation, subsetted.rotation.topRows(NR));
        expect_equal_rotation(ref.rotation, subsetted.rotation.bottomRows(NR), 1e-6); // expect more-or-less the same rotation matrices at the top and bottom.

        expect_equal_vectors(ref.center, subsetted.center.head(NR));
        expect_equal_vectors(ref.center, subsetted.center.tail(NR));
        if (scale) {
            expect_equal_vectors(ref.scale, subsetted.scale.head(NR));
            expect_equal_vectors(ref.scale, subsetted.scale.tail(NR));
        } else {
            EXPECT_EQ(subsetted.scale.size(), 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SubsetPca,
    SubsetPcaBasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 5, 10), // number of PCs to obtain
        ::testing::Values(1, 3) // number of threads
    )
);

/**************************************************************/

class SubsetPcaBlockedTest : public ::testing::TestWithParam<std::tuple<bool, int, int, bool, int> >, public SubsetPcaTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_P(SubsetPcaBlockedTest, Basic) {
    auto param = GetParam();
    bool scale = std::get<0>(param);
    int rank = std::get<1>(param);
    int nblocks = std::get<2>(param);
    bool weighted = std::get<3>(param);
    int threads = std::get<4>(param);
    auto block = generate_blocks(dense_row->ncol(), nblocks);

    scran_pca::BlockedPcaOptions opt;
    opt.scale = scale;
    opt.number = rank;
    opt.num_threads = threads;
    if (weighted) {
        opt.block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;
    } else {
        opt.block_weight_policy = scran_blocks::WeightPolicy::NONE;
    }

    // We tightened the tolerances so that the rotation matrix comparisons are more accurate,
    // as otherwise the rotation matrix is not quite equal to the matrix * components product.
    opt.irlba_options.convergence_tolerance = 1e-12; 
    auto ref = scran_pca::blocked_pca(*dense_row, block.data(), opt);

    const auto NR = dense_row->nrow();
    std::vector<int> first_sub(NR);
    std::iota(first_sub.begin(), first_sub.end(), 0);

    scran_pca::SubsetPcaBlockedOptions sub_opt;
    sub_opt.scale = scale;
    sub_opt.number = rank;
    sub_opt.num_threads = threads;
    sub_opt.irlba_options.convergence_tolerance = opt.irlba_options.convergence_tolerance;

    for (int i = 0; i < 4; ++i) {
        std::shared_ptr<tatami::NumericMatrix> ptr; 
        if (i == 1) {
            ptr = dense_row;
        } else if (i == 2) {
            ptr = dense_column;
        } else if (i == 3) {
            ptr = sparse_row;
        } else {
            ptr = sparse_column;
        }
        tatami::DelayedBind<double, int> doubled(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ ptr, ptr }, true);
        auto subsetted = scran_pca::subset_pca_blocked(doubled, first_sub, block.data(), opt);

        expect_equal_pcs(ref.components, subsetted.components);
        expect_equal_vectors(ref.variance_explained, subsetted.variance_explained);
        expect_equal_rotation(ref.rotation, subsetted.rotation.topRows(NR));
        expect_equal_rotation(ref.rotation, subsetted.rotation.bottomRows(NR), 1e-6); // expect more-or-less the same rotation matrices at the top and bottom.

        expect_equal_matrices(ref.center, subsetted.center.leftCols(NR));
        expect_equal_matrices(ref.center, subsetted.center.rightCols(NR));
        if (scale) {
            expect_equal_vectors(ref.scale, subsetted.scale.head(NR));
            expect_equal_vectors(ref.scale, subsetted.scale.tail(NR));
        } else {
            EXPECT_EQ(subsetted.scale.size(), 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SubsetPca,
    SubsetPcaBlockedTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 5, 10), // number of PCs to obtain
        ::testing::Values(1, 2, 3), // number of blocks
        ::testing::Values(true, false), // weighted or not
        ::testing::Values(1, 3) // number of threads
    )
);

/**************************************************************/

class SubsetPcaErrorTest : public ::testing::Test, public SubsetPcaTestCore {
protected:
    static void SetUpTestSuite() {
        assemble();
    }
};

TEST_F(SubsetPcaErrorTest, Basic) {
    std::vector<int> first_sub(2);
    first_sub[0] = 1;

    scran_tests::expect_error(
        [&]() -> void {
            scran_pca::subset_pca(*dense_row, first_sub, scran_pca::SubsetPcaOptions{});
        },
        "sorted"
    ); 

    auto block = generate_blocks(dense_row->ncol(), 2);
    scran_tests::expect_error(
        [&]() -> void {
            scran_pca::subset_pca_blocked(*dense_row, first_sub, block.data(), scran_pca::SubsetPcaBlockedOptions{});
        },
        "sorted"
    ); 
}
