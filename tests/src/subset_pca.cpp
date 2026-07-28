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

static std::vector<int> choose_rows(const int NR, unsigned long seed) {
    std::vector<int> chosen;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<> udist;
    for (int r = 0; r < NR; ++r) {
        if (udist(rng) <= 0.2) {
            chosen.push_back(r);
        }
    }
    return chosen;
}

class SubsetPcaBasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int> > {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    static void SetUpTestSuite() {
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

    static void compare_results(
        const scran_pca::SubsetPcaResults<Eigen::MatrixXd, Eigen::VectorXd>& ref,
        const scran_pca::SubsetPcaResults<Eigen::MatrixXd, Eigen::VectorXd>& out,
        bool scale
    ) {
        expect_equal_pcs(ref.components, out.components);
        expect_equal_rotation(ref.rotation, out.rotation);
        expect_equal_vectors(ref.variance_explained, out.variance_explained);
        EXPECT_FLOAT_EQ(ref.total_variance, out.total_variance);
        expect_equal_vectors(ref.center, out.center);

        EXPECT_EQ(ref.scale.has_value(), out.scale.has_value());
        if (scale) {
            expect_equal_vectors(*(ref.scale), *(out.scale));
        }
    }
};

TEST_P(SubsetPcaBasicTest, Basic) {
    auto param = GetParam();
    bool scale = std::get<0>(param);
    int rank = std::get<1>(param);
    int threads = std::get<2>(param);

    scran_pca::SubsetPcaOptions sub_opt;
    sub_opt.scale = scale;
    sub_opt.number = rank;
    sub_opt.num_threads = threads;

    auto chosen = choose_rows(dense_row->nrow(), scale + rank + threads);
    auto subsetted = scran_pca::subset_pca(*dense_row, chosen, sub_opt);

    // Checking that we get more-or-less the same results with different matrix representations.
    auto res2 = scran_pca::subset_pca(*dense_column, chosen, sub_opt);
    compare_results(subsetted, res2, scale);

    auto res3 = scran_pca::subset_pca(*sparse_row, chosen, sub_opt);
    compare_results(subsetted, res3, scale);

    auto res4 = scran_pca::subset_pca(*sparse_column, chosen, sub_opt);
    compare_results(subsetted, res4, scale);

    // Checking that we get more-or-less the same results without matrix realization.
    sub_opt.realize_matrix = false;
    auto tres1 = scran_pca::subset_pca(*dense_row, chosen, sub_opt);
    compare_results(subsetted, tres1, scale);

    auto tres2 = scran_pca::subset_pca(*dense_column, chosen, sub_opt);
    compare_results(subsetted, tres2, scale);

    auto tres3 = scran_pca::subset_pca(*sparse_row, chosen, sub_opt);
    compare_results(subsetted, tres3, scale);

    auto tres4 = scran_pca::subset_pca(*sparse_column, chosen, sub_opt);
    compare_results(subsetted, tres4, scale);
}

TEST_P(SubsetPcaBasicTest, VersusReference) {
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

    tatami::DelayedBind<double, int> doubled_dr(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ dense_row, dense_row }, true);
    auto subsetted = scran_pca::subset_pca(doubled_dr, first_sub, opt);

    expect_equal_pcs(ref.components, subsetted.components);
    expect_equal_vectors(ref.variance_explained, subsetted.variance_explained);
    expect_equal_rotation(ref.rotation, subsetted.rotation.topRows(NR));
    expect_equal_rotation(ref.rotation, subsetted.rotation.bottomRows(NR), 1e-6); // expect more-or-less the same rotation matrices at the top and bottom.

    expect_equal_vectors(ref.center, subsetted.center.head(NR));
    expect_equal_vectors(ref.center, subsetted.center.tail(NR));
    if (scale) {
        expect_equal_vectors(*(ref.scale), subsetted.scale->head(NR));
        expect_equal_vectors(*(ref.scale), subsetted.scale->tail(NR));
    } else {
        EXPECT_FALSE(subsetted.scale.has_value());
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

class SubsetPcaEdgeTest : public ::testing::TestWithParam<bool> {};

TEST_P(SubsetPcaEdgeTest, OneCell) {
    const bool scale = GetParam();

    const int ngenes = 100;
    auto vec = scran_tests::simulate_vector(ngenes, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.lower = -10;
        sparams.upper = 10;
        sparams.seed = 3456;
        return sparams;
    }());
    tatami::DenseRowMatrix<double, int> mat(ngenes, 1, vec);

    scran_pca::SubsetPcaOptions opts;
    opts.number = 5;
    opts.scale = scale;

    auto chosen = choose_rows(mat.nrow(), /* seed = */ scale);
    auto res = scran_pca::subset_pca(mat, chosen, opts);

    // Checking that all values make sense.
    EXPECT_EQ(res.components.cols(), 1);
    EXPECT_EQ(res.components.rows(), 1);
    EXPECT_EQ(res.components.coeff(0, 0), 0);
    EXPECT_EQ(res.rotation.cols(), 1);
    EXPECT_EQ(res.rotation.rows(), ngenes);
    EXPECT_EQ(res.center.size(), ngenes);

    for (int g = 0; g < ngenes; ++g) {
        EXPECT_EQ(res.rotation.coeff(g, 0), (g == 0 ? 1 : 0));
        EXPECT_EQ(res.center.coeff(g), vec[g]);
    }

    EXPECT_EQ(res.variance_explained.size(), 1);
    EXPECT_EQ(res.variance_explained[0], 0);
    EXPECT_EQ(res.total_variance, 0);

    if (scale) {
        EXPECT_EQ(res.scale->size(), ngenes);
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_EQ((*(res.scale))[g], 1); // adjusted from zero to 1.
        }
    } else {
        EXPECT_FALSE(res.scale.has_value());
    }
}

TEST_P(SubsetPcaEdgeTest, NoCells) {
    const bool scale = GetParam();

    const int ngenes = 100;
    tatami::DenseRowMatrix<double, int> mat(ngenes, 0, std::vector<double>());
    auto chosen = choose_rows(ngenes, /* seed = */ scale);

    scran_pca::SubsetPcaOptions opts;
    opts.number = 5;
    opts.scale = scale;
    auto res = scran_pca::subset_pca(mat, chosen, opts);

    // Checking that all values make sense.
    EXPECT_EQ(res.components.cols(), 0);
    EXPECT_EQ(res.components.rows(), 0);
    EXPECT_EQ(res.rotation.cols(), 0);
    EXPECT_EQ(res.rotation.rows(), ngenes);
    EXPECT_EQ(res.center.size(), ngenes);

    for (int g = 0; g < ngenes; ++g) {
        EXPECT_EQ(res.center[g], 0);
    }

    EXPECT_EQ(res.variance_explained.size(), 0);
    EXPECT_EQ(res.total_variance, 0);

    if (scale) {
        EXPECT_EQ(res.scale->size(), ngenes);
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_EQ((*(res.scale))[g], 1); // adjusted from zero to 1.
        }
    } else {
        EXPECT_FALSE(res.scale.has_value());
    }
}

TEST_P(SubsetPcaEdgeTest, NoGenes) {
    const auto scale = GetParam();

    const int ncells = 100;
    tatami::DenseRowMatrix<double, int> mat(0, ncells, std::vector<double>());

    scran_pca::SubsetPcaOptions opts;
    opts.number = 5;
    opts.scale = scale;

    std::vector<int> chosen;
    auto res = scran_pca::subset_pca(mat, chosen, opts);

    // Checking that all values make sense.
    EXPECT_EQ(res.components.cols(), ncells);
    EXPECT_EQ(res.components.rows(), 0);
    EXPECT_EQ(res.rotation.cols(), 0);
    EXPECT_EQ(res.rotation.rows(), 0);
    EXPECT_EQ(res.center.size(), 0);
    EXPECT_EQ(res.variance_explained.size(), 0);
    EXPECT_EQ(res.total_variance, 0);

    if (scale) {
        EXPECT_EQ(res.scale->size(), 0);
    } else {
        EXPECT_FALSE(res.scale.has_value());
    }
}

TEST_P(SubsetPcaEdgeTest, AllSelected) {
    const bool scale = GetParam();

    const int ngenes = 100;
    const int ncells = 10;
    auto vec = scran_tests::simulate_vector(ngenes * ncells, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.lower = -10;
        sparams.upper = 10;
        sparams.seed = 3456;
        return sparams;
    }());
    tatami::DenseRowMatrix<double, int> mat(ngenes, ncells, std::move(vec));

    scran_pca::SubsetPcaOptions opts;
    opts.number = 5;
    opts.scale = scale;

    std::vector<int> chosen(ngenes);
    std::iota(chosen.begin(), chosen.end(), 0);
    auto res = scran_pca::subset_pca(mat, chosen, opts);
    auto ref = scran_pca::simple_pca(mat, opts);

    expect_equal_pcs(ref.components, res.components);
    expect_equal_rotation(ref.rotation, res.rotation);
    expect_equal_vectors(ref.variance_explained, res.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res.total_variance);
    expect_equal_vectors(ref.center, res.center);
    if (scale) {
        expect_equal_vectors((*(ref.scale)), (*(res.scale)));
    } else {
        EXPECT_FALSE(res.scale.has_value());
    }
}

TEST_P(SubsetPcaEdgeTest, NoneSelected) {
    const bool scale = GetParam();

    const int ngenes = 100;
    const int ncells = 10;
    auto vec = scran_tests::simulate_vector(ngenes * ncells, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.lower = -10;
        sparams.upper = 10;
        sparams.seed = 3456;
        return sparams;
    }());
    tatami::DenseRowMatrix<double, int> mat(ngenes, ncells, std::move(vec));

    scran_pca::SubsetPcaOptions opts;
    opts.number = 5;
    opts.scale = scale;

    std::vector<int> chosen;
    auto res = scran_pca::subset_pca(mat, chosen, opts);

    EXPECT_EQ(res.components.cols(), ncells);
    EXPECT_EQ(res.components.rows(), 0);
    EXPECT_EQ(res.rotation.cols(), 0);
    EXPECT_EQ(res.rotation.rows(), ngenes);
    EXPECT_EQ(res.variance_explained.size(), 0);
    EXPECT_EQ(res.total_variance, 0);

    auto varout = tatami_stats::variance(true, mat, {});
    scran_tests::compare_almost_equal_containers(varout.mean, res.center, {});

    if (scale) {
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_FLOAT_EQ(std::sqrt(varout.variance[g]), (*(res.scale))[g]);
        }
    } else {
        EXPECT_FALSE(res.scale.has_value());
    }
}

INSTANTIATE_TEST_SUITE_P(
    SubsetPca,
    SubsetPcaEdgeTest,
    ::testing::Values(false, true) // to scale or not to scale?
);

/**************************************************************/

class SubsetPcaBlockedTest : public ::testing::TestWithParam<std::tuple<bool, int, int, bool, int> > {
protected:
    inline static std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    static void SetUpTestSuite() {
        if (dense_row) {
            return;
        }

        size_t nr = 174, nc = 166;
        auto vec = scran_tests::simulate_vector(nr * nc, [&]() {
            scran_tests::SimulateVectorParameters sparams;
            sparams.density = 0.1;
            sparams.lower = -10;
            sparams.upper = 10;
            sparams.seed = 2026;
            return sparams;
        }());

        dense_row.reset(new tatami::DenseRowMatrix<double, int>(nr, nc, std::move(vec)));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_compressed_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_compressed_sparse(dense_row.get(), false);
    }

    static void compare_results(
        const scran_pca::SubsetPcaBlockedResults<Eigen::MatrixXd, Eigen::VectorXd>& ref,
        const scran_pca::SubsetPcaBlockedResults<Eigen::MatrixXd, Eigen::VectorXd>& out,
        bool scale
    ) {
        expect_equal_pcs(ref.components, out.components);
        expect_equal_rotation(ref.rotation, out.rotation);
        expect_equal_vectors(ref.variance_explained, out.variance_explained);
        EXPECT_FLOAT_EQ(ref.total_variance, out.total_variance);
        expect_equal_matrices(ref.center, out.center);

        EXPECT_EQ(ref.scale.has_value(), out.scale.has_value());
        if (scale) {
            expect_equal_vectors((*(ref.scale)), (*(out.scale)));
        }
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

    scran_pca::SubsetPcaBlockedOptions sub_opt;
    sub_opt.scale = scale;
    sub_opt.number = rank;
    sub_opt.num_threads = threads;
    if (weighted) {
        sub_opt.block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;
    } else {
        sub_opt.block_weight_policy = scran_blocks::WeightPolicy::NONE;
    }

    auto chosen = choose_rows(dense_row->nrow(), scale + rank + threads);
    auto subsetted = scran_pca::subset_pca_blocked(*dense_row, chosen, block.data(), nblocks, sub_opt);

    // Checking that we get more-or-less the same results with other representations. 
    auto res2 = scran_pca::subset_pca_blocked(*dense_column, chosen, block.data(), nblocks, sub_opt);
    compare_results(subsetted, res2, scale);

    auto res3 = scran_pca::subset_pca_blocked(*sparse_row, chosen, block.data(), nblocks, sub_opt);
    compare_results(subsetted, res3, scale);

    auto res4 = scran_pca::subset_pca_blocked(*sparse_column, chosen, block.data(), nblocks, sub_opt);
    compare_results(subsetted, res4, scale);

    // Checking that we get more-or-less the same results without matrix realization.
    sub_opt.realize_matrix = false;
    auto tres1 = scran_pca::subset_pca_blocked(*dense_row, chosen, block.data(), nblocks, sub_opt);
    compare_results(subsetted, tres1, scale);

    auto tres2 = scran_pca::subset_pca_blocked(*dense_column, chosen, block.data(), nblocks, sub_opt);
    compare_results(subsetted, tres2, scale);

    auto tres3 = scran_pca::subset_pca_blocked(*sparse_row, chosen, block.data(), nblocks, sub_opt);
    compare_results(subsetted, tres3, scale);

    auto tres4 = scran_pca::subset_pca_blocked(*sparse_column, chosen, block.data(), nblocks, sub_opt);
    compare_results(subsetted, tres4, scale);
}

TEST_P(SubsetPcaBlockedTest, VersusReference) {
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
    auto ref = scran_pca::blocked_pca(*dense_row, block.data(), nblocks, opt);

    const auto NR = dense_row->nrow();
    std::vector<int> first_sub(NR);
    std::iota(first_sub.begin(), first_sub.end(), 0);

    scran_pca::SubsetPcaBlockedOptions sub_opt;
    sub_opt.scale = scale;
    sub_opt.number = rank;
    sub_opt.num_threads = threads;
    sub_opt.irlba_options.convergence_tolerance = opt.irlba_options.convergence_tolerance;

    tatami::DelayedBind<double, int> doubled_dr(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ dense_row, dense_row }, true);
    auto subsetted = scran_pca::subset_pca_blocked(doubled_dr, first_sub, block.data(), nblocks, opt);

    expect_equal_pcs(ref.components, subsetted.components);
    expect_equal_vectors(ref.variance_explained, subsetted.variance_explained);
    expect_equal_rotation(ref.rotation, subsetted.rotation.topRows(NR));
    expect_equal_rotation(ref.rotation, subsetted.rotation.bottomRows(NR), 1e-6); // expect more-or-less the same rotation matrices at the top and bottom.

    expect_equal_matrices(ref.center, subsetted.center.leftCols(NR));
    expect_equal_matrices(ref.center, subsetted.center.rightCols(NR));
    if (scale) {
        expect_equal_vectors(*(ref.scale), subsetted.scale->head(NR));
        expect_equal_vectors(*(ref.scale), subsetted.scale->tail(NR));
    } else {
        EXPECT_FALSE(subsetted.scale.has_value());
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

class SubsetPcaBlockedEdgeTest : public ::testing::TestWithParam<bool> {};

TEST_P(SubsetPcaBlockedEdgeTest, OneCell) {
    const bool scale = GetParam();

    const int ngenes = 100;
    auto vec = scran_tests::simulate_vector(ngenes, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.lower = -10;
        sparams.upper = 10;
        sparams.seed = 3456;
        return sparams;
    }());
    tatami::DenseRowMatrix<double, int> mat(ngenes, 1, vec);

    scran_pca::SubsetPcaBlockedOptions opts;
    opts.number = 5;
    opts.scale = scale;

    auto chosen = choose_rows(mat.nrow(), /* seed = */ scale);
    std::vector<int> block(1);
    auto res = scran_pca::subset_pca_blocked(mat, chosen, block.data(), 1, opts);

    // Checking that all values make sense.
    EXPECT_EQ(res.components.cols(), 1);
    EXPECT_EQ(res.components.rows(), 1);
    EXPECT_EQ(res.components.coeff(0, 0), 0);
    EXPECT_EQ(res.rotation.cols(), 1);
    EXPECT_EQ(res.rotation.rows(), ngenes);
    EXPECT_EQ(res.center.cols(), ngenes);
    EXPECT_EQ(res.center.rows(), 1);

    for (int g = 0; g < ngenes; ++g) {
        EXPECT_EQ(res.rotation.coeff(g, 0), (g == 0 ? 1 : 0));
        EXPECT_EQ(res.center.coeff(g, 0), vec[g]);
    }

    EXPECT_EQ(res.variance_explained.size(), 1);
    EXPECT_EQ(res.variance_explained[0], 0);
    EXPECT_EQ(res.total_variance, 0);

    if (scale) {
        EXPECT_EQ(res.scale->size(), ngenes);
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_EQ((*(res.scale))[g], 1); // adjusted from zero to 1.
        }
    }
}

TEST_P(SubsetPcaBlockedEdgeTest, NoCells) {
    const bool scale = GetParam();

    const int ngenes = 100;
    tatami::DenseRowMatrix<double, int> mat(ngenes, 0, std::vector<double>());

    scran_pca::SubsetPcaBlockedOptions opts;
    opts.number = 5;
    opts.scale = scale;

    std::vector<int> block;
    auto chosen = choose_rows(mat.nrow(), /* seed = */ scale);
    auto res = scran_pca::subset_pca_blocked(mat, chosen, block.data(), 0, opts);

    // Checking that all values make sense.
    EXPECT_EQ(res.components.cols(), 0);
    EXPECT_EQ(res.components.rows(), 0);
    EXPECT_EQ(res.rotation.cols(), 0);
    EXPECT_EQ(res.rotation.rows(), ngenes);
    EXPECT_EQ(res.center.cols(), ngenes);
    EXPECT_EQ(res.center.rows(), 0);
    EXPECT_EQ(res.variance_explained.size(), 0);
    EXPECT_EQ(res.total_variance, 0);

    if (scale) {
        EXPECT_EQ(res.scale->size(), ngenes);
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_EQ((*(res.scale))[g], 1); // adjusted from zero to 1.
        }
    } else {
        EXPECT_FALSE(res.scale.has_value());
    }
}

TEST_P(SubsetPcaBlockedEdgeTest, NoGenes) {
    const auto scale = GetParam();

    const int ncells = 100;
    tatami::DenseRowMatrix<double, int> mat(0, ncells, std::vector<double>());

    scran_pca::SubsetPcaBlockedOptions opts;
    opts.number = 5;
    opts.scale = scale;

    std::vector<int> chosen;
    const int nblocks = 3;
    auto block = generate_blocks(mat.ncol(), nblocks);
    auto res = scran_pca::subset_pca_blocked(mat, chosen, block.data(), nblocks, opts);

    // Checking that all values make sense.
    EXPECT_EQ(res.components.cols(), ncells);
    EXPECT_EQ(res.components.rows(), 0);
    EXPECT_EQ(res.rotation.cols(), 0);
    EXPECT_EQ(res.rotation.rows(), 0);
    EXPECT_EQ(res.center.cols(), 0);
    EXPECT_EQ(res.center.rows(), nblocks);
    EXPECT_EQ(res.variance_explained.size(), 0);
    EXPECT_EQ(res.total_variance, 0);

    if (scale) {
        EXPECT_EQ(res.scale->size(), 0);
    } else {
        EXPECT_FALSE(res.scale.has_value());
    }
}

TEST_P(SubsetPcaBlockedEdgeTest, AllSelected) {
    const bool scale = GetParam();

    const int ngenes = 100;
    const int ncells = 10;
    auto vec = scran_tests::simulate_vector(ngenes * ncells, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.lower = -10;
        sparams.upper = 10;
        sparams.seed = 3456;
        return sparams;
    }());
    tatami::DenseRowMatrix<double, int> mat(ngenes, ncells, std::move(vec));

    scran_pca::SubsetPcaBlockedOptions opts;
    opts.number = 5;
    opts.scale = scale;

    std::vector<int> chosen(ngenes);
    std::iota(chosen.begin(), chosen.end(), 0);
    const int nblocks = 3;
    auto block = generate_blocks(mat.ncol(), nblocks);
    auto res = scran_pca::subset_pca_blocked(mat, chosen, block.data(), nblocks, opts);

    scran_pca::BlockedPcaOptions bopts;
    bopts.number = opts.number;
    bopts.scale = scale;
    auto ref = scran_pca::blocked_pca(mat, block.data(), nblocks, bopts);

    expect_equal_pcs(ref.components, res.components);
    expect_equal_rotation(ref.rotation, res.rotation);
    expect_equal_vectors(ref.variance_explained, res.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res.total_variance);
    expect_equal_matrices(ref.center, res.center);
    EXPECT_EQ(ref.scale.has_value(), res.scale.has_value());
    if (scale) {
        expect_equal_vectors((*(ref.scale)), (*(res.scale)));
    }
}

TEST_P(SubsetPcaBlockedEdgeTest, NoneSelected) {
    const bool scale = GetParam();

    const int ngenes = 100;
    const int ncells = 10;
    auto vec = scran_tests::simulate_vector(ngenes * ncells, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.lower = -10;
        sparams.upper = 10;
        sparams.seed = 3456;
        return sparams;
    }());
    tatami::DenseRowMatrix<double, int> mat(ngenes, ncells, std::move(vec));

    scran_pca::SubsetPcaBlockedOptions opts;
    opts.number = 5;
    opts.scale = scale;

    std::vector<int> chosen;
    const int nblocks = 3;
    auto block = generate_blocks(mat.ncol(), nblocks);
    auto res = scran_pca::subset_pca_blocked(mat, chosen, block.data(), nblocks, opts);

    EXPECT_EQ(res.components.cols(), ncells);
    EXPECT_EQ(res.components.rows(), 0);
    EXPECT_EQ(res.rotation.cols(), 0);
    EXPECT_EQ(res.rotation.rows(), ngenes);
    EXPECT_EQ(res.variance_explained.size(), 0);
    EXPECT_EQ(res.total_variance, 0);

    EXPECT_EQ(res.center.rows(), nblocks);
    EXPECT_EQ(res.center.cols(), ngenes);
    for (int b = 0; b < nblocks; ++b) {
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_NE(res.center.coeff(b, g), 0);
        }
    }

    if (scale) {
        EXPECT_EQ(res.scale->size(), ngenes);
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_GT(res.scale->coeff(g), 0);
        }
    } else {
        EXPECT_FALSE(res.scale.has_value());
    }
}

INSTANTIATE_TEST_SUITE_P(
    SubsetPca,
    SubsetPcaBlockedEdgeTest,
    ::testing::Values(false, true) // to scale or not to scale?
);

/**************************************************************/

TEST(SubsetPca, Errors) {
    tatami::DenseRowMatrix<double, int> mat(100, 10, std::vector<double>(1000));

    std::vector<int> first_sub(2);
    first_sub[0] = 1;

    scran_tests::expect_error(
        [&]() -> void {
            scran_pca::subset_pca(mat, first_sub, scran_pca::SubsetPcaOptions{});
        },
        "sorted"
    ); 

    auto block = generate_blocks(mat.ncol(), 2);
    scran_tests::expect_error(
        [&]() -> void {
            scran_pca::subset_pca_blocked(mat, first_sub, block.data(), 2, scran_pca::SubsetPcaBlockedOptions{});
        },
        "sorted"
    ); 
}
