#include <gtest/gtest.h>

#include "compare_pcs.h"

#include "scran_tests/scran_tests.hpp"
#include "tatami/tatami.hpp"

#include "scran_pca/simple_pca.hpp"

static void compare_results(
    const scran_pca::SimplePcaResults<Eigen::MatrixXd, Eigen::VectorXd>& ref,
    const scran_pca::SimplePcaResults<Eigen::MatrixXd, Eigen::VectorXd>& out,
    bool scale
) {
    expect_equal_pcs(ref.components, out.components);
    expect_equal_rotation(ref.rotation, out.rotation);
    expect_equal_vectors(ref.variance_explained, out.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, out.total_variance);
    expect_equal_vectors(ref.center, out.center);
    if (scale) {
        expect_equal_vectors(ref.scale, out.scale);
    }
}

/******************************************/

class SimplePcaBasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int> > {
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
            sparams.seed = 69;
            return sparams;
        }());

        dense_row.reset(new tatami::DenseRowMatrix<double, int>(nr, nc, std::move(vec)));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_compressed_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_compressed_sparse(dense_row.get(), false);
    }
};

TEST_P(SimplePcaBasicTest, Test) {
    auto param = GetParam();
    bool scale = std::get<0>(param);
    int rank = std::get<1>(param);
    int threads = std::get<2>(param);

    scran_pca::SimplePcaOptions opt;
    opt.scale = scale;
    opt.number = rank;
    auto ref = scran_pca::simple_pca(*dense_row, opt);

    if (threads == 1) {
        EXPECT_EQ(ref.variance_explained.size(), rank);
        EXPECT_EQ(ref.components.rows(), rank);
        EXPECT_EQ(ref.components.cols(), dense_row->ncol());

        // Checking that we scaled the PCs correctly.
        are_pcs_centered(ref.components);
        const int NC = dense_row->ncol();
        for (int r = 0; r < rank; ++r) {
            double var = 0;
            for (int c = 0; c < NC; ++c) {
                const auto x = ref.components.coeff(r, c);
                var += x * x;
            }
            var /= NC - 1;
            EXPECT_FLOAT_EQ(var, ref.variance_explained[r]);
        }

        if (scale) {
            EXPECT_FLOAT_EQ(dense_row->nrow(), ref.total_variance);
        } else {
            auto vars = tatami_stats::variance(true, *dense_row, {});
            auto total_var = std::accumulate(vars.variance.begin(), vars.variance.end(), 0.0);
            EXPECT_FLOAT_EQ(total_var, ref.total_variance);
        }

        auto sums = tatami_stats::sum(true, *dense_row, {});
        for (auto& ss : sums) {
            ss /= NC;
        }
        scran_tests::compare_almost_equal_containers(sums, ref.center, {});

        EXPECT_TRUE(ref.total_variance >= std::accumulate(ref.variance_explained.begin(), ref.variance_explained.end(), 0.0));

        if (scale) {
            EXPECT_EQ(ref.scale.size(), dense_row->nrow());
        } else {
            EXPECT_EQ(ref.scale.size(), 0);
        }

    } else {
        opt.num_threads = threads;
        auto res1 = scran_pca::simple_pca(*dense_row, opt);
        compare_results(ref, res1, scale);
    }

    // Checking that we get more-or-less the same results. 
    auto res2 = scran_pca::simple_pca(*dense_column, opt);
    compare_results(ref, res2, scale);

    auto res3 = scran_pca::simple_pca(*sparse_row, opt);
    compare_results(ref, res3, scale);

    auto res4 = scran_pca::simple_pca(*sparse_column, opt);
    compare_results(ref, res4, scale);

    // Checking that we get more-or-less the same results. 
    opt.realize_matrix = false;

    auto tres1 = scran_pca::simple_pca(*dense_row, opt);
    compare_results(ref, tres1, scale);

    auto tres2 = scran_pca::simple_pca(*dense_column, opt);
    compare_results(ref, tres2, scale);

    auto tres3 = scran_pca::simple_pca(*sparse_row, opt);
    compare_results(ref, tres3, scale);

    auto tres4 = scran_pca::simple_pca(*sparse_column, opt);
    compare_results(ref, tres4, scale);
}

INSTANTIATE_TEST_SUITE_P(
    SimplePca,
    SimplePcaBasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 5, 10), // number of PCs to obtain
        ::testing::Values(1, 3) // number of threads
    )
);

/******************************************/

class SimplePcaMoreTest : public ::testing::TestWithParam<std::tuple<bool, int> > {};

TEST_P(SimplePcaMoreTest, ZeroVariance) {
    auto param = GetParam();
    bool scale = std::get<0>(param);
    int rank = std::get<1>(param);

    // Injecting all-zeros into the last row. 
    size_t nr = 109, nc = 153;
    auto vec = scran_tests::simulate_vector(nr * nc, [&]() {
        scran_tests::SimulateVectorParameters sparams;
        sparams.density = 0.1;
        sparams.lower = -10;
        sparams.upper = 10;
        sparams.seed = scale * 100 + rank;
        return sparams;
    }());
    std::size_t last_row = (nr - 1) * nc;
    std::fill_n(vec.begin() + last_row, nc, 0);

    auto dense_row = std::make_unique<tatami::DenseRowMatrix<double, int> >(nr, nc, vec);
    auto dense_column = tatami::convert_to_dense<double, int>(*dense_row, false, {});
    auto sparse_row = tatami::convert_to_compressed_sparse<double, int>(*dense_row, true, {});
    auto sparse_column = tatami::convert_to_compressed_sparse<double, int>(*dense_row, false, {});

    std::vector<double> removed(vec.begin(), vec.begin() + last_row);
    tatami::DenseRowMatrix<double, int> leftovers(nr - 1, nc, std::move(removed));

    scran_pca::SimplePcaOptions opt;
    opt.scale = scale;
    opt.number = rank;

    // The initial vector is slightly different when we lose a feature, so we manually force our own random initialization.
    auto raw_init = scran_tests::simulate_vector(nr, [&]() {
        scran_tests::SimulateVectorParameters sparams;
        sparams.lower = -2;
        sparams.upper = 2;
        sparams.seed = scale * 10 + rank;
        return sparams;
    }());

    raw_init.back() = 0;

    Eigen::VectorXd init(nr - 1);
    std::copy_n(raw_init.begin(), nr - 1, init.data());
    opt.irlba_options.initial = init;
    auto ref = scran_pca::simple_pca(leftovers, opt);

    Eigen::VectorXd init2(nr);
    std::copy_n(raw_init.begin(), nr, init2.data());
    opt.irlba_options.initial = init2;
    auto res1 = scran_pca::simple_pca(*dense_row, opt);

    expect_equal_pcs(ref.components, res1.components);
    expect_equal_rotation(ref.rotation, res1.rotation.topRows(nr - 1));
    EXPECT_EQ(res1.rotation.row(nr - 1).squaredNorm(), 0);

    expect_equal_vectors(ref.variance_explained, res1.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res1.total_variance);

    expect_equal_vectors(ref.center, res1.center.head(nr - 1));
    EXPECT_EQ(res1.center[nr - 1], 0);

    if (scale) {
        expect_equal_vectors(ref.scale, res1.scale.head(nr - 1));
        EXPECT_EQ(res1.scale[nr - 1], 1);
    }

    // Checking that we get more-or-less the same results. 
    auto res2 = scran_pca::simple_pca(*dense_column, opt);
    compare_results(res1, res2, scale);

    auto res3 = scran_pca::simple_pca(*sparse_row, opt);
    compare_results(res1, res3, scale);

    auto res4 = scran_pca::simple_pca(*sparse_column, opt);
    compare_results(res1, res4, scale);

    // Checking that we get more-or-less the same results. 
    opt.realize_matrix = false;

    auto tres1 = scran_pca::simple_pca(*dense_row, opt);
    compare_results(res1, tres1, scale);

    auto tres2 = scran_pca::simple_pca(*dense_column, opt);
    compare_results(res1, tres2, scale);

    auto tres3 = scran_pca::simple_pca(*sparse_row, opt);
    compare_results(res1, tres3, scale);

    auto tres4 = scran_pca::simple_pca(*sparse_column, opt);
    compare_results(res1, tres4, scale);
}

INSTANTIATE_TEST_SUITE_P(
    SimplePca,
    SimplePcaMoreTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 5, 10) // number of PCs to obtain
    )
);
