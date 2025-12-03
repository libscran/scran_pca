#include <gtest/gtest.h>

#include "compare_pcs.h"

#include "scran_tests/scran_tests.hpp"
#include "tatami/tatami.hpp"

#include "scran_pca/subset_pca.hpp"

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
            sparams.seed = 69;
            return sparams;
        }());

        dense_row.reset(new tatami::DenseRowMatrix<double, int>(nr, nc, std::move(vec)));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_compressed_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_compressed_sparse(dense_row.get(), false);
    }
};

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
    opt.irlba_options.convergence_tolerance = 1e-10; 
    auto ref = scran_pca::simple_pca(*dense_row, opt);

    const auto NR = dense_row->nrow();
    std::vector<int> first_sub(NR);
    std::iota(first_sub.begin(), first_sub.end(), 0);

    scran_pca::SubsetPcaOptions sub_opt;
    sub_opt.scale = scale;
    sub_opt.number = rank;
    sub_opt.num_threads = threads;
    sub_opt.irlba_options.convergence_tolerance = opt.irlba_options.convergence_tolerance;

    {
        tatami::DelayedBind<double, int> dr_doubled(std::vector<std::shared_ptr<tatami::Matrix<double, int> > >{ dense_row, dense_row }, true);
        auto subsetted = scran_pca::subset_pca(dr_doubled, first_sub, opt);

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

    {
        tatami::DelayedBind<double, int> dc_doubled(std::vector<std::shared_ptr<tatami::Matrix<double, int> > >{ dense_column, dense_column }, true);
        auto subsetted = scran_pca::subset_pca(dc_doubled, first_sub, opt);

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

    {
        tatami::DelayedBind<double, int> sr_doubled(std::vector<std::shared_ptr<tatami::Matrix<double, int> > >{ sparse_row, sparse_row }, true);
        auto subsetted = scran_pca::subset_pca(sr_doubled, first_sub, opt);

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

    {
        tatami::DelayedBind<double, int> sc_doubled(std::vector<std::shared_ptr<tatami::Matrix<double, int> > >{ sparse_column, sparse_column }, true);
        auto subsetted = scran_pca::subset_pca(sc_doubled, first_sub, opt);

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
