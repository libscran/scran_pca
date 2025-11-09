#ifndef UTILS_H
#define UTILS_H

inline std::vector<int> generate_blocks(int nobs, int nblocks) {
    std::vector<int> blocks(nobs);
    for (int i = 0; i < nobs; ++i) {
        blocks[i] = i % nblocks;
    }
    return blocks;
}

inline Eigen::MatrixXd simulate_dense_matrix(Eigen::Index NR, Eigen::Index NC, unsigned long long seed) {
    Eigen::MatrixXd thing(NR, NC);  
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist;
    for (Eigen::Index r = 0; r < NR; ++r) {
        for (Eigen::Index c = 0; c < NC; ++c) {
            thing(r, c) = dist(rng);
        }
    }
    return thing;
}

inline Eigen::MatrixXd simulate_vector(Eigen::Index len, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist;
    Eigen::VectorXd vec(len);
    for (auto& v : vec) {
        v = dist(rng);
    }
    return vec;
}

typedef scran_tests::SimulatedCompressedSparseMatrix<double, int, std::size_t> SparseTriplets;

inline Eigen::MatrixXd sparse_to_dense(Eigen::Index NR, Eigen::Index NC, const SparseTriplets& simulated) {
    Eigen::MatrixXd output(NR, NC);  
    output.setZero();

    for (Eigen::Index c = 0; c < NC; ++c) {
        const auto start = simulated.pointers[c], end = simulated.pointers[c + 1];
        for (std::size_t i = start; i < end; ++i) {
            output.coeffRef(simulated.index[i], c) = simulated.data[i];
        }
    }

    return output;
}

inline std::vector<std::shared_ptr<tatami::NumericMatrix> > fragment_matrices_by_block(const std::shared_ptr<tatami::NumericMatrix>& x, const std::vector<int>& block, int nblocks) {
    std::vector<std::shared_ptr<tatami::NumericMatrix> > collected;

    for (int b = 0; b < nblocks; ++b) {
        std::vector<int> keep;

        for (size_t i = 0; i < block.size(); ++i) {
            if (block[i] == b) {
                keep.push_back(i);
            }
        }

        if (keep.size() > 1) {
            collected.push_back(tatami::make_DelayedSubset<1>(x, keep));
        }
    }

    return collected;
}

#endif
