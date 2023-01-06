#include "map_elites.hpp"

template <typename Params, typename S = double>
struct FitArm {
    using indiv_t = Eigen::Matrix<S, 1, Params::dim_search_space, Eigen::RowMajor>;
    using features_t = Eigen::Matrix<S, 1, Params::dim_features, Eigen::RowMajor>;
    // keep the intermediate values to avoid reallocations
    indiv_t _t;
    indiv_t _c;
    features_t _features;

    const features_t& eval(const indiv_t& v, S& fit)
    {
        fit = 1 - std::sqrt((v.array() - v.mean()).square().sum() / (v.size() - 1.0));
        _t = 2 * M_PI * v.array() - M_PI;
        _c = indiv_t::Zero();
        std::partial_sum(_t.begin(), _t.end(), _c.begin(), std::plus<double>());
        _features[0] = _c.array().cos().sum() / (2. * v.size()) + 0.5;
        _features[1] = _c.array().sin().sum() / (2. * v.size()) + 0.5;
        return _features;
    }
};

struct Params {
    static constexpr int dim_features = 2;
    static constexpr int dim_search_space = 10;
    static constexpr int batch_size = 128;
    static constexpr double sigma_1 = 0.15;
    static constexpr double sigma_2 = 0.01;
    static constexpr double infill_pct = 0.2;
    static constexpr bool verbose = false;
    static constexpr bool grid = true;
    static constexpr int grid_size = 64;
    static constexpr int num_cells = grid ? grid_size * grid_size : 12000; // 12000; // 8192;
};

int main()
{
    using fit_t = FitArm<Params, float>;
    using map_elites_t = map_elites::MapElites<Params, fit_t, float>;

    auto start = std::chrono::high_resolution_clock::now();
    map_elites_t map_elites;

    std::ofstream qd_ofs("qd.dat");

    for (size_t i = 0; i < 0.5 * 1e6 / Params::batch_size; ++i) {
        map_elites.step();
        qd_ofs << i * Params::batch_size << " " << map_elites.qd_score() << std::endl;
        if (Params::verbose)
            std::cout << i << " ";
        std::cout.flush();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Total time:" << t / 1000.0 << "s" << std::endl;

    std::cout << "writing...";
    std::cout.flush();
    std::ofstream c("centroids.dat");
    c << map_elites.centroids() << std::endl;
    std::ofstream f("fit.dat");
    f << map_elites.archive_fit() << std::endl;
    std::cout << "done" << std::endl;
    return 0;
}