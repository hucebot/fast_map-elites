#include <type_traits>

#include "map_elites.hpp"

// def square(theta):
//     freq = 5
//     b = np.sin(theta * freq)
//     f = 1. - np.prod(theta)
//     return f,b

// def checkered(theta):
//     freq = 5
//     b = np.sin(theta * freq)
//     f = np.prod(np.sin(theta * 50))
//     return f,b

// def line(theta):
//     freq = 10
//     x = theta[0] * theta[1]
//     b = [x*math.cos(x),x*math.sin(x) ]
//     f = 1-np.std(theta)
//     return f,b

// #
// #https://math.stackexchange.com/questions/3135263/approximate-rounding-to-nearest-integer-with-a-continuous-function
// # https://math.stackexchange.com/questions/2033727/how-do-i-approximate-a-staircase-function-rounding-with-triangle-wave
// # https://stackoverflow.com/questions/46596636/differentiable-round-function-in-tensorflow
// def cisland(theta):
//     #theta = np.random.rand(len(theta))
//     b = theta - np.sin(20 * math.pi * theta) / (20 * math.pi)
//     f = theta[0]*theta[1]
//     return f,b

// def island(theta):
//     b = np.around(theta, 1)
//     f = theta[0]*theta[1]
//     return f,b

// def circle(theta):
//     def gaussian(x, mu, sig):
//         return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
//     freq = 40
//     c = np.ones(len(theta)) * 0.5
//     n = np.linalg.norm(theta - c)
//     b = np.sin(theta * freq)
//     f = gaussian(n, 0.5, 0.3)
//     return f,b

namespace benchmark {
    template <typename Params, typename S = double>
    struct Square {
        static_assert(Params::dim_search_space == Params::dim_features, 
            "In benchmark::square, the dimension of features is equal to the dimension of the feature space");
        using indiv_t = Eigen::Matrix<S, 1, Params::dim_search_space, Eigen::RowMajor>;
        using features_t = Eigen::Matrix<S, 1, Params::dim_features, Eigen::RowMajor>;
        static constexpr double freq = 5.0;

        // keep the intermediate values to avoid reallocations
        indiv_t _t;
        indiv_t _c;
        features_t _features;

        const features_t& eval(const indiv_t& v, S& fit)
        {
            _features = (freq * v).array().sin() * 0.5 + 1.;
            fit = 1 - v.prod();
            return _features;
        }
    };
} // namespace benchmark

struct Params {
    static constexpr int dim_features = 2;
    static constexpr int dim_search_space = 2;
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
    using fit_t = benchmark::Square<Params, float>;
    using map_elites_t = map_elites::MapElites<Params, fit_t, float>;

    auto start = std::chrono::high_resolution_clock::now();
    map_elites_t map_elites;

    std::ofstream qd_ofs("qd.dat");

    for (size_t i = 0; i < 1e6 / Params::batch_size; ++i) {
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