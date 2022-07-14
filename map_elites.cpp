#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <thread>

#ifdef USE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

#include <Eigen/Core>

template <typename Params, typename Fit, typename S = double>
class MapElites {
public:
    using centroids_t = Eigen::Matrix<S, Params::num_cells, Params::dim_features>;
    using archive_t = Eigen::Matrix<S, Params::num_cells, Params::dim_search_space>;
    using archive_fit_t = Eigen::Vector<S, Params::num_cells>;

    MapElites() : _centroids(centroids_t::Random()), _archive(archive_t::Random())
    {
        if (Params::grid) {
            assert(Params::dim_features == 2);
            for (int i = 0; i < Params::grid_size; ++i)
                for (int j = 0; j < Params::grid_size; ++j) {
                    _centroids(i * Params::grid_size + j, 0) = double(i) / Params::grid_size;
                    _centroids(i * Params::grid_size + j, 1) = double(j) / Params::grid_size;
                }
        }
    }

    const archive_t& archive() const { return _archive; }
    const archive_fit_t& archive_fit() const { return _archive_fit; }
    const centroids_t& centroids() const { return _centroids; }
    double qd_score() const { 
        double qd = 0;
        for (int i = 0; i < Params::num_cells; ++i)
            if (_archive_fit(i) != -std::numeric_limits<S>::max())
                qd += _archive_fit(i);
        return qd;
    }
    void step()
    {
        for (int i = 0; i < Params::batch_size * 2; ++i)
            _batch_ranks[i] = _rand_int(_r_gen); // yes, from all the map!
        for (int i = 0; i < Params::batch_size; ++i) // line variation
            _batch.row(i) = _archive.row(_batch_ranks[i * 2]) + Params::sigma_1 * _gaussian(_r_gen) * (_archive.row(_batch_ranks[i * 2]) - _archive.row(_batch_ranks[i * 2 + 1]));
        for (int i = 0; i < Params::batch_size; ++i) // gaussian mutation
            for (int j = 0; j < Params::dim_search_space; ++j)
                _batch(i, j) += _gaussian(_r_gen) * Params::sigma_2;
        for (int i = 0; i < Params::batch_size; ++i) // clip in [0,1]
            _batch.row(i) = _batch.row(i).cwiseMin(1).cwiseMax(0);

        _loop(0, Params::batch_size, [&](int i) { // evaluate the batch
            _batch_features.row(i) = _fit_functions[i].eval(_batch.row(i), _batch_fitness(i));
        });
        // competition
        std::fill(_new_rank.begin(), _new_rank.end(), -1);
        _loop(0, Params::batch_size, [&](int i) {
            // search for the closest centroid / the grid
            int best_i = -1;
            if (Params::grid) {
                int x = round(_batch_features(i, 0) * (Params::grid_size - 1));
                int y = round(_batch_features(i, 1) * (Params::grid_size - 1));
                best_i = x * Params::grid_size + y;
            }
            else {
                double best_dist = std::numeric_limits<S>::max();
                for (int j = 0; j < Params::num_cells; ++j) {
                    double d = (_batch_features.row(i) - _centroids.row(j)).squaredNorm();
                    if (d < best_dist) {
                        best_dist = d;
                        best_i = j;
                    }
                }
            }
            if (_batch_fitness.row(i)[0] > _archive_fit(best_i))
                _new_rank[i] = best_i;
        });
        // apply the new ranks
        for (int i = 0; i < Params::batch_size; ++i) {
            if (_new_rank[i] != -1) {
                _archive.row(_new_rank[i]) = _batch.row(i);
                _archive_fit(_new_rank[i]) = _batch_fitness(i);
            }
        }
    }

protected:
    // to make it easy to switch between parallel and non-parallel
    template <typename F>
    inline void _loop(size_t begin, size_t end, const F& f)
    {
#ifdef USE_TBB
        tbb::parallel_for(size_t(begin), end, size_t(1), [&](size_t i) {
            f(i);
        });
#else
        for (size_t i = begin; i < end; ++i)
            f(i);
#endif
    }
    // our main data
    centroids_t _centroids;
    archive_t _archive;
    archive_fit_t _archive_fit = archive_fit_t::Constant(-std::numeric_limits<S>::max());

    // batch
    using batch_t = Eigen::Matrix<S, Params::batch_size, Params::dim_search_space>;
    using batch_fit_t = Eigen::Vector<S, Params::batch_size>;
    using batch_features_t = Eigen::Matrix<S, Params::batch_size, Params::dim_features>;
    using fit_functions_t = std::array<Fit, Params::batch_size>;
    using batch_ranks_t = Eigen::Vector<int, Params::batch_size * 2>;
    using new_rank_t = Eigen::Vector<int, Params::batch_size>;

    batch_t _batch;
    batch_fit_t _batch_fitness;
    batch_features_t _batch_features;
    batch_ranks_t _batch_ranks;
    fit_functions_t _fit_functions;
    new_rank_t _new_rank;

    // random
    std::random_device _rand_device;
    std::default_random_engine _r_gen{_rand_device()};
    std::uniform_int_distribution<int> _rand_int{0, Params::num_cells - 1};
    std::normal_distribution<> _gaussian{0, 1};
};

template <typename Params, typename S = double>
struct FitArm {
    using indiv_t = Eigen::Matrix<S, 1, Params::dim_search_space>;
    using features_t = Eigen::Matrix<S, 1, Params::dim_features>;
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
    static constexpr int batch_size = 512;
    static constexpr double sigma_1 = 0.35;
    static constexpr double sigma_2 = 0.01;
    static constexpr bool verbose = false;
    static constexpr bool grid = true;
    static constexpr int grid_size = 64;
    static constexpr int num_cells = grid ? grid_size * grid_size : 12000; // 12000; // 8192;
};

int main()
{
    using fit_t = FitArm<Params, float>;
    using map_elites_t = MapElites<Params, fit_t, float>;

    auto start = std::chrono::high_resolution_clock::now();
    map_elites_t map_elites;

    std::ofstream qd_ofs("qd.dat");

    for (size_t i = 0; i < 5e5 / Params::batch_size; ++i) {
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