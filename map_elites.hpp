#ifndef MAP_ELITES_FAST_HPP_
#define MAP_ELITES_FAST_HPP_

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

namespace map_elites {
    template <typename Params, typename Fit, typename S = double>
    class MapElites {
    public:
        using centroids_t = Eigen::Matrix<S, Params::num_cells, Params::dim_features, Eigen::RowMajor>;
        using archive_t = Eigen::Matrix<S, Params::num_cells, Params::dim_search_space, Eigen::RowMajor>;
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
        double qd_score() const
        {
            double qd = 0;
            for (int i = 0; i < Params::num_cells; ++i)
                if (_archive_fit(i) != -std::numeric_limits<S>::max())
                    qd += _archive_fit(i);
            return qd;
        }
        void step()
        {
            for (int i = 0; i < Params::batch_size * 2; ++i)
                _batch_ranks[i] = _rand_int(_r_gen); // yes, from all the map, including niches not filled yet
            for (int i = 0; i < Params::batch_size * 2; ++i) // if empty, we change the random vector to foster exploration
                if (_archive_fit(_batch_ranks[i]) ==  -std::numeric_limits<S>::max())
                    _archive.row(_batch_ranks[i]) = Eigen::Vector<S, Params::dim_search_space>::Random();
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
                    (_centroids.rowwise() - _batch_features.row(i)).rowwise().squaredNorm().minCoeff(&best_i);
                }
                if (_batch_fitness(i) > _archive_fit(best_i))
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
        using batch_t = Eigen::Matrix<S, Params::batch_size, Params::dim_search_space, Eigen::RowMajor>;
        using batch_fit_t = Eigen::Vector<S, Params::batch_size>;
        using batch_features_t = Eigen::Matrix<S, Params::batch_size, Params::dim_features,  Eigen::RowMajor>;
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

} // namespace map_elites
#endif