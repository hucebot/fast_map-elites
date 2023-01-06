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

#ifdef USE_BOOST // SOBOL centroids is a bit better, but requires boost
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#endif

#include <Eigen/Core>

namespace map_elites {
    template <typename Params, typename Fit, typename S = double>
    class MapElites {
    public:
        using centroids_t = Eigen::Matrix<S, Params::num_cells, Params::dim_features, Eigen::RowMajor>;
        using archive_t = Eigen::Matrix<S, Params::num_cells, Params::dim_search_space, Eigen::RowMajor>;
        using archive_fit_t = Eigen::Vector<S, Params::num_cells>;

        MapElites()
        {
            if (Params::grid) {
                assert(Params::dim_features == 2);
                for (int i = 0; i < Params::grid_size; ++i)
                    for (int j = 0; j < Params::grid_size; ++j) {
                        _centroids(i * Params::grid_size + j, 0) = double(i) / Params::grid_size;
                        _centroids(i * Params::grid_size + j, 1) = double(j) / Params::grid_size;
                    }
            }
            else {
#ifdef USE_BOOST
                // make the centroids with a sobol sequence (only if boost available)
                boost::random::sobol gen(Params::dim_features);
                boost::random::uniform_01<> rand;
                for (int i = 0; i < _centroids.rows(); ++i)
                    for (int j = 0; j < _centroids.cols(); ++j)
                        _centroids(i, j) = rand(gen);
#endif // otherwise we already have random centroids
            }
            _filled_ids.reserve(Params::num_cells);
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
            if (_infill) {
                _batch = _rand<batch_t>();
            }
            else { // normal loop
                // selection
                for (int i = 0; i < Params::batch_size * 2; ++i) // fill with random id of of filled cells
                    _batch_ids[i] = _filled_ids[_rand_id(_r_gen, uint_dist_param_t(0, _filled_ids.size()-1))];

                // variation
                for (int i = 0; i < Params::batch_size; ++i) // line variation
                    _batch.row(i) = _archive.row(_batch_ids[i * 2]) + Params::sigma_1 * _gaussian(_r_gen) * (_archive.row(_batch_ids[i * 2]) - _archive.row(_batch_ids[i * 2 + 1]));
                for (int i = 0; i < Params::batch_size; ++i) // gaussian mutation with bounce back
                    for (int j = 0; j < Params::dim_search_space; ++j) {
                        // bounce does not seem to change much, but it's not worse than a simple cap
                        double r = _gaussian(_r_gen) * Params::sigma_2;
                        _batch(i, j) = _batch(i, j) + r;
                        _batch(i, j) += _batch(i, j) > 1 ? (1 - _batch(i, j)) * 2 : 0;
                        _batch(i, j) += _batch(i, j) < 0 ? -_batch(i, j) * 2 : 0;
                        assert(_batch(i, j) >= 0 && "Params::sigma_2 too large!");
                        assert(_batch(i, j) <= 1 && "Params::sigma_2 too large!");
                    }
            }

            _loop(0, Params::batch_size, [&](int i) { // evaluate the batch
                _batch_features.row(i) = _fit_functions[i].eval(_batch.row(i), _batch_fitness(i));
            });
            // competition
            std::fill(_new_id.begin(), _new_id.end(), -1);
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
                    _new_id[i] = best_i;
            });

            // apply the new ids
            for (int i = 0; i < Params::batch_size; ++i) {
                if (_new_id[i] != -1) {
                    _archive.row(_new_id[i]) = _batch.row(i);
                    if (_archive_fit(_new_id[i]) == -std::numeric_limits<S>::max())
                        _filled_ids.push_back(_new_id[i]);
                    _archive_fit(_new_id[i]) = _batch_fitness(i);
                }
            }

            // we stop the infill when we have enough cells filled
            _infill = (_filled_ids.size() < Params::infill_pct * _archive.rows());
            assert(_filled_ids.size() <= Params::num_cells);
        }

    protected:
        // to make it easy to switch between parallel and non-parallel
        template <typename F>
        inline void
        _loop(size_t begin, size_t end, const F& f)
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
        // random in [0,1] (Eigen is in [-1,1])
        template <typename M>
        inline M _rand() const
        {
            return 0.5 * (M::Random().array() + 1.);
        }


        // our main data
        centroids_t _centroids = _rand<centroids_t>();
        archive_t _archive = _rand<archive_t>();
        archive_fit_t _archive_fit = archive_fit_t::Constant(-std::numeric_limits<S>::max());

        // internal list of filled cells
        std::vector<int> _filled_ids;

        // true when we are still at the infill stage
        bool _infill = true;

        // batch
        using batch_t = Eigen::Matrix<S, Params::batch_size, Params::dim_search_space, Eigen::RowMajor>;
        using batch_fit_t = Eigen::Vector<S, Params::batch_size>;
        using batch_features_t = Eigen::Matrix<S, Params::batch_size, Params::dim_features, Eigen::RowMajor>;
        using fit_functions_t = std::array<Fit, Params::batch_size>;
        using batch_ids_t = Eigen::Vector<int, Params::batch_size * 2>;
        using new_id_t = Eigen::Vector<int, Params::batch_size>;

        batch_t _batch;
        batch_fit_t _batch_fitness;
        batch_features_t _batch_features;
        batch_ids_t _batch_ids;
        fit_functions_t _fit_functions;
        new_id_t _new_id;

        // random
        using uint_dist_param_t = std::uniform_int_distribution<>::param_type;
        std::random_device _rand_device;
        std::default_random_engine _r_gen{_rand_device()};
        std::uniform_int_distribution<int> _rand_id;
        std::normal_distribution<> _gaussian{0, 1};
    };

} // namespace map_elites
#endif