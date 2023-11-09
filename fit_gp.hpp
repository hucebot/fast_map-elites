#ifndef FIT_GP__
#define FIT_GP__

#include "map_elites.hpp"
#include "ugp.hpp"

namespace meta_mapelites {
    template <typename Params, typename S = double>
    struct FitGP {
        using features_t = Eigen::Matrix<S, 1, Params::dim_features, Eigen::RowMajor>;
        using indiv_t = Eigen::Matrix<S, 1, Params::dim_search_space, Eigen::RowMajor>;
        using gp_t = GP<Params::gp_num_points, Params::dim_search_space>;
        using gps_t = std::array<gp_t, Params::dim_features + 1>;
        using data_gps_t = std::array<S, gp_t::data_size>;
        static constexpr int meta_indiv_size = (Params::dim_features + 1) * gp_t::data_size;
        using meta_indiv_t = Eigen::Matrix<S, 1, meta_indiv_size, Eigen::RowMajor>;
        gps_t _gps;
        features_t _features;

        void set(const meta_indiv_t& v)
        {
            for (int i = 0; i < Params::dim_features + 1; ++i)
                _gps[i].set(v.template block<1, gp_t::data_size>(0, i * gp_t::data_size));
        }

        const features_t& eval(const indiv_t& v, S& fit)
        {
            for (int i = 0; i < Params::dim_features; ++i)
                _features(i) = _gps[i].query(v);
            fit = _gps[Params::dim_features].query(v);
            _features = _features.cwiseMin(1).cwiseMax(0);
            assert(!std::isnan(_features.sum()));
            assert(!std::isinf(_features.sum()));
            assert(_features.minCoeff() >= 0);
            assert(_features.maxCoeff() <= 1.0);
            return _features;
        }
    };

    template <typename Params, typename MetaParams, typename S = double>
    struct FitMapElites {
        using indiv_t = Eigen::Matrix<S, 1, MetaParams::dim_search_space, Eigen::RowMajor>;
        using features_t = Eigen::Matrix<S, 1, MetaParams::dim_features, Eigen::RowMajor>;
        using features_time_t = Eigen::Matrix<S, MetaParams::nb_iterations, MetaParams::dim_features, Eigen::RowMajor>;
        features_t _features;
        features_t _max_features;
        features_time_t _features_time;

        using fit_t = FitGP<Params>;
        using map_elites_t = map_elites::MapElites<Params, fit_t>;

        map_elites_t map_elites;
        fit_t fit_function;

        FitMapElites() {}
        FitMapElites(const FitMapElites&) {}
        FitMapElites& operator=(const FitMapElites&) { return *this; }

        const features_t& eval(const indiv_t& v, S& fit)
        {
            map_elites.reset();
            fit_function.set(v);
            map_elites.set_fit_function(fit_function);

            for (int i = 0; i < _features_time.rows(); ++i) {
                map_elites.step();
                _features_time(i, 0) = map_elites.archive_fit().maxCoeff();
                _features_time(i, 1) = map_elites.coverage();
            }
            _max_features = _features_time.row(_features_time.rows() - 1);

            double mean = 0;
            for (int i = 0; i < map_elites.filled_ids().size(); ++i)
                mean += map_elites.archive_fit()[map_elites.filled_ids()[i]];
            mean /= map_elites.filled_ids().size();
            double std = 0; // TODO ignore if fit low
            for (int i = 0; i < map_elites.filled_ids().size(); ++i) {
                double d = (map_elites.archive_fit()[map_elites.filled_ids()[i]] - mean);
                std += d * d;
            }
            if (map_elites.filled_ids().size() > 0)
                std = sqrt(std / map_elites.filled_ids().size());
            else
                std = 0;
            fit = std * map_elites.coverage(); // coverage * std_deviation
            // time to reach 95% of best value

            // TODO use log
            for (int j = 0; j < _features.cols(); ++j)
                for (int i = 0; _features_time(i, j) < 0.99 * _max_features(j) && i + 1 < _features_time.rows(); ++i)
                    _features[j] = i;
            _features = _features / _features_time.rows();

            // std::cout<<"fit:"<<fit<<" features:"<<_features<<std::endl;
            assert(_features.minCoeff() >= 0);
            assert(_features.maxCoeff() <= 1.0);
            return _features;
        }
    };

    void fit(const Eigen::MatrixXd& batch,
    Eigen::MatrixXd& batch_fit,
    Eigen::MatrixXd& batch_features);
} // namespace meta_mapelites
#endif