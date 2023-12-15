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
} // namespace meta_mapelites
#endif