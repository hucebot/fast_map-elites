#ifndef FIT_META__
#define FIT_META__

#include "fit_gp.hpp"
#include "map_elites.hpp"

namespace meta_mapelites {
    template <typename Params, typename ParamsRandom, typename MetaParams, typename S = double>
    struct FitMetaMapElites {
        using indiv_t = Eigen::Matrix<S, 1, MetaParams::dim_search_space, Eigen::RowMajor>;
        using features_t = Eigen::Matrix<S, 1, MetaParams::dim_features, Eigen::RowMajor>;
        using features_time_t = Eigen::Matrix<S, MetaParams::nb_iterations, MetaParams::dim_features, Eigen::RowMajor>;

        features_t _features;
        features_t _max_features;
        features_time_t _features_time;

#ifdef FUNCTION_COMPOSITION
        using fit_t = FitFunction<Params>;
#else
        using fit_t = FitGP<Params>;
#endif
        using map_elites_t = map_elites::MapElites<Params, fit_t>;
        using random_elites_t = map_elites::MapElites<ParamsRandom, fit_t>;
        using archive_fit_t = typename map_elites_t::archive_fit_t;
        fit_t fit_function;
        archive_fit_t final_me_archive;
        archive_fit_t final_re_archive;
        // typename fit_t::indiv_t center = fit_t::indiv_t::Ones() * 0.5;
        indiv_t center = (indiv_t::Ones() * 0.5).normalized();

        FitMetaMapElites() {}
        FitMetaMapElites(const FitMetaMapElites&) {}
        FitMetaMapElites& operator=(const FitMetaMapElites&) { return *this; }

        const features_t& eval(const indiv_t& v, S& fit)
        {
            // random
            random_elites_t random_elites;
            fit_function.set(v);
            random_elites.set_fit_function(fit_function);
            random_elites.reset();
            for (int i = 0; i < _features_time.rows(); ++i) {
                random_elites.step();
            }
            final_re_archive = random_elites.archive_fit();

            // map-elites
            static constexpr int n_me = 3;
            double me_mean = std::numeric_limits<double>::max(),
                   me_coverage = std::numeric_limits<double>::max(),
                   me_qd_score = std::numeric_limits<double>::max();

            for (int i = 0; i < n_me; ++i) {
                map_elites_t map_elites;
                map_elites.set_fit_function(fit_function);
                for (int i = 0; i < _features_time.rows(); ++i) {
                    map_elites.step();
                }
                me_mean = std::min(me_mean, map_elites.mean());
                me_coverage = std::min(me_coverage, map_elites.coverage());
                me_qd_score = std::min(me_qd_score, map_elites.qd_score());
                // only the last one will be kept
                final_me_archive = map_elites.archive_fit();
            }

            _features[0] = (double)random_elites.coverage() / me_coverage;
            _features[1] = (double)random_elites.mean() / me_mean;
            // std::cout << random_elites.qd_score() << " " << me_qd_score << " " << me_mean << std::endl;
            fit = (me_qd_score - random_elites.qd_score()); // TODO: normalize the QD score?

            if (std::isnan(_features[0]) || std::isnan(_features[1])
                || std::isinf(_features[0])
                || std::isinf(_features[1])) {
                _features[0] = 0.0;
                _features[1] = 0.0;
            }

            _features = (_features).cwiseMin(1.0).cwiseMax(0.0);
            // // just for display / loging / debugging
            // std::cout<<"fit:"<<fit<<" features:"<<_features<<" " << _features_time.rows()<< " "<<std::isnan(_features[0]) << std::endl;
            assert(!std::isnan(_features[0]));
            assert(!std::isnan(_features[1]));
            assert(_features.minCoeff() >= 0);
            assert(_features.maxCoeff() <= 1.0);
            return _features;
        }
    };
} // namespace meta_map_elites
#endif