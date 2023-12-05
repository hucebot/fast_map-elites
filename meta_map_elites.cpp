#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <memory>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "map_elites.hpp"
#include "ugp.hpp"

// alternative : INDUCTION POINTS !
template <typename Params, typename S = double>
struct FitFunction { // describe generic composition of functions
    // the vector that specifies the function itself
    static constexpr int num_function_kinds = 5; // sin, cos, tanh, linear, exp
    static constexpr int func_spec_dim = Params::layer_1 // number of functions in layer 1
        + Params::layer_2 // number of functions in layer 2
        + (1 + Params::dim_features) // functions of the output layer
        + Params::dim_search_space * Params::layer_1 // weights from input to layer 1
        + Params::layer_1 * Params::layer_2 // weights from layer 1 to layer 2
        + Params::layer_2 * (1 + Params::dim_features); // weights from layer_2 to outputs
    using func_spec_t = Eigen::Vector<S, func_spec_dim>;
    Eigen::Matrix<S, Params::dim_search_space, Params::layer_1, Eigen::RowMajor> _weights_in_to_1;
    Eigen::Matrix<S, Params::layer_1, Params::layer_2, Eigen::RowMajor> _weights_1_to_2;
    Eigen::Matrix<S, Params::layer_2, (1 + Params::dim_features), Eigen::RowMajor> _weights_2_to_out;
    Eigen::Matrix<int, 1, Params::layer_1, Eigen::RowMajor> _layer_1_funcs;
    Eigen::Matrix<S, 1, Params::layer_1, Eigen::RowMajor> _layer_1_res;
    Eigen::Matrix<int, 1, Params::layer_2, Eigen::RowMajor> _layer_2_funcs;
    Eigen::Matrix<S, 1, Params::layer_2, Eigen::RowMajor> _layer_2_res;
    Eigen::Matrix<int, 1, 1 + Params::dim_features, Eigen::RowMajor> _out_funcs;
    Eigen::Matrix<S, 1, 1 + Params::dim_features, Eigen::RowMajor> _out_res;

    // the solutions that are evaluated by this function
    using indiv_t = Eigen::Matrix<S, 1, Params::dim_search_space, Eigen::RowMajor>;
    using features_t = Eigen::Matrix<S, 1, Params::dim_features, Eigen::RowMajor>;

    // keep the intermediate values to avoid reallocations
    features_t _features;
    func_spec_t _spec;

    FitFunction() {} // needed for now
    FitFunction(const FitFunction& f)
    {
        set(f._spec);
    }

    // make a specific function: develop the genotype
    void set(const func_spec_t& func_spec)
    {
        _spec = func_spec;
        int k = 0;

        for (size_t i = 0; i < Params::layer_1; ++i)
            _layer_1_funcs[i] = int(func_spec[k++] * num_function_kinds);

        for (size_t i = 0; i < Params::layer_2; ++i)
            _layer_2_funcs[i] = int(func_spec[k++] * num_function_kinds);

        for (size_t i = 0; i < Params::dim_features + 1; ++i)
            _out_funcs[i] = int(func_spec[k++] * num_function_kinds);

        for (size_t i = 0; i < Params::dim_search_space; ++i)
            for (size_t j = 0; j < Params::layer_1; ++j)
                _weights_in_to_1(i, j) = func_spec[k++] * 2 - 1.0;

        for (size_t i = 0; i < Params::layer_1; ++i)
            for (size_t j = 0; j < Params::layer_2; ++j)
                _weights_1_to_2(i, j) = func_spec[k++] * 2 - 1.0;

        for (size_t i = 0; i < Params::layer_2; ++i)
            for (size_t j = 0; j < (1 + Params::dim_features); ++j)
                _weights_2_to_out(i, j) = func_spec[k++] * 2 - 1.0;
    }

    const features_t& eval(const indiv_t& v, S& fit)
    {
        _layer_1_res = (v * _weights_in_to_1);
        _inplace_func(_layer_1_funcs, _layer_1_res);
        _layer_2_res = _layer_1_res * _weights_1_to_2;
        _inplace_func(_layer_2_funcs, _layer_2_res);
        _out_res = (_layer_2_res * _weights_2_to_out);
        _inplace_func(_out_funcs, _out_res);

        fit = 0.5 * tanh(_out_res[0] + 1); //[0,1]

        // guarantee features in [0,1]
        for (int i = 0; i < Params::dim_features; ++i)
            _features[i] = 0.5 * (tanh(_out_res[1 + i]) + 1);
        _features = _features.cwiseMin(1).cwiseMax(0); // to be sure...
        assert(_features.minCoeff() >= 0);
        assert(_features.maxCoeff() <= 1.0);
        return _features;
    }
    template <typename T1, typename T2>
    void _inplace_func(const T1& funcs, T2& vec)
    {
        assert(funcs.cols() == vec.cols());
        for (int i = 0; i < funcs.cols(); ++i) {
            if (funcs[i] == 0)
                vec[i] = sin(vec[i]);
            else if (funcs[i] == 1)
                vec[i] = cos(vec[i]);
            else if (funcs[i] == 2)
                vec[i] = exp(vec[i]);
            else if (funcs[i] == 3)
                vec[i] = vec[i]; // nothing
            else if (funcs[i] == 4)
                vec[i] = tanh(vec[i]); // nothing
            else {
                std::cerr << "Unknown function ID:" << funcs[i] << " for i=" << i << std::endl;
                assert(0);
            }
        }
    }
};
bool print = false;

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
std::ofstream ofs_temp("fit_track.dat");

template <typename Params, typename ParamsRandom, typename MetaParams, typename S = double>
struct FitMapElites {
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

    FitMapElites() {}
    FitMapElites(const FitMapElites&) {}
    FitMapElites& operator=(const FitMapElites&) { return *this; }

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
        //std::cout << random_elites.qd_score() << " " << me_qd_score << " " << me_mean << std::endl;
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

struct Params {
    static constexpr int layer_1 = 3;
    static constexpr int layer_2 = 4;
    static constexpr int gp_num_points = 20;

    static constexpr int dim_features = 2;
    static constexpr int dim_search_space = 5;
    static constexpr int batch_size = 64;
    static constexpr double sigma_1 = 0.15;
    static constexpr double sigma_2 = 0.01;
    static constexpr double infill_pct = 0.05;
    static constexpr bool verbose = false;
    static constexpr bool grid = true;
    static constexpr bool parallel = false;
    static constexpr int grid_size = 64;
    static constexpr int num_cells = grid ? grid_size * grid_size : 12000; // 12000; // 8192;
    static constexpr double min_fit = -1e10;
};

struct ParamsRandom {
    static constexpr int layer_1 = 3;
    static constexpr int layer_2 = 4;
    static constexpr int gp_num_points = 20;

    static constexpr int dim_features = 2;
    static constexpr int dim_search_space = 5;
    static constexpr int batch_size = 64;
    static constexpr double sigma_1 = 0.15;
    static constexpr double sigma_2 = 0.05;
    static constexpr double infill_pct = -1; // we always infill, no evolution
    static constexpr bool verbose = false;
    static constexpr bool grid = true;
    static constexpr bool parallel = false;
    static constexpr int grid_size = 64;
    static constexpr int num_cells = grid ? grid_size * grid_size : 12000; // 12000; // 8192;
    static constexpr double min_fit = -1e10;
};

struct MetaParams {
    static constexpr int dim_features = 2;
#ifdef FUNCTION_COMPOSITION
    static constexpr int dim_search_space = FitFunction<Params>::func_spec_dim;
#else
    static constexpr int dim_search_space = FitGP<Params>::meta_indiv_size;
#endif
    static constexpr int batch_size = 64;
    static constexpr int nb_iterations = 100000 / batch_size;
    static constexpr double sigma_1 = 0.15;
    static constexpr double sigma_2 = 0.01; // bigger?
    static constexpr bool verbose = true;
    static constexpr bool grid = true;
    static constexpr bool parallel = true;
    static constexpr int grid_size = 64;
    static constexpr int num_cells = grid ? grid_size * grid_size : 12000; // 12000; // 8192;
    static constexpr double min_fit = 1;
    static constexpr double infill_pct = 150. / num_cells; // 0.05;
};

// load an eigen matrix
template <typename M>
void load(const std::string& path, M& m)
{
    std::cout << "loading " << path << "..." << std::endl;
    std::ifstream ifs(path.c_str());
    assert(ifs.good());
    // load all the values (treat \n as space here, so no info on column)
    std::vector<double> data = std::vector<double>{
        std::istream_iterator<double>(ifs),
        std::istream_iterator<double>()};
    // ensure that all the lines are full
    assert(data.size() % m.cols() == 0);
    // copy to the matrix
    for (int i = 0, k = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j)
            m(i, j) = data[k++];
}

int main(int argc, char** argv)
{
    srand((unsigned int)time(0));
    // using gp_t = GP<10, 2>;
    // gp_t gp; // 4 points, 2D
    // gp_t::data_t d = 0.5 * (gp_t::data_t::Random().array() + 1.0);
    // std::ofstream gpf("gp.dat");
    // auto start_gp = std::chrono::high_resolution_clock::now();
    // gp_t::point_t p;
    // gp.set(d);
    // for (float fi = 0; fi < 1.0; fi += 0.01)
    //     for (float fj = 0; fj < 1.0; fj += 0.01) {
    //         p = gp_t::point_t(fi, fj);
    //         gpf << fi << " " << fj << " " << gp.query(p) << std::endl;
    //     }
    // std::ofstream gpp("gp_points.dat");
    // gpp << gp._points<<std::endl;

    // auto end_gp = std::chrono::high_resolution_clock::now();
    // double tt = std::chrono::duration_cast<std::chrono::milliseconds>(end_gp - start_gp).count();
    // std::cout << "GP time:" << tt / 1000.0 << "s" << std::endl;
    using fit_t = FitMapElites<Params, ParamsRandom, MetaParams>;
    using map_elites_t = map_elites::MapElites<MetaParams, fit_t>;

    if (argc == 1) { // run the algorithm
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "creating meta map-elites, search space=" << MetaParams::dim_search_space << std::endl;
        auto map_elites = std::make_unique<map_elites_t>(); // put on heap and not on stack! (otherwise segv)

        std::cout << "starting meta map-elites" << std::endl;
        std::ofstream qd_ofs("qd.dat");

        for (size_t i = 0; i < 550 /*1e6 / Params::batch_size*/; ++i) {
            map_elites->step();
            qd_ofs << i * Params::batch_size << " " << map_elites->qd_score() << " " << map_elites->coverage() << std::endl;
            if (MetaParams::verbose)
                std::cout << map_elites->coverage() << "[" << i << "] ";
            std::cout.flush();
            if (i % 10 == 0) {
                std::cout << "writing result...";
                std::cout.flush();
                std::ofstream c("centroids.dat");
                c << map_elites->centroids() << std::endl;
                std::ofstream f("fit.dat");
                f << map_elites->archive_fit() << std::endl;
                std::ofstream a("archive.dat");
                a << map_elites->archive() << std::endl;
                std::cout << "done" << std::endl;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Total time:" << t / 1000.0 << "s" << std::endl;

        std::cout << "writing result...";
        std::cout.flush();
        std::ofstream c("centroids.dat");
        c << map_elites->centroids() << std::endl;
        std::ofstream f("fit.dat");
        f << map_elites->archive_fit() << std::endl;
        std::ofstream a("archive.dat");
        a << map_elites->archive() << std::endl;
        std::cout << "done" << std::endl;
    }
    else {

        assert(argc == 3);
        std::cout << "Loading " << argv[1] << " and " << argv[2] << std::endl;

        auto archive = std::make_shared<map_elites_t::archive_t>();
        auto archive_fit = std::make_shared<map_elites_t::archive_fit_t>();

        load(argv[1], *archive);
        load(argv[2], *archive_fit);

        std::ofstream all_fit("all_fit.dat");
        std::cout << "writing...";
        for (int i = 0; i < 500; ++i) {
            // tbb::parallel_for(size_t(0), size_t(archive->rows() / 100), size_t(1), [&](size_t i) {

            if ((*archive_fit)[i] > -1e10) // hack, should be == -std::numeric_limits<S>::max()
            {
                std::cout << i << ":";
                std::cout.flush();
                FitMapElites<Params, ParamsRandom, MetaParams> fit;
                double f = 0;
                auto features = fit.eval(archive->row(i), f);
                {
                    std::ofstream ofs("data/me_fit_" + std::to_string(i) + ".dat");
                    ofs << fit.final_me_archive;
                }
                {
                    std::ofstream ofs("data/re_fit_" + std::to_string(i) + ".dat");
                    ofs << fit.final_re_archive;
                }
                std::cout << (*archive_fit)[i] << "=>" << f << " ";
                std::cout.flush();
            }
            // all_fit << i << " " << f << " " << features << " " << (*archive_fit)[i] << std::endl;
        } //); // end parallel for
    }
    return 0;
}