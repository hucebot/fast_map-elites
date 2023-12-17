#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <iterator>
#include <memory>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "fit_meta.hpp"

namespace meta_mapelites {
   
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
} // namespace meta_mapelites


int main(int argc, char** argv)
{
    using namespace meta_mapelites;
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
    using fit_t = meta_mapelites::FitMetaMapElites<Params, ParamsRandom, MetaParams>;
    using map_elites_t = map_elites::MapElites<MetaParams, fit_t>;

    if (argc == 1) { // run the algorithm
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "creating meta map-elites, search space=" << MetaParams::dim_search_space << std::endl;
        auto map_elites = std::make_unique<map_elites_t>(); // put on heap and not on stack! (otherwise segv)

        std::cout << "starting meta map-elites" << std::endl;
        std::ofstream qd_ofs("qd.dat");

        for (size_t i = 0; i < 100000 / Params::batch_size*/; ++i) {
            map_elites->step();
            qd_ofs << i * Params::batch_size << " " << map_elites->qd_score_normalized() << " " << map_elites->coverage() << std::endl;
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
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < archive->rows(); ++i) {
            // tbb::parallel_for(size_t(0), size_t(archive->rows() / 100), size_t(1), [&](size_t i) {

            if ((*archive_fit)[i] > -1e10) // hack, should be == -std::numeric_limits<S>::max()
            {
                std::cout << i << ":";
                std::cout.flush();
                FitMetaMapElites<Params, ParamsRandom, MetaParams> fit;
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
          auto end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Total time:" << t / 1000.0 << "s" << std::endl;
    }
    return 0;
}
