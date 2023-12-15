#ifndef META_PARAMS__
#define META_PARAMS__

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
}
#endif