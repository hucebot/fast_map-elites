#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "fit_gp.hpp"

namespace nb = nanobind;
namespace meta_mapelites {
    struct Params {
        static constexpr int gp_num_points = 20;
        static constexpr int dim_features = 2;
        static constexpr int dim_search_space = 5;
        static constexpr bool verbose = false;
        static constexpr bool grid = true;
        static constexpr bool parallel = false;
        static constexpr double min_fit = -1e10;
    };

    // params of the meta fitness
    struct MetaParams {
        static constexpr int dim_features = 2;
        static constexpr int dim_search_space = FitGP<Params>::meta_indiv_size;
        static constexpr int batch_size = 64;
        static constexpr int nb_iterations = 100000 / batch_size;
        static constexpr bool verbose = true;
        static constexpr bool grid = true;
        static constexpr bool parallel = true;
        static constexpr int grid_size = 64;
        static constexpr int num_cells = grid ? grid_size * grid_size : 12000; // 12000; // 8192;
    };

} // namespace meta_mapelites

// this is the fitness function found (based on GPs)
// batch : (batch_size, dim) batch of solutions.
// for some reason this cannot be in the namespace?
void fit_gp(const nb::DRef<Eigen::VectorXd>& fit_params,
    const nb::DRef<Eigen::MatrixXd>& batch,
    nb::DRef<Eigen::VectorXd> batch_fit,
    nb::DRef<Eigen::MatrixXd> batch_features)
{
    using namespace meta_mapelites;

    assert(batch.rows() > 0);
    assert(batch_features.cols() == Params::dim_features);
    assert(batch.cols() == Params::dim_search_space);
    assert(batch_fit.rows() == batch.rows());
    assert(batch_features.rows() == batch.rows());

    //std::cout << "received batch:" << batch << std::endl;

    using fit_t = FitGP<Params>;
    fit_t::indiv_t _indiv;
    fit_t::features_t _features;

    for (int i = 0; i < batch.rows(); ++i) {
        fit_t _fit;
        // copy to indiv
        for (int j = 0; j < batch.cols(); ++j)
            _indiv(j) = batch(i, j);
        // copy to fit
        _fit.set(fit_params);
        // eval
        // std::cout<<i<<" " << _indiv << std::endl;
        _features = _fit.eval(_indiv, batch_fit(i, 0));
       // std::cout << "[" << i << "]" << batch_fit(i, 0) << " ";
        std::cout.flush();
        // copy the features
        for (int j = 0; j < _features.size(); ++j)
            batch_features(i, j) = _features(j);
    }//);
}

// // this is the meta-fitness
// void fit(const nb::DRef<Eigen::MatrixXd>& batch,
//     nb::DRef<Eigen::VectorXd> batch_fit,
//     nb::DRef<Eigen::MatrixXd> batch_features)
// {
//     using namespace meta_mapelites;

//     assert(batch.rows() > 0);
//     assert(batch_features.cols() == Params::dim_features);
//     assert(batch.cols() == Params::dim_search_space);
//     assert(batch_fit.rows() == batch.rows());
//     assert(batch_features.rows() == batch.rows());

//     std::cout << "received batch:" << batch.rows() << " x " << batch.cols() << std::endl;

//     // for (int i = 0; i < batch.rows(); ++i) {
//     tbb::parallel_for(size_t(0), size_t(batch.rows()), [&](size_t i) {
//         using fit_t = FitMapElites<Params, MetaParams>;
//         thread_local fit_t::indiv_t _indiv;
//         thread_local fit_t::features_t _features;
//         thread_local fit_t _fit;
//         // copy to indiv
//         for (int j = 0; j < batch.cols(); ++j)
//             _indiv(j) = batch(i, j);
//         // eval
//         // std::cout<<i<<" " << _indiv << std::endl;
//         _features = _fit.eval(_indiv, batch_fit(i, 0));
//         std::cout << "[" << i << "]" << batch_fit(i, 0) << " ";
//         std::cout.flush();
//         // copy the features
//         for (int j = 0; j < _features.size(); ++j)
//             batch_features(i, j) = _features(j);
//     });
// }

NB_MODULE(pf_mapelites, m)
{
    m.def("fit_gp", &fit_gp);
}