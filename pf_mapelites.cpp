// This the python binding with nanobind

#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "fit_gp.hpp"
#include "fit_meta.hpp"
#include "meta_params.hpp"

namespace nb = nanobind;

// this is the wrapper around the fitness function found (based on GPs)
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

    // std::cout << "received batch:" << batch << std::endl;

    using fit_t = FitGP<Params>;
    fit_t::indiv_t _indiv;
    fit_t::features_t _features;

    // no parallel loop here because we parallelize the whole process at the higher level
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
    } //);
}

// this is the meta-fitness
void fit_meta(const nb::DRef<Eigen::MatrixXd>& batch,
    nb::DRef<Eigen::VectorXd> batch_fit,
    nb::DRef<Eigen::MatrixXd> batch_features)
{
    using namespace meta_mapelites;

    assert(batch.rows() > 0);
    assert(batch_features.cols() == Params::dim_features);
    assert(batch.cols() == Params::dim_search_space);
    assert(batch_fit.rows() == batch.rows());
    assert(batch_features.rows() == batch.rows());


    // for (int i = 0; i < batch.rows(); ++i) {
    tbb::parallel_for(size_t(0), size_t(batch.rows()), [&](size_t i) {
        using fit_t = FitMetaMapElites<Params, ParamsRandom, MetaParams>;
        thread_local fit_t::indiv_t _indiv; // this can be thread_local because we simply copy data
        thread_local fit_t::features_t _features;
        fit_t _fit; // reinitialize a copy each time ; this is safer
        // copy to indiv
        for (int j = 0; j < batch.cols(); ++j)
            _indiv(j) = batch(i, j);
        // eval
        // std::cout<<i<<" " << _indiv << std::endl;
        _features = _fit.eval(_indiv, batch_fit(i, 0));
        std::cout << "[" << i << "]" << batch_fit(i, 0) << " ";
        std::cout.flush();
        // copy the features
        for (int j = 0; j < _features.size(); ++j)
            batch_features(i, j) = _features(j);
    });
}

NB_MODULE(pf_mapelites, m)
{
    m.def("fit_gp", &fit_gp)
    .def("fit_meta", &fit_meta);
}
