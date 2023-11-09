#ifndef UGP_HPP__
#define  UGP_HPP__
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

// A fast Gaussian process without the variance, for fixed-sized data
template <int NumPoints, int SearchSpace, typename S = double>
struct GP {
    //    static constexpr double length_scale = 0.05; // could be a parameter
    static constexpr int data_size = NumPoints * (SearchSpace + 1) + 1 + 1;
    // a flat vector with all the point, then all the values (a row vector from MAP-Elite)
    using data_t = Eigen::Matrix<S, 1, data_size, Eigen::RowMajor>;

    // a single point (a row), for queries
    using point_t = Eigen::Matrix<S, 1, SearchSpace, Eigen::RowMajor>;
    // a list of points, for our main data
    using points_t = Eigen::Matrix<S, NumPoints, SearchSpace, Eigen::RowMajor>;
    // each point is associated to a value
    using values_t = Eigen::Vector<S, NumPoints>;

    // internally used by the GP
    using K_t = Eigen::Matrix<S, NumPoints, NumPoints, Eigen::RowMajor>;
    using k_t = Eigen::RowVector<S, NumPoints>;
    points_t _points;
    values_t _values;
    double _length_scale;
    double _period;

    K_t _K;
    Eigen::Vector<S, NumPoints> _alpha;
    K_t _L;
    k_t _k;

    GP() {}

    void set(const data_t& data)
    {
        // copy the data to the points / value
        int k = 0;
        for (size_t i = 0; i < NumPoints; ++i)
            for (size_t j = 0; j < SearchSpace; ++j)
                _points(i, j) = data[k++];
        for (size_t i = 0; i < NumPoints; ++i)
            _values(i) = data[k++];
        _length_scale = data[k++] * 0.99 + 0.01;
        _period = data[k++] * 0.99 + 0.01;

        // compute the kernel (Gram matrix)
        _compute_K(_points, _points, _K);

        // precompute the expensive stuffs -- O(n^3)
        _L = Eigen::LLT<K_t>(_K).matrixL();
        _alpha = _L.template triangularView<Eigen::Lower>().solve(_values);
        _L.template triangularView<Eigen::Lower>().adjoint().solveInPlace(_alpha);
    }

    template <typename T1, typename T2>
    double _kernel_exp(const T1& p1, const T2& p2) const
    {
        return std::exp(-(p1 - p2).squaredNorm() / (2 * _length_scale * _length_scale));
    }

    template <typename T1, typename T2>
    double _kernel_periodic(const T1& p1, const T2& p2)
    {
        return exp(-2. / (_length_scale * _length_scale) * ((p1 - p2).array().abs() * M_PI / _period).array().sin().square().sum());
    }

    // compute the gram matrix, we use a basic kernel
    template <typename M1, typename M2, typename Res>
    void _compute_K(const M1& m1, const M2& m2, Res& kernel)
    {
        // we only need to compute once because the norm is symetric
        for (size_t i = 0; i < NumPoints; i++)
            for (size_t j = 0; j <= i; ++j)
                kernel(i, j) = _kernel_periodic(m1.row(i), m2.row(j));
          
        for (size_t i = 0; i < NumPoints; i++)
            for (size_t j = 0; j < i; ++j)
                kernel(j, i) = kernel(i, j);
    }

    double query(const point_t& v)
    {
        for (int i = 0; i < NumPoints; i++)
            _k(i) = _kernel_periodic(_points.row(i), v);
        return _k * _alpha; // _k is a ROW vector here
    }
};

#endif