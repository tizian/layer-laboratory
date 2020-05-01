#pragma once

#include <mitsuba/core/fwd.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <complex>

using namespace std::complex_literals;

NAMESPACE_BEGIN(mitsuba)

class Layer;

typedef Eigen::Matrix<double,  Eigen::Dynamic, 1> VectorX;
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> VectorXc;

typedef Eigen::SparseMatrix<double, Eigen::ColMajor, long> MatrixS;

typedef Eigen::Matrix<double,  Eigen::Dynamic, Eigen::Dynamic> MatrixX;
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> MatrixXc;

using Vector3f = Vector<double, 3>;
using Float32P = Packet<float>;
using Int32P = Packet<int>;
using Color3f  = Color<float, 3>;
using Frame3f  = Frame<double>;

/// Pad to vectorizable size
template <size_t width> size_t pad(size_t n) {
    return (n + width - 1) / width;
}

/// Cosine/sine addition theorem
template <typename T, typename S>
inline auto trig_addition(T sc, S inc) {
    return std::make_pair(
        sc.second * inc.first  + sc.first * inc.second,
        sc.second * inc.second - sc.first * inc.first
    );
}

/// Cosine/sine subtraction theorem
template <typename T, typename S>
inline auto trig_subtraction(T sc, S inc) {
    return std::make_pair(
        sc.first * inc.second  - inc.first * sc.second,
        sc.second * inc.second + sc.first * inc.first
    );
}

NAMESPACE_END(mitsuba)
