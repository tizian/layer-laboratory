#pragma once

#include <mitsuba/layer/fwd.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/math.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Evaluate a 1D Cosine series.
 *
 * \param coeffs
 *    Coefficient storage
 *
 * \param m
 *    Denotes the size of \c coeffs in phi dimension
 *
 * \param phi
 *    Value for which the series should be evaluated
 *
 * \return
 *     The value of the series at this value
 */
extern MTS_EXPORT_LAYER float eval_cosine_series_1d(const float *coeffs, size_t m, float phi);

/**
 * \brief Evaluate a 2D Fourier series.
 *
 * \param coeffs
 *    Coefficient storage
 *
 * \param ms
 *    Denotes the size of \c coeffs in phi_s dimension
 *
 * \param md
 *    Denotes the size of \c coeffs in phi_d dimension
 *
 * \param phi_s
 *    Azimuthal sum angle for which the series should be evaluated
 *
 * \param phi_d
 *    Azimuthal difference angle for which the series should be evaluated
 *
 * \return
 *     The value of the series for these angles
 */
extern MTS_EXPORT_LAYER float eval_fourier_series_2d(const float *coeffs, size_t ms, size_t md, float phi_s, float phi_d);

/**
 * \brief Simultaneously evaluate <em>three</em> 2D Fourier series
 * corresponding to the color channels (Y, R, B) and return
 * a spectral power distribution
 *
 * \param coeffs
 *    Coefficient storage
 *
 * \param ms
 *    Denotes the size of \c coeffs in phi_s dimension
 *
 * \param md
 *    Denotes the size of \c coeffs in phi_d dimension
 *
 * \param phi_s
 *    Azimuthal sum angle for which the series should be evaluated
 *
 * \param phi_d
 *    Azimuthal difference angle for which the series should be evaluated
 *
 * \return
 *     The value of the series for these angles
 */
extern Color3f eval_3_fourier_series_2d(float * const coeffs[3], size_t ms, size_t md, float phi_s, float phi_d);

/**
 * \brief Compute a Cosine series of the given sampled function by integrating
 * it against the basis functions using Filon quadrature
 *
 * This version computes a Cosine series of a real function, thus results in a
 * set of real Fourier coefficients for (l=0, ..., orders-1)
 *
 * Filon quadrature works by constructing a piecewise quadratic interpolant of
 * the original function. The Fourier basis functions are then integrated
 * against this representation, which has an analytic solution. This avoids all
 * of the problems of traditional quadrature schemes involving highly
 * oscillatory integrals. It is thus possible to obtain accurate coefficients
 * even for high orders.
 *
 * \param values
 *    Function to be integrated
 *
 * \param size
 *    Desired resolution of the piecewise quadratic interpolant
 *
 * \param[out] coeffs
 *    Output buffer used to store the computed coefficients. The function adds
 *    the computed coefficients to the buffer rather than overwriting the
 *    existing contents.
 *
 * \param orders
 *    Desired number of coefficients
 *
 * \param a
 *    Start of the integration, can be set to values other than zero. Note that
 *    the Cosine series basis functions are not orthogonal anymore in this case.
 *
 * \param b
 *    End of the integration, can be set to values other than pi. Note that the
 *    Cosine series basis functions are not orthogonal anymore in this case.
 *
 * \remark
 *    In the Python API, the \c coeffs array is directly returned.
 */
extern void filon_integrate_cosine(const std::function<double(double)> &f, size_t size,
                                   double *coeffs, size_t orders,
                                   double a = 0, double b = math::Pi<double>);

/**
 * \brief Compute a Fourier series of the given sampled function by integrating
 * it against the basis functions using Filon quadrature
 *
 * This version computes a Fourier series of a complex function. It results in
 * a set of complex Fourier coefficients for (l=-orders/2, ..., orders/2)
 *
 * Filon quadrature works by constructing a piecewise cubic spline interpolant
 * of the original function. The Fourier basis functions are then integrated
 * against this representation, which has an analytic solution. This avoids all
 * of the problems of traditional quadrature schemes involving highly
 * oscillatory intetrals. It is thus possible to obtain accurate coefficients
 * even for high orders.
 *
 * \param values
 *    Uniformly sampled values of the function to be integrated between
 *    \c a and \c b
 *
 * \param size
 *    Number of function samples
 *
 * \param[out] coeffs
 *    Output buffer used to store the computed coefficients. The function adds
 *    the computed coefficients to the buffer rather than overwriting the
 *    existing contents.
 *
 * \param orders
 *    Desired number of coefficients. Assumed to be odd.
 *
 * \param a
 *    Start of the integration, can be set to values other than zero. Note that
 *    the Cosine series basis functions are not orthogonal anymore in this case.
 *
 * \param b
 *    End of the integration, can be set to values other than pi. Note that the
 *    Cosine series basis functions are not orthogonal anymore in this case.
 *
 * \remark
 *    In the Python API, the \c coeffs array is directly returned.
 */
extern MTS_EXPORT_LAYER void filon_integrate_exp(const std::complex<double> *values, size_t size,
                                                 std::complex<double> *coeffs, size_t orders,
                                                 double a = 0, double b = 2*math::Pi<double>);

/**
 * \brief Computes the Fourier series of a product of Fourier series
 * using discrete convolution.
 *
 * \param a
 *    First input vector of (real) Fourier coefficients
 *
 * \param b
 *    Second input vector of (complex) Fourier coefficients
 *
 * \param[out] result
 *    Output vector of convolved (complex) Fourier coefficients
 *
 * \remark
 *    In the Python API, the \c result vector is directly returned).
 */
extern void convolve_fourier_series(const VectorX &a, const VectorXc &b, VectorXc &result);

/**
 * \brief Computes the 2D Fourier series of a product of a "diagonal" Fourier series
 * with another 2D series using discrete convolution.
 *
 * \param a
 *    First input vector of Fourier coefficients (diagonal of a 2D matrix)
 *
 * \param B
 *    Input matrix of 2D Fourier coefficients
 *
 * \param[out] result
 *    The resulting output matrix of convolved 2D Fourier coefficients
 */
extern void convolve_fourier_series_diagonal(const VectorX &a, const MatrixX &B, MatrixX &result);

/**
 * \brief Computes the 2D Fourier series of a product of an "anti-diagonal" Fourier series
 * with another 2D series using discrete convolution.
 *
 * \param a
 *    First input vector of Fourier coefficients (anto-diagonal of a 2D matrix)
 *
 * \param B
 *    Input matrix of 2D Fourier coefficients
 *
 * \param[out] result
 *    The resulting output matrix of convolved 2D Fourier coefficients
 */
extern void convolve_fourier_series_antidiagonal(const VectorX &a, const MatrixX &B, MatrixX &result);


/**
 * \brief Computes the complex to complex Fourier series via FFTW.
 *
 * \param values
 *    Input array of complex function samples in [0, 2pi)
 *
 * \param size
 *    Size of input array. Required to be an odd number.
 *
 * \param[out] coeffs
 *    Output array of complex Fourier series coefficients (same length as \c values)
 *
 * \remark
 *    In the Python API, the \c size parameter is inferred from the \c values array
 *    and the \c coeffs vector is directly returned.
 */
extern MTS_EXPORT_LAYER void fftw_transform_c2c(const std::complex<double> *values, size_t size,
                                                std::complex<double> *coeffs);

/**
 * \brief Computes the real to complex Fourier series via FFTW.
 * The resulting series will have the "Hermitian" symmetry.
 *
 * \param values
 *    Input array of real function samples in [0, 2pi)
 *
 * \param size
 *    Size of input array. Required to be an odd number.
 *
 * \param[out] coeffs
 *    Output array of complex Fourier series coefficients (same length as \c values)
 *
 * \remark
 *    In the Python API, the \c size parameter is inferred from the \c values array
 *    and the \c coeffs vector is directly returned.
 */
extern MTS_EXPORT_LAYER void fftw_transform_r2c(const double *values, size_t size,
                                                std::complex<double> *coeffs);

/**
 * \brief Computes the Fourier series for a 2D matrix of function sample values
 * via FFTW.
 * The resulting series is assumed to only have real Fourier coefficients.
 * This is useful to e.g. project measured BSDF slices in phi_s, phi_d into the
 * Fourier representation.
 *
 * \param values
 *    Input matrix of real function samples in [0, 2pi) x [0, 2pi).
 *    Required to have odd dimensions.
 *
 * \return
 *    Matrix of real 2D Fourier series coefficients with same size as the input.
 */
extern MTS_EXPORT_LAYER MatrixX fftw_transform_2d(const MatrixX &values);

NAMESPACE_END(mitsuba)
