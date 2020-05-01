#pragma once

#include <mitsuba/layer/fwd.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Evaluate the HG model using the mu_i, mu_o, phi_d parameterization
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param phi_d
 *    Azimuthal difference angle
 *
 * \param g
 *    "Anisotropy" parameter
 */
extern MTS_EXPORT_LAYER double henyey_greenstein(double mu_o, double mu_i, double phi_d, double g);

/**
 * \brief Compute a Fourier series of the HG model
 *
 * This function first finds the 0-th and 1st-order Fourier coefficients using
 * elliptic integrals.
 *
 * The other coefficients can then be determined much more efficiently; the
 * approach here is based on the idea that the ratio of adjacent coefficients
 * eventually converges to a constant value. Using a 2nd-order Taylor
 * expansion, we can obtain a fairly accurate estimate of this ratio somewhere
 * "in the middle" (i.e. for large $n$, but well before the aforementioned
 * convergence).
 *
 * Using a backwards recurrence scheme, we can then determine all previous
 * ratios and, thereby (using the explicitly computed first values), all
 * Fourier coefficients.
 *
 * This approach is based on the article
 *
 * "A Recurrence Formula For Computing Fourier Components of the
 *  Henyey-Greenstein Phase Function" by E.G. Yanovitskij
 *
 * Journal of Quantitative Spectroscopy & Radiative Transfer, 57, no 1. 1977
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param g
 *    "Anisotropy" parameter
 *
 * \param md
 *    Indicates a desired maximum number of Fourier coefficients. The
 *    implementation will blur out higher Frequency content to try to
 *    achieve this number.
 *
 * \param result
 *    Storage for the generated Fourier coefficients
 */
extern MTS_EXPORT_LAYER void henyey_greenstein_fourier_series(double mu_o, double mu_i, double g, int md,
                                                              double relerr, VectorX &result);

/**
 * \brief Evaluate the von Mises-Fisher model using the mu_i, mu_o, phi_d parameterization
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param phi_d
 *    Azimuthal difference angle
 *
 * \param kappa
 *    "Anisotropy" parameter
 */
extern MTS_EXPORT_LAYER double von_mises_fisher(double mu_o, double mu_i, double phi_d, double kappa);

/**
 * \brief Compute a Fourier series of the von Mises-Fisher model
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param kappa
 *    "Anisotropy" parameter
 *
 * \param relerr
 *    Relative error goal
 *
 * \param result
 *    Storage for the generated Fourier coefficients
 */
extern MTS_EXPORT_LAYER void von_mises_fisher_fourier_series(double mu_o, double mu_i, double kappa,
                                                             double relerr, VectorX &result);

NAMESPACE_END(mitsuba)
