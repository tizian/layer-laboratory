#pragma once

#include <mitsuba/layer/fwd.h>

NAMESPACE_BEGIN(mitsuba)

extern std::atomic<double> microfacet_timer;

/**
 * \brief Evaluate the Beckmann distribution-based microfacet BSDF by
 * Walter et al. using the mu_i, mu_o, phi_s, phi_d parameterization.
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param phi_s
 *    Azimuthal angle sum
 *
 * \param phi_d
 *    Azimuthal angle difference
 *
 * \param alpha_u
 *    Beckmann roughness parameter in U direction
 *
 * \param alpha_v
 *    Beckmann roughness parameter in V direction
 *
 * \param eta
 *    Relative index of refraction (complex)
 *
 * \param isotropic_g
 *    Indicates whether an isotropic version of the shadowing-masking function
 *    should be used. (Default: false)
 */
extern MTS_EXPORT_LAYER double microfacet(double mu_o, double mu_i,
                                          double phi_s, double phi_d,
                                          double alpha_u, double alpha_v,
                                          std::complex<double> eta,
                                          bool isotropic_g = false);

/**
 * \brief Evaluate the Beckmann distribution-based microfacet BSDF by
 * Walter et al. using the mu_i, mu_o, phi_s, phi_d parameterization. This
 * version only contains the exponential term.
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param phi_s
 *    Azimuthal angle sum
 *
 * \param phi_d
 *    Azimuthal angle difference
 *
 * \param alpha_u
 *    Beckmann roughness parameter in U direction
 *
 * \param alpha_v
 *    Beckmann roughness parameter in V direction
 *
 * \param eta
 *    Relative index of refraction (complex)
 */
extern MTS_EXPORT_LAYER double microfacet_exp(double mu_o, double mu_i,
                                              double phi_s, double phi_d,
                                              double alpha_u, double alpha_v,
                                              std::complex<double> eta);

/**
 * \brief Evaluate the Beckmann distribution-based microfacet BSDF by
 * Walter et al. using the mu_i, mu_o, phi_s, phi_d parameterization. This
 * version only contains the "remainder" component, i.e. Fresnel and
 * some normalization terms.
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param phi_s
 *    Azimuthal angle sum
 *
 * \param phi_d
 *    Azimuthal angle difference
 *
 * \param alpha_u
 *    Beckmann roughness parameter in U direction
 *
 * \param alpha_v
 *    Beckmann roughness parameter in V direction
 *
 * \param eta
 *    Relative index of refraction (complex)
 *
 * \param fresnel_only
 *    Usually, this part of the microfacet model implementation also includes
 *    other low-frequency/constant normalization terms. Optionally, with this
 *    flag, only Fresnel itself is evaluated. (Default=false)
 */
extern MTS_EXPORT_LAYER double microfacet_fresnel(double mu_o, double mu_i,
                                                  double phi_s, double phi_d,
                                                  double alpha_u, double alpha_v,
                                                  std::complex<double> eta,
                                                  bool fresnel_only=false);

/**
 * \brief Evaluate the Beckmann distribution-based microfacet BSDF by
 * Walter et al. using the mu_i, mu_o, phi_s, phi_d parameterization. This
 * version only contains the shadowing-masking term.
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param phi_s
 *    Azimuthal angle sum
 *
 * \param phi_d
 *    Azimuthal angle difference
 *
 * \param alpha_u
 *    Beckmann roughness parameter in U direction
 *
 * \param alpha_v
 *    Beckmann roughness parameter in V direction
 *
 * \param eta
 *    Relative index of refraction (complex)
 *
 * \param isotropic_g
 *    Indicates whether an isotropic version of the shadowing-masking function
 *    should be used. (Default: false)
 */
extern MTS_EXPORT_LAYER double microfacet_G(double mu_o, double mu_i,
                                            double phi_s, double phi_d,
                                            double alpha_u, double alpha_v,
                                            std::complex<double> eta,
                                            bool isotropic_g = false);

/**
 * \brief Smith's 1D shadowing masking term for the Beckmann microfacet
 * distribution
 *
 * \param v
 *    Incident direction
 *
 * \param m
 *    Microsurface normal
 *
 * \param alpha_u
 *    Beckmann roughness parameter in U direction
 *
 * \param alpha_v
 *    Beckmann roughness parameter in V direction
 */
extern MTS_EXPORT_LAYER double smith_G1(const Vector3f &v, const Vector3f &m,
                                        double alpha_u, double alpha_v);

/**
 * \brief Compute the effective roughness for an anisotropic microfacet
 * model
 *
 * \param v
 *    Direction
 *
 * \param alpha_u
 *    Roughness parameter in U direction
 *
 * \param alpha_v
 *    Roughness parameter in V direction
 */
extern MTS_EXPORT_LAYER double project_roughness(const Vector3f &v,
                                                 double alpha_u, double alpha_v);

/**
 * \brief Return Fourier series coefficients for an exponential
 * of a cosine, specifically the expression "exp(A + B*cos(phi + C))"
 *
 * \param A
 *    The 'A' coefficient in the above expression
 *
 * \param B
 *    The 'B' coefficient in the above expression
 *
 * \param C
 *    The 'C' coefficient in the above expression
 *
 * \param relerr
 *    Relative error goal
 *
 * \return
 *    Resulting Fourier series coefficients
 */
extern MTS_EXPORT_LAYER VectorXc exp_cos_fourier_series(double A, double B, double C,
                                                        double relerr);

/**
 * \brief Compute a Fourier series of the Fresnel part for the
 * Beckmann-distribution based anisotropic microfacet BSDF by Walter et al.
 * Covers both the dielectric and conducting case.
 * The result is a 1D Fourier series over phi_d, valid for all phi_s.
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param alpha_u
 *    Beckmann roughness parameter in U direction
 *
 * \param alpha_v
 *    Beckmann roughness parameter in V direction
 *
 * \param eta
 *    Relative index of refraction (complex)
 *
 * \param md
 *    Indicates a desired number of Fourier coefficients
 *
 * \param phi_max
 *    The implementation minimizes the fitting error on the interval [0, phi_max].
 *    If in doubt, set phi_max = math::Pi
 *
 * \param svd_reg
 *    Use old SVD based regularization. Disabled by default.
 *
 * \param fresnel_only
 *    Usually, this part of the microfacet model implementation also includes
 *    other low-frequency/constant normalization terms. Optionally, with this
 *    flag, only Fresnel itself is evaluated. (Default=false)
 *
 * \return
 *    Resulting Fourier series coefficients
 */
extern MTS_EXPORT_LAYER VectorX fresnel_fourier_series(double mu_o, double mu_i,
                                                       double alpha_u, double alpha_v,
                                                       std::complex<double> eta,
                                                       int md, double phi_max,
                                                       bool svd_reg=false,
                                                       bool fresnel_only=false);

/**
 * \brief Compute a Fourier series of the Smith G1 function along either incident
 * or outgoing azimuth angle phi.
 * Can be (diagonally) convolved against an existing phi_s / phi_d Fourier series
 * to model multiplication with the smith G2 Shadowing-Masking component.
 *
 * \param mu
 *    Either incident or outgoing zenith angle cosine
 *
 * \param alpha_u
 *    Beckmann roughness parameter in U direction
 *
 * \param alpha_v
 *    Beckmann roughness parameter in V direction
 *
 * \param m
 *    Indicates a desired number of Fourier coefficients
 *
 * \param n_samples
 *    Indicates a desired number of function samples to be used for Filon integration
 *
 * \return
 *    Resulting Fourier series coefficients
 */
extern MTS_EXPORT_LAYER VectorX smith_G1_fourier_series(double mu, double alpha_u, double alpha_v,
                                                        size_t order, size_t n_samples = 30);

/**
 * \brief Compute a 2D Fourier series of the Beckmann-distribution based
 * microfacet BSDF by Walter et al.
 * Covers both the dielectric and conducting case.
 *
 * \param mu_i
 *    Incident zenith angle cosine
 *
 * \param mu_o
 *    Exitant zenith angle cosine
 *
 * \param alpha_u
 *    Beckmann roughness parameter in U direction
 *
 * \param alpha_v
 *    Beckmann roughness parameter in V direction
 *
 * \param eta
 *    Relative index of refraction (complex)
 *
 * \param ms
 *    Indicates a desired number of Fourier coefficients for first dimension (phi_s)
 *
 * \param md
 *    Indicates a desired number of Fourier coefficients for second dimension (phi_d)
 *    The implementation will blur out higher Frequency content to try to
 *    achieve this number.
 *
 * \param relerr
 *    A relative error threshold after which series terms can safely
 *    be truncated
 *
 * \param[out] result
 *    Storage for the generated Fourier coefficients
 *
 * \param component
 *    Indicates whether which component should be considered:
 *    The full model (default, 0), exponential (1), shadowing-masking (2),
 *    Fresnel (3)
 *
 * \param n_samples_phi_s
 *    Number of samples to use for the Filon integration along phi_s.
 *
 * \remark
 *    In the Python API, the \result output parameter is directly returned
 */
extern MTS_EXPORT_LAYER void microfacet_fourier_series(double mu_o, double mu_i,
                                                       double alpha_u, double alpha_v,
                                                       std::complex<double> eta,
                                                       int ms, int md, double relerr,
                                                       MatrixX &result,
                                                       int component = 0, int n_samples_phi_s = 30, bool svd_reg = false);

extern MTS_EXPORT_LAYER void microfacet_reflection_exp_coeffs(double mu_o, double mu_i,
                                                              double alpha_u, double alpha_v,
                                                              double phi_s, VectorX &c);

extern MTS_EXPORT_LAYER void microfacet_refraction_exp_coeffs(double mu_o, double mu_i,
                                                              double alpha_u, double alpha_v,
                                                              double phi_s, double eta, VectorX &c);

extern MTS_EXPORT_LAYER bool microfacet_inside_lowfreq_interval(double mu_o, double mu_i, double phi_s,
                                                                double alpha_u, double alpha_v,
                                                                std::complex<double> eta,
                                                                double phi_d, double relerr = 1e-3);

NAMESPACE_END(mitsuba)
