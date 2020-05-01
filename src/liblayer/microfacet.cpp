#include <mitsuba/render/fresnel.h>
#include <mitsuba/layer/microfacet.h>
#include <mitsuba/layer/fourier.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/math.h>
#include <enoki/special.h>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <cmath>
#include <chrono>

using namespace std::chrono;

#if defined HAVE_FFTW
    #include <fftw3.h>
    // #define USE_FFTW
#endif

NAMESPACE_BEGIN(mitsuba)

std::atomic<double> microfacet_timer;

double microfacet(double mu_o, double mu_i, double phi_s, double phi_d,
                  double alpha_u, double alpha_v,
                  std::complex<double> eta_,
                  bool isotropic_g) {
    double phi_i = 0.5 * (phi_s - phi_d);
    double phi_o = 0.5 * (phi_s + phi_d);

    double sin_theta_i = safe_sqrt(1-mu_i*mu_i),
           sin_theta_o = safe_sqrt(1-mu_o*mu_o),
           cos_phi_i = std::cos(phi_i),
           sin_phi_i = std::sin(phi_i),
           cos_phi_o = std::cos(phi_o),
           sin_phi_o = std::sin(phi_o);

    Vector3f wi(-sin_theta_i*cos_phi_i,
                -sin_theta_i*sin_phi_i,
                -mu_i);
    Vector3f wo(sin_theta_o*cos_phi_o,
                sin_theta_o*sin_phi_o,
                mu_o);

    bool reflect = -mu_i*mu_o > 0;

    if (mu_o == 0 || mu_i == 0)
        return 0;

    bool conductor = (std::imag(eta_) != 0);
    if (conductor && !reflect)
        return 0;
    std::complex<double> eta = (-mu_i > 0 || conductor) ? eta_ : std::complex<double>(1) / eta_;

    Vector3f wh = normalize(wi + wo * (reflect ? 1.0 : std::real(eta)));
    wh = enoki::mulsign(wh, Frame3f::cos_theta(wh));

    /* The following two visibility checks are usually part of the smith_G1
       implementations, but are considered to be part of the "Fresnel + Remainder"
       term in this framework. */
    if (dot(wi, wh) * Frame3f::cos_theta(wi) <= 0 || dot(wo, wh) * Frame3f::cos_theta(wo) <= 0)
        return 0;

    double cos_theta_h = (double)Frame3f::cos_theta(wh),
           cos_theta_h_2 = cos_theta_h * cos_theta_h;

    double exponent = ((double) (wh.x() * wh.x()) / (alpha_u * alpha_u) +
                       (double) (wh.y() * wh.y()) / (alpha_v * alpha_v)) / cos_theta_h_2;

    /* For the isotropic shadowing-masking function, take max{alpha_u, alpha_v} to ensure energy conservation. */
    double alpha_u_g = isotropic_g ? std::max(alpha_u, alpha_v) : alpha_u;
    double alpha_v_g = isotropic_g ? alpha_u_g : alpha_v;

    double eta_real = std::real(eta),
           eta_imag = std::imag(eta);

    double D = cos_theta_h <= 0 ? 0.0
                                : std::exp(-exponent) / (math::Pi<double> * alpha_u * alpha_v * cos_theta_h_2 * cos_theta_h_2),
           F = !conductor ? (double) std::get<0>(fresnel(float(dot(wi, wh)), float(std::real(eta_))))
                          : (double) fresnel_conductor(float(dot(wi, wh)), enoki::Complex<float>(eta_real, eta_imag)),
           G = smith_G1(wi, wh, alpha_u_g, alpha_v_g) * smith_G1(wo, wh, alpha_u_g, alpha_v_g);

    if (reflect) {
        return F * D * G / (4 * std::abs(mu_i*mu_o));
    } else {
        double sqrt_denom = (double) dot(wi, wh) + std::real(eta) * (double) dot(wo, wh);

        return std::abs(((1 - F) * D * G * std::real(eta) * std::real(eta) * (double) dot(wi, wh)
                         * (double) dot(wo, wh)) / (mu_i*mu_o * sqrt_denom * sqrt_denom));
    }
}

double microfacet_exp(double mu_o, double mu_i, double phi_s, double phi_d,
                      double alpha_u, double alpha_v,
                      std::complex<double> eta_) {
    double phi_i = 0.5 * (phi_s - phi_d);
    double phi_o = 0.5 * (phi_s + phi_d);

    double sin_theta_i = safe_sqrt(1-mu_i*mu_i),
           sin_theta_o = safe_sqrt(1-mu_o*mu_o),
           cos_phi_i = std::cos(phi_i),
           sin_phi_i = std::sin(phi_i),
           cos_phi_o = std::cos(phi_o),
           sin_phi_o = std::sin(phi_o);

    Vector3f wi(-sin_theta_i*cos_phi_i,
                -sin_theta_i*sin_phi_i,
                -mu_i);
    Vector3f wo(sin_theta_o*cos_phi_o,
                sin_theta_o*sin_phi_o,
                mu_o);

    bool reflect = -mu_i*mu_o > 0;

    if (mu_o == 0 || mu_i == 0)
        return 0;

    bool conductor = (std::imag(eta_) != 0);
    if (conductor && !reflect)
        return 0;
    std::complex<double> eta = (-mu_i > 0 || conductor) ? eta_ : std::complex<double>(1) / eta_;

    Vector3f wh = normalize(wi + wo * (reflect ? 1.0 : std::real(eta)));
    wh = enoki::mulsign(wh, Frame3f::cos_theta(wh));

    double cos_theta_h_2 = (double) Frame3f::cos_theta_2(wh);

    double exponent = ((double) (wh.x() * wh.x()) / (alpha_u * alpha_u) +
                       (double) (wh.y() * wh.y()) / (alpha_v * alpha_v)) / cos_theta_h_2;

    return std::exp(-exponent);
}

double microfacet_fresnel(double mu_o, double mu_i, double phi_s, double phi_d,
                          double alpha_u, double alpha_v,
                          std::complex<double> eta_,
                          bool fresnel_only) {
    (void)phi_s; // Unused

    double sin_theta_i = safe_sqrt(1-mu_i*mu_i),
           sin_theta_o = safe_sqrt(1-mu_o*mu_o),
           cos_phi = std::cos(phi_d),
           sin_phi = std::sin(phi_d);

    Vector3f wi(-sin_theta_i, 0, -mu_i);
    Vector3f wo(sin_theta_o*cos_phi, sin_theta_o*sin_phi, mu_o);

    bool reflect = -mu_i*mu_o > 0;

    if (mu_o == 0 || mu_i == 0)
        return 0;

    bool conductor = (std::imag(eta_) != 0);
    if (conductor && !reflect)
        return 0;
    std::complex<double> eta = (-mu_i > 0 || conductor) ? eta_ : std::complex<double>(1) / eta_;

    Vector3f wh = normalize(wi + wo * (reflect ? 1.0 : std::real(eta)));
    wh = enoki::mulsign(wh, Frame3f::cos_theta(wh));

    /* The following two visibility checks are usually part of the smith_G1
       implementations, but are considered to be part of the "Fresnel + Remainder"
       term in this framework. */
    if (dot(wi, wh) * Frame3f::cos_theta(wi) <= 0 || dot(wo, wh) * Frame3f::cos_theta(wo) <= 0)
        return 0;

    double cos_theta_h = (double) Frame3f::cos_theta(wh),
           cos_theta_h_2 = cos_theta_h * cos_theta_h;

    double eta_real = std::real(eta),
           eta_imag = std::imag(eta);

    double D = cos_theta_h == 0 ? 0.0
                                : 1.0 / (math::Pi<double> * alpha_u * alpha_v * cos_theta_h_2 * cos_theta_h_2),
           F = !conductor ? (double) std::get<0>(fresnel(float(dot(wi, wh)), float(std::real(eta_))))
                          : (double) fresnel_conductor(float(dot(wi, wh)), enoki::Complex<float>(eta_real, eta_imag));

    if (fresnel_only) {
        if (reflect) {
            return F;
        } else {
            return std::abs(1-F);
        }
    }

    if (reflect) {
        return F * D / (4.0 * std::abs(mu_i*mu_o));
    } else {
        double sqrt_denom = (double) dot(wi, wh) + std::real(eta) * (double) dot(wo, wh);
        return std::abs(((1 - F) * D * std::real(eta) * std::real(eta) * (double) dot(wi, wh)
                        * (double) dot(wo, wh)) / (mu_i*mu_o * sqrt_denom * sqrt_denom));
    }
}

double microfacet_G(double mu_o, double mu_i, double phi_s, double phi_d,
                    double alpha_u, double alpha_v,
                    std::complex<double> eta_,
                    bool isotropic_g ) {
    double phi_i = 0.5 * (phi_s - phi_d);
    double phi_o = 0.5 * (phi_s + phi_d);

    double sin_theta_i = safe_sqrt(1-mu_i*mu_i),
           sin_theta_o = safe_sqrt(1-mu_o*mu_o),
           cos_phi_i = std::cos(phi_i),
           sin_phi_i = std::sin(phi_i),
           cos_phi_o = std::cos(phi_o),
           sin_phi_o = std::sin(phi_o);

    Vector3f wi(-sin_theta_i*cos_phi_i,
                -sin_theta_i*sin_phi_i,
                -mu_i);
    Vector3f wo(sin_theta_o*cos_phi_o,
                sin_theta_o*sin_phi_o,
                mu_o);

    bool reflect = -mu_i*mu_o > 0;

    if (mu_o == 0 || mu_i == 0)
        return 0;

    bool conductor = (std::imag(eta_) != 0);
    if (conductor && !reflect)
        return 0;
    std::complex<double> eta = (-mu_i > 0 || conductor) ? eta_ : std::complex<double>(1.0) / eta_;

    Vector3f wh = normalize(wi + wo * (reflect ? 1.0 : std::real(eta)));
    wh = enoki::mulsign(wh, Frame3f::cos_theta(wh));

    /* For the isotropic shadowing-masking function, take max{alpha_u, alpha_v}
       to ensure energy conservation. */
    double alpha_u_g = isotropic_g ? std::max(alpha_u, alpha_v) : alpha_u;
    double alpha_v_g = isotropic_g ? alpha_u_g : alpha_v;

    return smith_G1(wi, wh, alpha_u_g, alpha_v_g) *
           smith_G1(wo, wh, alpha_u_g, alpha_v_g);
}

double smith_G1(const Vector3f &v, const Vector3f &m, double alpha_u, double alpha_v) {
    (void)m; // Unused

    /* The following visibility check usually found in smith_G1 implementations is
       treated in the "Fresnel + Remainder" term in this framework. */
    // if (dot(v, m) * Frame3f::cos_theta(v) <= 0)
    //     return 0;

    const double tan_theta = (double) std::abs(Frame3f::tan_theta(v));
    if (tan_theta == 0)
        return 1;

    double alpha = project_roughness(v, alpha_u, alpha_v);

    double a = 1.0 / (alpha * tan_theta);
    if (a < 1.6) {
        /* Use a fast and accurate (<0.35% rel. error) rational
           approximation to the shadowing-masking function */
        const double a_sqr = a * a;
        return (3.535 * a + 2.181 * a_sqr)
             / (1.0 + 2.276 * a + 2.577 * a_sqr);
    }

    return 1.0;
}

VectorX smith_G1_fourier_series(double mu, double alpha_u, double alpha_v,
                                size_t order, size_t n_samples) {
    /* Here we compute the Fourier series of the Smith Shadowing-Masking component,
       for one of the directions phi_i, phi_o via Filon integration.
       Note that we need to sample the function in the domain [0, pi] and perform
       the integration over [0, 2pi] s.t. the "diagonal" versions of these series
       end up being 2pi-periodic in phi_s, phi_d. */
    double cos_theta     = std::abs(mu),
           sin_theta     = std::sqrt(1.0 - cos_theta*cos_theta),
           tan_theta     = sin_theta / cos_theta;

    double phi_step     =  math::Pi<double> / (n_samples - 1),
           cos_phi_prev =  std::cos(phi_step), cos_phi_cur = 1.0,
           sin_phi_prev = -std::sin(phi_step), sin_phi_cur = 0.0,
           two_cos_phi  =  2.0 * cos_phi_prev;

    VectorXc values(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        double cos_phi_next = two_cos_phi*cos_phi_cur - cos_phi_prev,
               sin_phi_next = two_cos_phi*sin_phi_cur - sin_phi_prev;

        double cos_phi_2 = cos_phi_cur*cos_phi_cur;
        double sin_phi_2 = sin_phi_cur*sin_phi_cur;

        double alpha = std::sqrt(cos_phi_2 * alpha_u * alpha_u +
                                 sin_phi_2 * alpha_v * alpha_v);

        double result = 1.0;
        double a = 1.0 / (alpha * tan_theta);
        if (a < 1.6) {
            /* Use a fast and accurate (<0.35% rel. error) rational
               approximation to the shadowing-masking function */
            double a_sqr = a * a;
            result = (3.535 * a + 2.181 * a_sqr)
                 / (1.0 + 2.276 * a + 2.577 * a_sqr);
        }
        values[i] = result;

        cos_phi_prev = cos_phi_cur; cos_phi_cur = cos_phi_next;
        sin_phi_prev = sin_phi_cur; sin_phi_cur = sin_phi_next;
    }

    VectorXc coeffs(order);
    coeffs.setZero();
    filon_integrate_exp(values.data(), n_samples, coeffs.data(), order, 0, 2*math::Pi<double>);

    return coeffs.real();
}

double project_roughness(const Vector3f &v, double alpha_u, double alpha_v) {
    double inv_sin_theta_2 = 1.0 / (double) Frame3f::sin_theta_2(v);

    if ((alpha_u == alpha_v) || inv_sin_theta_2 < 0)
        return alpha_u;

    double cos_phi_2 = (double) (v.x() * v.x()) * inv_sin_theta_2;
    double sin_phi_2 = (double) (v.y() * v.y()) * inv_sin_theta_2;

    return std::sqrt(cos_phi_2 * alpha_u * alpha_u +
                     sin_phi_2 * alpha_v * alpha_v);
}

static int expcos_coefficient_count(double B, double relerr) {
    double prod = 1, invB = 1 / B;
    if (B == 0)
        return 1;

    for (int i = 0; ; ++i) {
        prod /= 1 + i * invB;

        if (prod < relerr)
            return 2*i+1;
    }
}

static double mod_bessel_ratio(double z, double k) {
    const double eps = std::numeric_limits<double>::epsilon(),
                 inv_two_b = 2 / z;

    double i  = (double) k,
           D  = 1 / (inv_two_b * i++),
           Cd = D,
           C  = Cd;

    while (std::abs(Cd) > eps * std::abs(C)) {
        double coeff = inv_two_b * i++;
        D = 1 / (D + coeff);
        Cd *= coeff*D - 1;
        C += Cd;
    }

    return C;
}

static void bessel_functions(double c, double z, int n, VectorX &bessel) {
    bessel.resize(n);

    /* Determine the last ratio and work downwards */
    bessel[n-1] = mod_bessel_ratio(z, n - 1);
    for (int i = n-2; i > 0; --i)
        bessel[i] = z / (2*i + z*bessel[i+1]);

    /* Evaluate the exponentially scaled I0 and correct scaling */
    bessel[0] = enoki::i0e(z) * std::exp(z + c);

    /* Apply the ratios upwards */
    double prod = bessel[0];
    for (int i = 1; i < n; ++i) {
        prod *= bessel[i];
        bessel[i] = prod;
    }
}

static double max_B_heuristic(size_t m, double relerr) {
    if (relerr >= 1e-1)
        return 0.1662*std::pow((double) m, 2.05039);
    else if (relerr >= 1e-2)
        return 0.0818*std::pow((double) m, 2.04982);
    else if (relerr >= 1e-3)
        return 0.0538*std::pow((double) m, 2.05001);
    else if (relerr >= 1e-4)
        return 0.0406*std::pow((double) m, 2.04686);
    else if (relerr >= 1e-5)
        return 0.0337*std::pow((double) m, 2.03865);
    else if (relerr >= 1e-6)
        return 0.0299*std::pow((double) m, 2.02628);
    else {
        Log(Warn, "max_B(): unknown relative error bound!");
        return math::Infinity<double>;
    }
}

void microfacet_reflection_exp_coeffs(double mu_o, double mu_i, double alpha_u, double alpha_v, double phi_s, VectorX &c) {
    c.resize(3);

    double denom = 1.0 / (alpha_u * alpha_v * (mu_i - mu_o)),
           alpha_sqr_sum = (alpha_u * alpha_u + alpha_v * alpha_v),
           alpha_sqr_dif = (alpha_u * alpha_u - alpha_v * alpha_v),
           root = safe_sqrt((1 - mu_i * mu_i) * (1 - mu_o * mu_o)),
           tmp = (mu_i * mu_i + mu_o * mu_o - 2);

    double c0 = 0.5 * denom * denom * alpha_sqr_sum * tmp,
           c1 = denom * denom * alpha_sqr_sum * root,
           c2 = denom * denom * -alpha_sqr_dif * root,
           c3 = 0.5 * denom * denom * -alpha_sqr_dif * (mu_o * mu_o - 1),
           c4 = 0.5 * denom * denom * -alpha_sqr_dif * (mu_i * mu_i - 1);

    std::complex<double> z(c1 + (c3 + c4) * std::cos(phi_s),
                                (c3 - c4) * std::sin(phi_s));

    double A = c0 + c2 * std::cos(phi_s),
           B = std::abs(z),
           C = std::arg(z);

    c[0] = A;
    c[1] = B;
    c[2] = C;
}

void microfacet_refraction_exp_coeffs(double mu_o, double mu_i, double alpha_u, double alpha_v, double phi_s, double eta, VectorX &c) {
    c.resize(3);

    double denom = 1.0 / (alpha_u * alpha_v * (mu_i - eta * mu_o)),
           alpha_sqr_sum = (alpha_u * alpha_u + alpha_v * alpha_v),
           alpha_sqr_dif = (alpha_u * alpha_u - alpha_v * alpha_v),
           root = safe_sqrt((1 - mu_i * mu_i) * (1 - mu_o * mu_o));

    double c0 = 0.5 * denom * denom * alpha_sqr_sum * (mu_i * mu_i - 1 + eta * eta * (mu_o * mu_o - 1)),
           c1 = denom * denom * alpha_sqr_sum * eta * root,
           c2 = denom * denom * -alpha_sqr_dif * eta * root,
           c3 = 0.5 * denom * denom * -alpha_sqr_dif * eta * eta * (mu_o * mu_o - 1),
           c4 = 0.5 * denom * denom * -alpha_sqr_dif * (mu_i * mu_i - 1);

    std::complex<double> z(c1 + (c3 + c4) * std::cos(phi_s),
                                (c3 - c4) * std::sin(phi_s));

    double A = c0 + c2 * std::cos(phi_s),
           B = std::abs(z),
           C = std::arg(z);

    c[0] = A;
    c[1] = B;
    c[2] = C;
}

VectorXc exp_cos_fourier_series(double A, double B, double C, double relerr) {
    VectorXc result;
    int md = expcos_coefficient_count(B, relerr);
    int mdh = md / 2;

    // Create Fourier Series for exp(A + B * cos(phi_d + 0))
    VectorX bessel;
    bessel_functions(A, B, mdh + 1, bessel);

    int truncated_size = bessel.size();
    for (int i = 1; i < bessel.size(); ++i) {
         if (2*std::abs(bessel[i]) < bessel[0] * relerr) {
            truncated_size = i;
            break;
        }
    }
    VectorX bessel_trunc = bessel.head(truncated_size);
    mdh = bessel_trunc.size() - 1;
    md = 2*mdh+1;

    result.resize(md);

    result.head(mdh) = bessel_trunc.tail(mdh).reverse().cast<std::complex<double>>();
    result.tail(mdh + 1) = bessel_trunc.cast<std::complex<double>>();

    if (std::abs(C) > 0) {
        // Shift Fourier Series by phase C
        std::complex<double> exp_phase_inc = std::exp(1i * C);
        std::complex<double> exp_phase = std::exp(-1i * (double) mdh * C);
        for (int i = 0; i < md; ++i) {
            result[i] *= exp_phase;
            exp_phase *= exp_phase_inc;
        }
    }

    return result;
}

inline void sample_function(const std::function<double(double)> &f,
                            double *values, size_t size,
                            double a, double b) {
    double delta = (b - a) / (size - 1);
    for (size_t i = 0; i < size; ++i) {
        double x = a + i*delta;
        values[i] = f(x);
    }
}

inline MatrixX coefficient_conversion_matrix(int m, double phi_a, double phi_b) {
    /* Precompute some sines and cosines */
    VectorX cos_phi_a(m), sin_phi_a(m), cos_phi_b(m), sin_phi_b(m);
    for (int i = 0; i < m; ++i) {
        std::tie(sin_phi_a[i], cos_phi_a[i]) = enoki::sincos(i*phi_a);
        std::tie(sin_phi_b[i], cos_phi_b[i]) = enoki::sincos(i*phi_b);
    }

    MatrixX A(m, m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (i != j) {
                A(i, j) = A(j, i) = (i * cos_phi_b[j] * sin_phi_b[i] +
                                     j * cos_phi_a[i] * sin_phi_a[j] -
                                     i * cos_phi_a[j] * sin_phi_a[i] -
                                     j * cos_phi_b[i] * sin_phi_b[j]) / (i*i - j*j);
            } else if (i != 0) {
                A(i, i) = (std::sin(2*i * phi_b) -
                           std::sin(2*i * phi_a) +
                           2*i * (phi_b - phi_a)) / (4*i);
            } else {
                A(i, i) = phi_b - phi_a;
            }
        }
    }
    return A;
}

VectorX fresnel_fourier_series(double mu_o, double mu_i,
                                double alpha_u, double alpha_v,
                                std::complex<double> eta_,
                                int md, double phi_max,
                                bool svd_reg, bool fresnel_only) {
    int m = md / 2 + 1; // Number of Cosine series coefficients
    VectorX result;

    bool reflect = -mu_i * mu_o > 0;

    double sin_mu_2 = safe_sqrt((1.0 - mu_i * mu_i) * (1.0 - mu_o * mu_o)),
           phi_critical = 0.0;

    bool conductor = (std::imag(eta_) != 0);
    std::complex<double> eta = (-mu_i > 0 || conductor) ? eta_ : std::complex<double>(1) / eta_;

    if (reflect) {
        if (!conductor) {
            double tmp = (2.0*std::real(eta)*std::real(eta) - mu_i*mu_o - 1.0) / sin_mu_2;
            phi_critical = safe_acos(tmp);
        }
    } else if (!reflect) {
        if (conductor) {
            Throw("microfacet_no_exp_fourier_series(): Encountered refraction case for a conductor!");
        }
        double eta_denser = (std::real(eta) > 1 ? std::real(eta) : 1 / std::real(eta));
        double tmp = (1 - eta_denser * mu_i * mu_o) / (eta_denser * sin_mu_2);
        phi_critical = safe_acos(tmp);
    }

    bool phi_critical_inside_interval;
    if (reflect && mu_i > 0) {
        /* For reflection from bottom, phi_critical = 0 results in a high-frequency feature that should be captured. */
        phi_critical_inside_interval = phi_critical < phi_max - math::Epsilon<double>;
    } else {
        phi_critical_inside_interval = phi_critical > math::Epsilon<double> && phi_critical < phi_max - math::Epsilon<double>;
    }

    if (!conductor && phi_critical_inside_interval) {
        /* Uh oh, some high frequency content (critical angle) leaked in the
           generally low frequency part. Increase the number of coefficients so
           that we can capture it. Fortunately, this happens very rarely. */
        m = std::max(m, 100);
        md = 2*m-1;
    }

    VectorX coeffs;
    if (svd_reg) {
        coeffs.resize(m);
    } else {
        coeffs.resize(2*m); // Allocate two times the space for QR regularization below
    }
    coeffs.setZero();

    const int samples = 200;
    auto integrand = std::bind(&microfacet_fresnel, mu_o, mu_i, 0.0, std::placeholders::_1, alpha_u, alpha_v, eta_, fresnel_only);

    if (reflect) {
        if (phi_critical_inside_interval) {
            filon_integrate_cosine(integrand, samples, coeffs.data(), m, 0, phi_critical);
            filon_integrate_cosine(integrand, samples, coeffs.data(), m, phi_critical, phi_max);
        } else {
            filon_integrate_cosine(integrand, samples, coeffs.data(), m, 0, phi_max);
        }
    } else {
        filon_integrate_cosine(integrand, samples, coeffs.data(), m, 0, std::min(phi_critical, phi_max));
    }

    if (phi_max < math::Pi<double> - math::Epsilon<double>) {
        /* The fit only occurs on a subset [0, phi_max], where the Fourier
           basis functions are not orthogonal anymore! The following then
           does a change of basis to proper Fourier coefficients. */

        auto start = high_resolution_clock::now();

        if (svd_reg) {
            MatrixX A = coefficient_conversion_matrix(m, 0, phi_max);

            auto svd = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
            const MatrixX &U = svd.matrixU();
            const MatrixX &V = svd.matrixV();
            const VectorX &sigma = svd.singularValues();

            if (sigma[0] == 0) {
                result.resize(1);
                result(0) = 0;
                return result;
            }

            VectorX temp = VectorX::Zero(m);
            coeffs[0] *= math::Pi<double>;
            coeffs.tail(m-1) *= 0.5 * math::Pi<double>;
            for (int i = 0; i < m; ++i) {
                if (sigma[i] < 1e-9 * sigma[0])
                    break;
                temp += V.col(i) * U.col(i).dot(coeffs) / sigma[i];
            }
            result = temp;
        } else {
            double eps = 1e-5;
            MatrixX A(2*m, m);

            /* We want the fit to be as close-as-possible on the interval [0, phi_max].
             We then add soft regularization constraints for the series to be small outside. */
            A.block(0, 0, m, m) = coefficient_conversion_matrix(m, 0, phi_max);
            A.block(m, 0, m, m) = coefficient_conversion_matrix(m, phi_max, math::Pi<double>) * eps;

            // Scale coefficients
            coeffs *= 0.5*math::Pi<double>;
            coeffs[0] *= 2;

            // QR factorization
            result = A.colPivHouseholderQr().solve(coeffs);
        }

        auto end = high_resolution_clock::now();
        duration<double> diff = (end-start);

        while (true) {
            double old = microfacet_timer;
            double update = old + diff.count();
            if (microfacet_timer.compare_exchange_strong(old, update))
                break;
        }

    } else {
        result = coeffs.head(m);   // No regularization necessary, just get rid of the zero padding from before again.
    }

    VectorX result_exp(md);
    result_exp.head(m - 1) = 0.5*result.tail(m - 1).reverse();
    result_exp[m-1] = result[0];
    result_exp.tail(m - 1) = 0.5*result.tail(m - 1);

    return result_exp;
}

void microfacet_fourier_series(double mu_o, double mu_i,
                               double alpha_u, double alpha_v,
                               std::complex<double> eta_,
                               int ms, int md, double relerr, MatrixX &result,
                               int component, int n_samples_phi_s, bool svd_reg) {
    bool reflect = -mu_i * mu_o > 0;

    bool conductor = (std::imag(eta_) != 0.0);
    std::complex<double> eta = (-mu_i > 0 || conductor) ? eta_ : std::complex<double>(1) / eta_;

    if (!reflect) {
        if (conductor) {
            /* No refraction in conductors */
            result.resize(1, 1);
            result(0, 0) = 0;
            return;
        }
    }

    if (component == 2) {
        /* Only compute shadowing-masking component */
        VectorX smith_G1_coeffs_i = smith_G1_fourier_series(mu_i, alpha_u, alpha_v, 9);
        VectorX smith_G1_coeffs_o = smith_G1_fourier_series(mu_o, alpha_u, alpha_v, 9);

        MatrixX G1_coeffs(9, 9);
        G1_coeffs.setZero();
        G1_coeffs.diagonal(0) = smith_G1_coeffs_o;

        MatrixX G2_coeffs;
        convolve_fourier_series_antidiagonal(smith_G1_coeffs_i, G1_coeffs, G2_coeffs);

        result = G2_coeffs;
        return;
    }

    /* At grazing angles, microfacet BSDFs can contain arbitrarily high frequencies.
       This value sets an upper bound on exponential cosine series parameter 'B' to
       prevent ringing in such cases. */
    double max_B = max_B_heuristic(md/2+1, relerr);

    /* Here we compute the Fourier coefficients of the Microfacet "remainder term", i.e.
       Fresnel and some normalizations. This is isotropic and thus invariant over phi_s,
       so we only compute it once here.
       This part is later convolved with the exponential component, so it makes sense to
       only compute it accurately for the interval [0, phi_max] in which the exponential
       component takes large enough values.
       We find such a conservative interval (useful for all phi_s) with a golden section search here. */
    VectorX fresnel_coeffs;
    if (component == 0 || component == 3) {
        auto gss = [] (const std::function<double(double)>& f, double a, double b, double tol=1e-3) {
            double gr = 0.5*(std::sqrt(5.0) + 1.0);
            double inv_gr = 1.0 / gr;

            while (true) {
                double x1 = b - (b - a) * inv_gr,
                       x2 = a + (b - a) * inv_gr;

                if (std::abs(b - a) < tol)
                    break;
                else if (f(x1) < f(x2))
                    b = x2;
                else
                    a = x1;
            }

            return 0.5*(a + b);
        };

        auto get_phi = [&] (double phi_s) {
            VectorX c;
            if (reflect) {
                microfacet_reflection_exp_coeffs(mu_o, mu_i, alpha_u, alpha_v, phi_s, c);
            } else {
                microfacet_refraction_exp_coeffs(mu_o, mu_i, alpha_u, alpha_v, phi_s, std::real(eta), c);
            }
            double A = c[0];
            double B = c[1];
            double C = c[2];

            if (B > max_B && (std::abs(mu_i) < 0.2 || std::abs(mu_o) < 0.2)) {
                A = A + B - max_B + std::log(enoki::i0e(B) / enoki::i0e(max_B));
                B = max_B;
            }

            double phi_max = safe_acos(1.0 + std::log(relerr) / B);
            return -phi_max - C;
        };

        double phi_s = gss(get_phi, -math::Pi<double>, math::Pi<double>);
        double phi_max = std::min(-get_phi(phi_s), math::Pi<double>);

        fresnel_coeffs = fresnel_fourier_series(mu_o, mu_i,
                                                alpha_u, alpha_v, eta_,
                                                23, phi_max, svd_reg, component==3);
    }

    /* Sample phi_s dimension with regular samples and for each, create a Fourier series in phi_d. */

#ifdef USE_FFTW
    n_samples_phi_s = ms;   // Need as many samples as Fourier orders requested for FFTW integration
#else
    n_samples_phi_s = std::max(2, n_samples_phi_s); // Need at least 2 samples in order to do Filon integration.
#endif

    bool isotropic = false;
    if (alpha_u == alpha_v) {
        /* Isotropic Microfacet model. Only compute a 1D Fourier series and
           zero pad coefficient matrix to have dimension ms x md later. */
        n_samples_phi_s = 1;
        isotropic = true;
    }

    int md_max = 0;
    std::vector<VectorXc> coeffs;
    for (int s = 0; s < n_samples_phi_s; ++s) {
        #ifdef USE_FFTW
            int n_steps = n_samples_phi_s;
        #else
            int n_steps = n_samples_phi_s - 1;
        #endif

        double phi_s = 2 * math::Pi<double> * s / n_steps;

        if (n_samples_phi_s == 1) phi_s = 0;    // Handle division by zero for isotropic cases where ms == 1

        VectorX c;
        if (reflect) {
            microfacet_reflection_exp_coeffs(mu_o, mu_i, alpha_u, alpha_v, phi_s, c);
        } else {
            microfacet_refraction_exp_coeffs(mu_o, mu_i, alpha_u, alpha_v, phi_s, std::real(eta), c);
        }
        double A = c[0];
        double B = c[1];
        double C = c[2];

        /* Minor optimization: don't even bother computing the Fourier series
           if the contribution to the scattering model is miniscule */
        if (enoki::i0e(B) * std::exp(A + B) < 1e-10 && component == 0) {
            VectorXc final_coeffs(1);
            final_coeffs.setZero();
            md_max = std::max(md_max, (int) final_coeffs.size());
            coeffs.push_back(final_coeffs);
            continue;
        }

        if (B > max_B && (std::abs(mu_i) < 0.2 || std::abs(mu_o) < 0.2)) {
            A = A + B - max_B + std::log(enoki::i0e(B) / enoki::i0e(max_B));
            B = max_B;
        }

        /* Compute Fourier coefficients of the exponential term */
        VectorXc exp_coeffs = exp_cos_fourier_series(A, B, C, relerr);

        /* Perform discrete convolution of the exponential & Fresnel series, if requested */
        VectorXc final_coeffs;
        if (component == 0) {
            /* full model */
            convolve_fourier_series(fresnel_coeffs, exp_coeffs, final_coeffs);
        } else if (component == 1) {
            /* exponential part only */
            final_coeffs = exp_coeffs;
        } else if (component == 3) {
            /* Fresnel + remainder part */
            final_coeffs = fresnel_coeffs.cast<std::complex<double>>();
        }

        /* Already truncate very low coefficients before considering integration over phi_s */
        int md_tmp = final_coeffs.size(),
            mdh_tmp = md_tmp / 2;
        double zero_coeff = std::abs(final_coeffs[mdh_tmp]);
        double ref = zero_coeff * relerr;

        if (zero_coeff < relerr) {
            final_coeffs.resize(1);
            final_coeffs[0] = 0;
        } else {
            int md_trunc = md_tmp;
            for (int d = mdh_tmp; d >= 0; --d) {
                int od = d + mdh_tmp;
                if (std::abs(final_coeffs[od]) >= ref) {
                    md_trunc = 2*d+1;
                    break;
                }
            }
            int mdh_trunc = md_trunc / 2;

            VectorXc truncated = final_coeffs.segment(mdh_tmp - mdh_trunc, md_trunc);
            final_coeffs.resize(truncated.size());
            final_coeffs = truncated;
        }

        md_max = std::max(md_max, (int) final_coeffs.size());
        coeffs.push_back(final_coeffs);
    }

    /* Collect data in matrix */
    int md_max_half = md_max / 2;
    MatrixXc temp(n_samples_phi_s, md_max);
    temp.setZero();
    for (int i = 0; i < n_samples_phi_s; ++i) {
        int md = coeffs[i].size();
        int mdh = md / 2;
        temp.block(i, md_max_half - mdh, 1, md) = coeffs[i].transpose();
    }

    MatrixX result_tmp(ms, md_max);
    result_tmp.setZero();

    /* Integrate the sampled Fourier series over phi_s. */

    if (ms == 1) {
        /* We only have a 1D Fourier series, thus no integration over phi_s needed. */
        result_tmp.row(0) = temp.row(0).real();
    } else {
        #ifdef USE_FFTW
            /* Use FFT to compute Fourier Series of phi_s dimension */
            std::complex<double> *data       = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * ms),
                                 *spectrum   = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * ms);

            fftw_plan_with_nthreads(1);
            fftw_plan plan = fftw_plan_dft_1d(ms, (fftw_complex *) data, (fftw_complex *) spectrum, FFTW_FORWARD, FFTW_ESTIMATE);

            for (int d = 0; d < md_max; ++d) {
                for (int s = 0; s < ms; ++s) {
                    data[s] = temp(s, d);
                }
                fftw_execute(plan);
                for (int s = 0; s < ms; ++s) {
                    temp(s, d) = spectrum[s] / (double) ms;
                }
            }

            fftw_destroy_plan(plan);
            fftw_free(spectrum);
            fftw_free(data);

            /* Reorder Fourier coefficients. And we know that the coefficients must be real. */
            int msh = ms / 2;
            result_tmp.row(msh) = temp.row(0).real();
            result_tmp.block(0, 0, msh, md_max) = temp.block(msh + 1, 0, msh, md_max).real();
            result_tmp.block(msh + 1, 0, msh, md_max) = temp.block(1, 0, msh, md_max).real();
        #else
            if (isotropic || component == 3) {
                /* No integration over phi_s necessary */
                int msh = ms / 2;
                result_tmp.block(msh, 0, 1, md_max) = temp.real();
            } else {
                /* Compute Fourier series over phi_s with Filon integration.
                   We can save some time here by exploiting symmetry. Only integrate
                   one half of the data and then mirror it. */
                MatrixXc result_tmp_complex(ms, md_max);
                result_tmp_complex.setZero();
                for (int d = 0; d <= md_max_half; ++d) {
                    int od = d + md_max_half;
                    filon_integrate_exp(temp.col(od).data(), n_samples_phi_s, result_tmp_complex.col(od).data(), ms);
                }
                result_tmp_complex.block(0, 0, ms, md_max_half) = result_tmp_complex.block(0, md_max_half+1, ms, md_max_half);
                result_tmp_complex.block(0, 0, ms, md_max_half) = result_tmp_complex.block(0, 0, ms, md_max_half).colwise().reverse().eval();
                result_tmp_complex.block(0, 0, ms, md_max_half) = result_tmp_complex.block(0, 0, ms, md_max_half).rowwise().reverse().eval();

                /* We know that the coefficients must be real. */
                result_tmp = result_tmp_complex.real();
            }

        #endif
    }

    for (int s = 0; s < ms; ++s) {
      for (int d = 0; d < md_max; ++d) {
        if (std::abs(result_tmp(s, d)) < 1e-9) result_tmp(s, d) = 0;
      }
    }

    /* Truncate Fourier series */
    int md_trunc = md_max,
        ms_trunc = ms;

    int msh = ms / 2,
        mdh = md_max_half;

    double zero_coeff = std::abs(result_tmp(msh, mdh));
    double ref = zero_coeff * relerr * 0.01;

    // Check in phi_s dimension
    for (int s = 0; s <= msh; ++s) {
        bool keep = false;
        for (int d = -mdh; d <= mdh; ++d) {
            double value = std::abs(result_tmp(s + msh, d + mdh));
            if (value > ref) keep = true;
        }
        if (keep) {
            ms_trunc = 2*s+1;
        } else {
            break;
        }
    }

    // Check in phi_d dimension
    for (int d = 0; d <= mdh; ++d) {
        bool keep = false;
        for (int s = -msh; s <= msh; ++s) {
            double value = std::abs(result_tmp(s + msh, d + mdh));
            if (value > ref) keep = true;
        }
        if (keep) {
            md_trunc = 2*d+1;
        } else {
            break;
        }
    }

    int msh_trunc = ms_trunc / 2,
        mdh_trunc = md_trunc / 2;
    result.resize(ms_trunc, md_trunc);

    result = result_tmp.block(msh - msh_trunc, mdh - mdh_trunc, ms_trunc, md_trunc);

    if (component == 0) {
        /* Requested full model: add Smith Shadowing-masking term via 2 diagonal convolutions */
        VectorX smith_G1_coeffs_i = smith_G1_fourier_series(mu_i, alpha_u, alpha_v, 19);
        VectorX smith_G1_coeffs_o = smith_G1_fourier_series(mu_o, alpha_u, alpha_v, 19);

        MatrixX tmp;
        convolve_fourier_series_diagonal(smith_G1_coeffs_o, result, tmp);
        convolve_fourier_series_antidiagonal(smith_G1_coeffs_i, tmp, result);
    }
}

bool microfacet_inside_lowfreq_interval(double mu_o, double mu_i, double phi_s,
                                        double alpha_u, double alpha_v,
                                        std::complex<double> eta,
                                        double phi_d, double relerr) {
    bool reflect = -mu_i * mu_o > 0;

    VectorX c;
    if (reflect) {
        microfacet_reflection_exp_coeffs(mu_o, mu_i, alpha_u, alpha_v, phi_s, c);
    } else {
        microfacet_refraction_exp_coeffs(mu_o, mu_i, alpha_u, alpha_v, phi_s, std::real(eta), c);
    }
    double B = c[1];
    double C = c[2];

    /* Ideally, we would also remove high frequencies here,
       but then this would have to depend on our chosen md as well. */
    // if (B > max_B) {
    //     A = A + B - max_B + std::log(enoki::i0e(B) / enoki::i0e(max_B));
    //     B = max_B;
    // }

    double phi_max = safe_acos(1.0 + std::log(relerr) / B),
           phi_a = -phi_max - C,
           phi_b = +phi_max - C;

    if (phi_d > phi_a && phi_d < phi_b)
        return true;

    return false;
}

NAMESPACE_END(mitsuba)
