#include <mitsuba/layer/phase.h>
#include <mitsuba/layer/microfacet.h>
#include <mitsuba/core/math.h>
#include <enoki/special.h>

NAMESPACE_BEGIN(mitsuba)

double henyey_greenstein(double mu_o, double mu_i, double phi_d, double g) {
    double cos_theta = mu_i * mu_o + std::cos(phi_d) *
                       std::sqrt((1 - mu_i*mu_i) * (1 - mu_o*mu_o));

    double temp = 1.0 + g*g - 2.0 * g * cos_theta;

    return math::InvFourPi<double> * (1 - g*g) / (temp * std::sqrt(temp));
}

double von_mises_fisher(double mu_o, double mu_i, double phi_d, double kappa) {
    double cos_theta = mu_i * mu_o + std::cos(phi_d) *
                       std::sqrt((1 - mu_i*mu_i) * (1 - mu_o*mu_o));

    double scale;
    if (kappa == 0) {
        scale = 1 / (4 * math::Pi<double>);
    } else {
        scale = kappa / (4 * math::Pi<double> * std::sinh(kappa));
    }

    return scale * std::exp(kappa * cos_theta);
}

void henyey_greenstein_fourier_series(double mu_o, double mu_i, double g, int md,
                                      double relerr, VectorX &result) {
    if (g == 0) {
        result.resize(1);
        result[0] = math::InvFourPi<double>;
        return;
    }

    int even_coeffs = (md / 2 + 1);

    /* Compute A and B coefficients */
    double a = 1 + g*g - 2*g*mu_i*mu_o;
    double b = -2 * g * safe_sqrt((1 - mu_i*mu_i) * (1 - mu_o*mu_o));

    /* Find the first two Fourier coefficients using elliptic integrals */
    double abs_b = std::abs(b), arg = std::sqrt(2*abs_b / (a + abs_b));
    double K = enoki::comp_ellint_1(arg), E = enoki::comp_ellint_2(arg);
    double sqrt_ab = std::sqrt(a + abs_b), temp = (1 - g*g) * (0.5 * math::InvPi<double> * math::InvPi<double>);

    double coeff0 = (E * temp * sqrt_ab) / (a*a - b*b);
    double coeff1 = b == 0 ? 0 : (enoki::sign(b) * temp / (abs_b * sqrt_ab) * (K - a / (a - abs_b) * E));

    size_t m = std::max(even_coeffs * 2, 500);
    double *s = (double *) alloca(sizeof(double) * (m + 1));

    /* Compute the ratio between the $m$-th and $m+1$-th term
       using a second-order Taylor expansion */
    double z     =  a / safe_sqrt(a*a - b*b),
           delta =  z / safe_sqrt(z*z - 1);
    s[m] = (1 + 1 / (double) (2*m) - (1+3*z) / (double) (8*m*m)) * std::sqrt((z-1) / (z+1));

    do {
        /* Work backwards using a recurrence scheme */
        --m;
        s[m] = (2*m+3) / (4*(m+1) * delta - (2*m+1) * s[m+1]);
    } while (m != 0);

    /* Simple to get a bit of extra accuracy here: apply a correction
       in case s[0] does not quite match the known reference value */
    double C = 0.0;
    if (s[0] != 0)
        C = coeff1 / (coeff0 * s[0]);

    /* Now, multiply all ratios together to get the desired value */
    VectorX temp_result;
    temp_result.resize(even_coeffs);
    temp_result[0] = coeff0;

    int final_size = temp_result.size();
    double prod = coeff0 * C * 2;
    for (int j = 0; j < even_coeffs - 1; ++j) {
        if (prod == 0 || std::abs(prod) < coeff0 * relerr) {
            final_size = j+1;
            break;
        }
        prod *= s[j];
        if (j % 2 == 0) {
            temp_result[j+1] = prod;
        }
        else {
            temp_result[j+1] = prod * enoki::sign(g);
        }
    }

    /* Convert even Fourier series to exponential Fourier series format */
    int mdh = final_size - 1;
    result.resize(2*mdh+1);
    result.head(mdh) = 0.5 * temp_result.segment(1, mdh).reverse();
    result[mdh] = temp_result[0];
    result.tail(mdh) = 0.5 * temp_result.segment(1, mdh);
}

void von_mises_fisher_fourier_series(double mu_o, double mu_i, double kappa,
                                     double relerr, VectorX &result) {
    double A = kappa * mu_i * mu_o;
    double B = kappa * safe_sqrt((1 - mu_i*mu_i) * (1 - mu_o*mu_o));

    VectorXc temp_result = exp_cos_fourier_series(A, B, 0, relerr);
    result = temp_result.real();
}

NAMESPACE_END(mitsuba)
