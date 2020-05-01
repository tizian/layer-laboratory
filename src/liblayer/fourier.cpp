#include <mitsuba/layer/fourier.h>
#include <mitsuba/core/spline.h>

#if defined HAVE_FFTW
    #include <fftw3.h>
#endif

NAMESPACE_BEGIN(mitsuba)

float eval_cosine_series_1d(const float *coeffs, size_t m, float phi) {
    if (m <= 0) return 0.0f;

    auto sc = enoki::sincos(phi * enoki::arange<Float32P>());
    auto inc = enoki::sincos(Float32P::Size * phi);

    auto sum = enoki::zero<Float32P>();
    for (size_t k = 0; k < m; k += Float32P::Size) {
        auto coeff = enoki::load<Float32P>(coeffs + k);
        sum += coeff * sc.second;
        sc = trig_addition(sc, inc);
    }
    return enoki::hsum(sum);
}

float eval_fourier_series_2d(const float *coeffs, size_t ms, size_t md, float phi_s, float phi_d) {
    if (ms*md <= 0) return 0.0f;

    int msh = (int) ms / 2;

    // Prepare 1d Fourier series over phi_d
    size_t pd = pad<Float32P::Size>(md);
    float *coeffs_re = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, pd, true);
    float *coeffs_im = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, pd, true);

    auto sc = enoki::sincos(phi_s * float(-msh));
    auto inc = enoki::sincos(phi_s);

    const float *source = coeffs;
    for (int s = -msh; s <= msh; ++s) {
        for (size_t d = 0; d < pd; ++d) {
            size_t od = d * Float32P::Size;

            Float32P re = enoki::load<Float32P>(coeffs_re + od);
            Float32P im = enoki::load<Float32P>(coeffs_im + od);
            Float32P c  = enoki::load<Float32P>(source);

            re += sc.second * c;
            im += sc.first * c;

            enoki::store_unaligned<Float32P>(coeffs_re + od, re);
            enoki::store_unaligned<Float32P>(coeffs_im + od, im);
            source += Float32P::Size;
        }
        sc = trig_addition(sc, inc);
    }

    // Evaluate 1d Fourier series over phi_d
    auto sum = enoki::zero<Float32P>();

    auto scP = enoki::sincos(phi_d * enoki::arange<Float32P>());
    auto incP = enoki::sincos(Float32P::Size * phi_d);

    for (size_t d = 0; d < pd; ++d) {
        size_t od = d * Float32P::Size;
        auto re = enoki::load<Float32P>(coeffs_re + od);
        auto im = enoki::load<Float32P>(coeffs_im + od);

        sum += re * scP.second - im * scP.first;

        scP = trig_addition(scP, incP);
    }
    return 2 * enoki::hsum(sum) - coeffs_re[0];
}

Color3f eval_3_fourier_series_2d(float * const coeffs[3], size_t ms, size_t md, float phi_s, float phi_d) {
    if (ms*md <= 0) return 0.0f;

    int msh = (int) ms / 2;

    // Prepare 1d Fourier series over phi_d
    size_t pd = pad<Float32P::Size>(md);
    float *coeffs_re[3];
    float *coeffs_im[3];
    for (size_t ch = 0; ch < 3; ++ch) {
        coeffs_re[ch] = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, pd, true);
        coeffs_im[ch] = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, pd, true);
    }

    auto sc = enoki::sincos(phi_s * float(-msh));
    auto inc = enoki::sincos(phi_s);

    const float *source_y = coeffs[0];
    const float *source_r = coeffs[1];
    const float *source_b = coeffs[2];
    for (int s = -msh; s <= msh; ++s) {
        for (size_t d = 0; d < pd; ++d) {
            size_t od = d * Float32P::Size;

            Float32P re_y = enoki::load<Float32P>(coeffs_re[0] + od);
            Float32P im_y = enoki::load<Float32P>(coeffs_im[0] + od);
            Float32P c_y  = enoki::load<Float32P>(source_y);
            re_y += sc.second * c_y;
            im_y += sc.first * c_y;
            enoki::store_unaligned<Float32P>(coeffs_re[0] + od, re_y);
            enoki::store_unaligned<Float32P>(coeffs_im[0] + od, im_y);
            source_y += Float32P::Size;

            Float32P re_r = enoki::load<Float32P>(coeffs_re[1] + od);
            Float32P im_r = enoki::load<Float32P>(coeffs_im[1] + od);
            Float32P c_r  = enoki::load<Float32P>(source_r);
            re_r += sc.second * c_r;
            im_r += sc.first * c_r;
            enoki::store_unaligned<Float32P>(coeffs_re[1] + od, re_r);
            enoki::store_unaligned<Float32P>(coeffs_im[1] + od, im_r);
            source_r += Float32P::Size;

            Float32P re_b = enoki::load<Float32P>(coeffs_re[2] + od);
            Float32P im_b = enoki::load<Float32P>(coeffs_im[2] + od);
            Float32P c_b  = enoki::load<Float32P>(source_b);
            re_b += sc.second * c_b;
            im_b += sc.first * c_b;
            enoki::store_unaligned<Float32P>(coeffs_re[2] + od, re_b);
            enoki::store_unaligned<Float32P>(coeffs_im[2] + od, im_b);
            source_b += Float32P::Size;
        }
        sc = trig_addition(sc, inc);
    }

    // Evaluate 1d Fourier series over phi_d
    auto sum_y = enoki::zero<Float32P>();
    auto sum_r = enoki::zero<Float32P>();
    auto sum_b = enoki::zero<Float32P>();

    auto scP = enoki::sincos(phi_d * enoki::arange<Float32P>());
    auto incP = enoki::sincos(Float32P::Size * phi_d);

    for (size_t d = 0; d < pd; ++d) {
        size_t od = d * Float32P::Size;

        auto re_y = enoki::load<Float32P>(coeffs_re[0] + od);
        auto im_y = enoki::load<Float32P>(coeffs_im[0] + od);
        sum_y += re_y * scP.second - im_y * scP.first;

        auto re_r = enoki::load<Float32P>(coeffs_re[1] + od);
        auto im_r = enoki::load<Float32P>(coeffs_im[1] + od);
        sum_r += re_r * scP.second - im_r * scP.first;

        auto re_b = enoki::load<Float32P>(coeffs_re[2] + od);
        auto im_b = enoki::load<Float32P>(coeffs_im[2] + od);
        sum_b += re_b * scP.second - im_b * scP.first;

        scP = trig_addition(scP, incP);
    }
    float y = 2 * enoki::hsum(sum_y) - coeffs_re[0][0];
    float r = 2 * enoki::hsum(sum_r) - coeffs_re[1][0];
    float b = 2 * enoki::hsum(sum_b) - coeffs_re[2][0];

    float g = 1.39829f*y - 0.100913f*b - 0.297375f*r;
    return Color3f(r, g, b);
}

namespace {
    /// Filon cosine integration over a single spline segment (used by filon_integrate_cosine)
    inline void filon_cos(double phi[2], double f[3], size_t lmax, double *output) {
        double h = phi[1] - phi[0], inv_h = 1/h;

        output[0] += (math::InvPi<double> / 6.0) * h * (f[0]+4*f[1]+f[2]);

        double cos_phi_0_prev =  std::cos(phi[0]), cos_phi_0_cur = 1.0,
               cos_phi_1_prev =  std::cos(phi[1]), cos_phi_1_cur = 1.0,
               sin_phi_0_prev = -std::sin(phi[0]), sin_phi_0_cur = 0.0,
               sin_phi_1_prev = -std::sin(phi[1]), sin_phi_1_cur = 0.0,
               two_cos_phi_0  =  2.0 * cos_phi_0_prev,
               two_cos_phi_1  =  2.0 * cos_phi_1_prev;

        const double term_0 = 3*f[0]-4*f[1]+f[2],
                     term_1 = f[0]-4*f[1]+3*f[2],
                     term_2 = 4*(f[0]-2*f[1]+f[2]);

        for (size_t l = 1; l < lmax; ++l) {
            double cos_phi_0_next = two_cos_phi_0*cos_phi_0_cur - cos_phi_0_prev,
                   cos_phi_1_next = two_cos_phi_1*cos_phi_1_cur - cos_phi_1_prev,
                   sin_phi_0_next = two_cos_phi_0*sin_phi_0_cur - sin_phi_0_prev,
                   sin_phi_1_next = two_cos_phi_1*sin_phi_1_cur - sin_phi_1_prev;

            double inv_l    = 1.0 / l,
                   inv_l2_h  = inv_h*inv_l*inv_l,
                   inv_l3_h2 = inv_l2_h*inv_l*inv_h;

            output[l] += (2 * math::InvPi<double>) *
               ((inv_l2_h * (term_0 * cos_phi_0_next + term_1 * cos_phi_1_next) +
                 inv_l3_h2 * term_2 * (sin_phi_0_next - sin_phi_1_next) +
                 inv_l * (f[2] * sin_phi_1_next - f[0] * sin_phi_0_next)));

            cos_phi_0_prev = cos_phi_0_cur; cos_phi_0_cur = cos_phi_0_next;
            cos_phi_1_prev = cos_phi_1_cur; cos_phi_1_cur = cos_phi_1_next;
            sin_phi_0_prev = sin_phi_0_cur; sin_phi_0_cur = sin_phi_0_next;
            sin_phi_1_prev = sin_phi_1_cur; sin_phi_1_cur = sin_phi_1_next;
        }
    }
};

void filon_integrate_cosine(const std::function<double(double)> &f, size_t size,
                            double *coeffs, size_t orders,
                            double a, double b) {
    /* Avoid numerical overflow issues for extremely small intervals */
    if (std::abs(b-a) < 1e-15) {
        return;
    }

    if (size % 2 == 0)
        ++size;

    double value[3], phi[2],
           delta = (b-a) / (size - 1);
    phi[0] = a; value[0] = f(a);

    for (size_t i = 0; i < (size - 1) / 2; ++i) {
        phi[1]   = phi[0] + 2*delta;
        value[1] = f(phi[0] + delta);
        value[2] = f(phi[1]);

        filon_cos(phi, value, orders, coeffs);

        value[0] = value[2];
        phi[0]   = phi[1];
    }
}

void filon_integrate_exp(const std::complex<double> *values, size_t size,
                         std::complex<double> *coeffs, size_t orders,
                         double a, double b) {
    /* Avoid numerical overflow issues for extremely small intervals */
    if (std::abs(b - a) < 1e-15 || size == 0)
        return;
    if (orders % 2 == 0)
        Throw("filon_integrate_exp: 'orders' parameter must be odd!");

    double delta = (b - a) / (size - 1); // Width of each spline segment

    double inv_delta = 1 / delta,
           inv_delta2 = inv_delta*inv_delta,
           inv_delta4 = inv_delta2*inv_delta2;

    int mh = (int) orders / 2;

    std::complex<double> exp_delta = std::exp(-1i * delta),
                         exp_mh_delta = std::exp(1i * (double) mh * delta),
                         exp0_b = std::exp(1i * (double) mh * a),
                         inc0_b = std::exp(-1i * a);

    for (size_t i = 0; i < size - 1; ++i) {
        std::complex<double> f0 = values[i],
                             f1 = values[i+1];

        // Approximate derivatives using central differences
        std::complex<double> d0, d1;
        if (i > 0)
            d0 = 0.5 * (f1 - values[i-1]);
        else
            d0 = f1 - f0;

        if (i + 2 < size)
            d1 = 0.5 * (values[i+2] - f0);
        else
            d1 = f1 - f0;

        // Fit cubic spline of form Ax^3 + Bx^2 + Cx + D in [0,1]
        std::complex<double> A = d0 + d1 + 2.0*f0 - 2.0*f1,
                             B = 3.0*f1 - 3.0*f0 - 2.0*d0 - d1,
                             C = d0,
                             D = f0;

        // Integrate spline against basis functions, also rescaled to [0,1]:
        // exp(-i l ((x1 - x0) x + x0))

        coeffs[mh] += 0.5 * math::InvPi<double> * delta *
                      (3.0 * A + 4.0 * B + 6.0 * C + 12.0 * D) / 12.0;

        std::complex<double> exp0 = exp0_b,
                             inc0 = inc0_b,
                             exp1 = exp0 * exp_mh_delta,
                             inc1 = inc0 * exp_delta;

        double l = -mh;
        for (int li = -mh; li <= mh; ++li) {
            if (li != 0) {
                double inv_l = 1.0 / l,
                       inv_l2 = inv_l * inv_l,
                       inv_l4 = inv_l2 * inv_l2;

                std::complex<double> tmp0 = 6.0*A + l*delta*(2i*B + l*delta*(-C - 1i*D*l*delta)),
                                     tmp1 = -6.0*A - 2i*(3.0*A + B)*l*delta +
                                            (3.0*A + 2.0*B + C)*(l*l)*(delta*delta) +
                                            1i*(A + B + C + D)*(l*l*l)*(delta*delta*delta),
                                     tmp = tmp0 * exp0 + tmp1 * exp1;

                coeffs[li+mh] += 0.5 * math::InvPi<double> * delta * inv_delta4 * inv_l4 * tmp;
            }

            exp0 *= inc0;
            exp1 *= inc1;
            l += 1.0;
        }

        exp0_b *= exp_mh_delta;
        inc0_b *= exp_delta;
    }
}

void convolve_fourier_series(const VectorX &a, const VectorXc &b, VectorXc &result) {
    int n = a.size() + b.size() - 1;
    result.resize(n);
    result.setZero();

    for (int i = 0; i < result.size(); ++i) {
        int kmin = (i >= b.size() - 1) ? (i - (b.size() - 1)) : 0;
        int kmax = (i <  a.size() - 1) ? i                    : (a.size() - 1);

        for (int k = kmin; k <= kmax; ++k) {
            result[i] += (a[k] * b[i - k]);
        }
    }
}

void convolve_fourier_series_diagonal(const VectorX &a, const MatrixX &B, MatrixX &result) {
    assert(a.size() % 2 == 1 && B.rows() % 2 == 1 && B.cols() % 2 == 1);

    result.resize(B.rows() + a.size() - 1, B.cols() + a.size() - 1);
    result.setZero();

    int n_diags = std::max(B.rows() - 1, B.cols() - 1);

    result.setZero();
    for (int d = -n_diags; d <= n_diags; ++d) {
        int na = a.size();
        int nb = B.diagonal(d).size();
        int nc = result.diagonal(d).size();

        for (Eigen::Index i = 0; i < nc; ++i) {
            Eigen::Index kmin = (i >= na - 1) ? (i - (na - 1)) : 0;
            Eigen::Index kmax = (i <  nb - 1) ? i              : (nb - 1);

            for (Eigen::Index k = kmin; k <= kmax; ++k) {
                result.diagonal(d)[i] += (B.diagonal(d)[k] * a[i - k]);
            }
        }
    }
}

void convolve_fourier_series_antidiagonal(const VectorX &a, const MatrixX &B, MatrixX &result) {
    assert(a.size() % 2 == 1 && B.rows() % 2 == 1 && B.cols() % 2 == 1);

    result.resize(B.rows() + a.size() - 1, B.cols() + a.size() - 1);
    result.setZero();

    int n_diags = std::max(B.rows() - 1, B.cols() - 1);

    result.setZero();
    for (int d = -n_diags; d <= n_diags; ++d) {
        int na = a.size();
        int nb = B.diagonal(d).size();
        int nc = result.diagonal(d).size();

        for (Eigen::Index i = 0; i < nc; ++i) {
            Eigen::Index kmin = (i >= na - 1) ? (i - (na - 1)) : 0;
            Eigen::Index kmax = (i <  nb - 1) ? i              : (nb - 1);

            for (Eigen::Index k = kmin; k <= kmax; ++k) {
                result.diagonal(d)[i] += (B.rowwise().reverse().diagonal(d)[k] * a[i - k]);
            }
        }
    }

    result = result.rowwise().reverse().eval();
}

void fftw_transform_c2c(const std::complex<double> *values, size_t size,
                        std::complex<double> *coeffs) {
#if !defined(HAVE_FFTW)
    (void) values; (void) size; (void) coeffs;
    Throw("fftw_transform_c2c: You need to recompile with support for FFTW!");
#else
    if (size % 2 == 0)
        Throw("fftw_transform_c2c: 'size' must be odd!");

    std::complex<double> *data     = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * size),
                         *spectrum = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * size);

    fftw_plan_with_nthreads(1);
    fftw_plan plan = fftw_plan_dft_1d(size, (fftw_complex *) data, (fftw_complex *) spectrum, FFTW_FORWARD, FFTW_ESTIMATE);
    for (size_t i = 0; i < size; ++i) {
        data[i] = values[i];
    }

    fftw_execute(plan);

    double scale = 1.0 / size;
    size_t half = size / 2;
    for (size_t i = 0; i <= half; ++i) {
        coeffs[half + i] = spectrum[i] * scale;
    }
    for (size_t i = 0; i < half; ++i) {
        coeffs[i] = spectrum[half + 1 + i] * scale;
    }

    fftw_destroy_plan(plan);
    fftw_free(spectrum);
    fftw_free(data);
#endif
}

void fftw_transform_r2c(const double *values, size_t size,
                        std::complex<double> *coeffs) {
#if !defined(HAVE_FFTW)
    (void) values; (void) size; (void) coeffs;
    Throw("fftw_transform_r2c: You need to recompile with support for FFTW!");
#else
    if (size % 2 == 0)
        Throw("fftw_transform_r2c: 'size' must be odd!");

    size_t result_size = size / 2 + 1;

    double  *data                  = (double *) fftw_malloc(sizeof(double) * size);
    std::complex<double> *spectrum = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * result_size);

    fftw_plan_with_nthreads(1);
    fftw_plan plan = fftw_plan_dft_r2c_1d(size, (double *) data, (fftw_complex *) spectrum, FFTW_ESTIMATE);
    for (size_t i = 0; i < size; ++i) {
        data[i] = values[i];
    }

    fftw_execute(plan);

    double scale = 1.0 / size;
    size_t half = size / 2;
    for (size_t i = 0; i < result_size; ++i) {
        coeffs[half + i] = spectrum[i] * scale;
    }
    for (size_t i = 1; i < result_size; ++i) {
        coeffs[half - i] = std::conj(spectrum[i]) * scale;
    }

    fftw_destroy_plan(plan);
    fftw_free(spectrum);
    fftw_free(data);
#endif
}

extern MTS_EXPORT_LAYER MatrixX fftw_transform_2d(const MatrixX &values) {
    #if !defined(HAVE_FFTW)
        (void) values;
        Throw("fftw_transform_2d: You need to recompile with support for FFTW!");
    #else
        size_t ms = values.rows(),
               md = values.cols();

        if (ms % 2 == 0 || md % 2 == 0)
            Throw("fftw_transform_2d: Both dimensions of the input data must be odd!");

        MatrixXc coeffs(ms, md);

        /* First, individually transform the phi_d slices (rows) of the values */
        size_t md_spectrum = md / 2 + 1;

        double  *data_d                  = (double *) fftw_malloc(sizeof(double) * md);
        std::complex<double> *spectrum_d = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * md_spectrum);

        fftw_plan_with_nthreads(1);
        fftw_plan plan_d = fftw_plan_dft_r2c_1d(md, (double *) data_d, (fftw_complex *) spectrum_d, FFTW_ESTIMATE);

        double scale_d = 1.0 / md;
        size_t mdh = md / 2;
        for (size_t s = 0; s < ms; ++s) {
            for (size_t d = 0; d < md; ++d) {
                data_d[d] = values(s, d);
            }

            fftw_execute(plan_d);

            for (size_t d = 0; d < md_spectrum; ++d) {
                coeffs(s, mdh + d) = spectrum_d[d] * scale_d;
            }
            for (size_t d = 1; d < md_spectrum; ++d) {
                coeffs(s, mdh - d) = std::conj(spectrum_d[d]) * scale_d;
            }
        }

        fftw_destroy_plan(plan_d);
        fftw_free(spectrum_d);
        fftw_free(data_d);

        /* Second, transform the resulting phi_s coefficient slices (columns) */
        std::complex<double> *data_s     = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * ms),
                             *spectrum_s = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * ms);

        fftw_plan_with_nthreads(1);
        fftw_plan plan_s = fftw_plan_dft_1d(ms, (fftw_complex *) data_s, (fftw_complex *) spectrum_s, FFTW_FORWARD, FFTW_ESTIMATE);

        double scale_s = 1.0 / ms;
        size_t msh = ms / 2;
        for (size_t d = 0; d < md; ++d) {
            for (size_t s = 0; s < ms; ++s) {
                data_s[s] = coeffs(s, d);
            }

            fftw_execute(plan_s);

            for (size_t s = 0; s <= msh; ++s) {
                coeffs(msh + s, d) = spectrum_s[s] * scale_s;
            }
            for (size_t s = 0; s < msh; ++s) {
                coeffs(s, d) = spectrum_s[msh + 1 + s] * scale_s;
            }
        }

        fftw_destroy_plan(plan_s);
        fftw_free(spectrum_s);
        fftw_free(data_s);

        return coeffs.real();
    #endif
}

NAMESPACE_END(mitsuba)