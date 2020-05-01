#include <mitsuba/layer/layer.h>
#include <mitsuba/layer/sparse.h>
#include <mitsuba/layer/microfacet.h>
#include <mitsuba/layer/phase.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/spline.h>
#include <tbb/tbb.h>

NAMESPACE_BEGIN(mitsuba)

// #define TIME_FRESNEL

namespace {
    template <typename VectorType> MatrixS extended_sparse_diagonal(const VectorType &vec, MatrixS::Index ml) {
        MatrixS::Index n = vec.size();
        MatrixS result(n * ml, n * ml);

        for (MatrixS::Index l = 0; l < ml; ++l) {
            for (MatrixS::Index i = 0; i < vec.size(); ++i) {
                result.insert(l*n + i, l*n + i) = vec[i];
            }
        }
        result.makeCompressed();
        return result;
    }

    void scale_columns(Layer &layer, const VectorX &d) {
        if ((size_t) d.size() != layer.resolution())
            Throw("scale_columns(): size mismatch!");

        size_t ml = layer.fourier_orders().first + layer.fourier_orders().second - 1;

        MatrixS scale = extended_sparse_diagonal(d.head(d.size()/2), ml);
        layer.reflection_top() = mmul(layer.reflection_top(), scale);
        layer.transmission_bottom_top() = mmul(layer.transmission_bottom_top(), scale);

        scale = extended_sparse_diagonal(d.tail(d.size()/2), ml);
        layer.reflection_bottom() = mmul(layer.reflection_bottom(), scale);
        layer.transmission_top_bottom() = mmul(layer.transmission_top_bottom(), scale);
    }

    inline void apply_surface_integration_weights(Layer &layer) {
        scale_columns(layer, layer.weights().cwiseProduct(layer.nodes().cwiseAbs()) * 2 * math::Pi<double>);
    }

    inline void remove_surface_integration_weights(Layer &layer) {
        VectorX ones(layer.nodes().size());
        ones.setOnes();
        scale_columns(layer, ones.cwiseQuotient(layer.weights().cwiseProduct(layer.nodes().cwiseAbs()) * 2 * math::Pi<double>));
    }

    inline void apply_medium_integration_weights(Layer &layer) {
        scale_columns(layer, layer.weights() * 2 * math::Pi<double>);
    }
};

Layer::Layer(const VectorX &nodes, const VectorX &weights, size_t fourier_orders_s, size_t fourier_orders_d)
    : m_nodes(nodes),
      m_weights(weights),
      m_fourier_orders_s(fourier_orders_s),
      m_fourier_orders_d(fourier_orders_d) {
    if (nodes.size() < 2)
        Throw("Need at least 2 integration nodes!");
    else if (nodes.size() % 2 == 1)
        Throw("The number of integration nodes must be even!");
    for (int i = 0; i < nodes.size(); ++i)
        if (nodes[i] == 0)
            Throw("The set of integrations includes mu=0 -- this is not allowed.");
    if (fourier_orders_s % 2 == 0 || fourier_orders_d % 2 == 0)
        Throw("The number of Fourier orders must be odd!");

    if (nodes[0] < nodes[1]) {
        size_t n = (size_t) nodes.size();
        /* Order integration weights so that they are usable for adding-doubling */
        m_weights.head(n/2).reverseInPlace();
        m_nodes.head(n/2).reverseInPlace();
    }

    size_t matrix_size = (m_nodes.size() / 2) * (fourier_orders_d + fourier_orders_s - 1);
    m_reflection_top.resize(matrix_size, matrix_size);
    m_reflection_bottom.resize(matrix_size, matrix_size);
    m_transmission_top_bottom.resize(matrix_size, matrix_size);
    m_transmission_bottom_top.resize(matrix_size, matrix_size);
}

void Layer::clear() {
    m_reflection_top.setZero();
    m_reflection_bottom.setZero();
    m_transmission_top_bottom.setZero();
    m_transmission_bottom_top.setZero();
}

void Layer::clear_backside() {
    m_reflection_bottom.setZero();
}

void Layer::set_empty() {
    m_reflection_top.setZero();
    m_reflection_bottom.setZero();
    m_transmission_top_bottom.setIdentity();
    m_transmission_bottom_top.setIdentity();
}

void Layer::set_diffuse(double albedo) {
    std::vector<Quintet> quintets;
    quintets.reserve(resolution() * resolution() / 2);

    size_t n = resolution(), h = n/2;
    for (size_t i = 0; i < n; ++i) {
        for (size_t o = 0; o < n; ++o) {
            if ((i < h && o >= h) || (o < h && i >= h))
                quintets.emplace_back(0, 0, o, i, albedo * math::InvPi<double>);
        }
    }

    set_quintets(quintets);
    apply_surface_integration_weights(*this);
}

void Layer::set_microfacet(std::complex<double> eta, double alpha_u, double alpha_v,
                           size_t target_fourier_orders_s, size_t target_fourier_orders_d,
                           int component, int n_samples_phi_s, bool svd_reg) {
    size_t n = resolution();
    std::vector<Quintet> quintets;
    tbb::spin_mutex mutex;

    target_fourier_orders_s = std::max(target_fourier_orders_s, fourier_orders().first);
    target_fourier_orders_d = std::max(target_fourier_orders_d, fourier_orders().second);

    size_t total_fourier_orders = target_fourier_orders_s * target_fourier_orders_d;

    while (true) {
        double tmp = microfacet_timer;
        if (microfacet_timer.compare_exchange_strong(tmp, 0.0))
            break;
    }

#ifdef TIME_FRESNEL
    auto start = std::chrono::high_resolution_clock::now();
#endif

#if 1
    // Multithreaded
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, resolution()),
        [&](const tbb::blocked_range<size_t> &range) {
            std::vector<Quintet> local_quintets;
            local_quintets.reserve(total_fourier_orders * resolution());

            MatrixX result;
            for (size_t i = range.begin(); i < range.end(); ++i) {
                for (size_t o = 0; o < n; ++o) {

                    /* Sign flip due to different convention (depth values
                     * increase opposite to the normal direction) */
                    microfacet_fourier_series(-m_nodes[o], -m_nodes[i], alpha_u, alpha_v, eta,
                                              (int) target_fourier_orders_s, (int) target_fourier_orders_d,
                                              ERROR_GOAL, result, component, n_samples_phi_s, svd_reg);

                    int ms = result.rows();
                    int md = result.cols();

                    int msh = ms / 2;
                    int mdh = md / 2;

                    for (int d = -mdh; d <= mdh; ++d) {
                        for (int s = -msh; s <= msh; ++s) {
                            double value = result(s + msh, d + mdh);
                            if (std::abs(value) > DROP_THRESHOLD)
                                local_quintets.emplace_back(s, d, o, i, value);
                        }
                    }
                }
            }
            tbb::spin_mutex::scoped_lock lock(mutex);
            quintets.insert(quintets.end(), local_quintets.begin(), local_quintets.end());
        }
    );
#else
    // Singlethrading for debugging
    std::vector<Quintet> local_quintets;
    local_quintets.reserve(total_fourier_orders * resolution());
    MatrixX result;
    for (size_t i = 0; i < n; ++i) {
        for (size_t o = 0; o < n; ++o) {

            /* Sign flip due to different convention (depth values
             * increase opposite to the normal direction) */
            microfacet_fourier_series(-m_nodes[o], -m_nodes[i], alpha_u, alpha_v, eta,
                                      target_fourier_orders_s, target_fourier_orders_d,
                                      ERROR_GOAL, result, component, n_samples_phi_s, svd_reg);

            int ms = result.rows();
            int md = result.cols();

            int msh = ms / 2;
            int mdh = md / 2;

            for (int d = -mdh; d <= mdh; ++d) {
                for (int s = -msh; s <= msh; ++s) {
                    double value = result(s + msh, d + mdh);
                    if (std::abs(value) > DROP_THRESHOLD)
                        local_quintets.emplace_back(s, d, o, i, value);
                }
            }
        }
    }
    quintets.insert(quintets.end(), local_quintets.begin(), local_quintets.end());
#endif

    set_quintets(quintets);

#ifdef TIME_FRESNEL
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = (end-start);
    std::cout << "Layer creation took " << diff.count() << "s" << std::endl;
    std::cout << "  Fresnel component was " << microfacet_timer << "s" << std::endl;
#endif

    apply_surface_integration_weights(*this);
}

void Layer::set_isotropic(double albedo) {
    std::vector<Quintet> quintets;
    quintets.reserve(resolution() * resolution());

    size_t n = resolution();
    for (size_t i = 0; i < n; ++i) {
        for (size_t o = 0; o < n; ++o) {
            quintets.emplace_back(0, 0, o, i, albedo * math::InvFourPi<double>);
        }
    }

    set_quintets(quintets);
    apply_medium_integration_weights(*this);
}

void Layer::set_henyey_greenstein(double albedo, double g) {
    std::vector<Quintet> quintets;
    tbb::spin_mutex mutex;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, resolution()),
        [&](const tbb::blocked_range<size_t> &range) {
            std::vector<Quintet> local_quintets;
            local_quintets.reserve(fourier_orders().second * resolution());

            VectorX result;
            for (size_t i = range.begin(); i < range.end(); ++i) {
                for (size_t o = 0; o <= i; ++o) {
                    henyey_greenstein_fourier_series(m_nodes[o], m_nodes[i], g,
                                                     (int) fourier_orders().second, ERROR_GOAL, result);

                    int mdh = result.size() / 2;
                    for (int d = -mdh; d <= mdh; ++d) {
                        local_quintets.emplace_back(0, d, o, i, result[d + mdh] * albedo);
                        if (i != o) {
                            local_quintets.emplace_back(0, d, i, o, result[d + mdh] * albedo);
                        }
                    }
                }
                tbb::spin_mutex::scoped_lock lock(mutex);
                quintets.insert(quintets.end(), local_quintets.begin(), local_quintets.end());
            }
        }
    );

    set_quintets(quintets);
    apply_medium_integration_weights(*this);
}

void Layer::set_von_mises_fisher(double albedo, double kappa) {
    std::vector<Quintet> quintets;
    tbb::spin_mutex mutex;

    double scale;
    if (kappa == 0) {
        scale = albedo / (4 * math::Pi<double>);
    } else {
        scale = albedo * kappa / (4 * math::Pi<double> * std::sinh(kappa));
    }

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, resolution()),
        [&](const tbb::blocked_range<size_t> &range) {
            std::vector<Quintet> local_quintets;
            local_quintets.reserve(fourier_orders().second * resolution());

            VectorX result;
            for (size_t i = range.begin(); i < range.end(); ++i) {
                for (size_t o = 0; o <= i; ++o) {
                    von_mises_fisher_fourier_series(m_nodes[o], m_nodes[i], kappa, ERROR_GOAL, result);

                    int mdh = result.size() / 2;
                    for (int d = -mdh; d <= mdh; ++d) {
                        local_quintets.emplace_back(0, d, o, i, result[d + mdh] * scale);
                        if (i != o) {
                            local_quintets.emplace_back(0, d, i, o, result[d + mdh] * scale);
                        }
                    }
                }
                tbb::spin_mutex::scoped_lock lock(mutex);
                quintets.insert(quintets.end(), local_quintets.begin(), local_quintets.end());
            }
        }
    );

    set_quintets(quintets);
    apply_medium_integration_weights(*this);
}

void Layer::set_fourier_coeffs(std::vector<MatrixX> coeffs) {
    std::vector<Quintet> quintets;
    tbb::spin_mutex mutex;

    size_t n = resolution();

    if (coeffs.size() != n*n)
        Throw("Layer::set_fourier_coeffs(): incompatible size of elevation angles!");

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n),
        [&](const tbb::blocked_range<size_t> &range) {
            std::vector<Quintet> local_quintets;
            local_quintets.reserve(fourier_orders().first * fourier_orders().second * resolution());

            for (size_t i = range.begin(); i < range.end(); ++i) {
                for (size_t o = 0; o < n; ++o) {
                    const MatrixX &coeffs_io = coeffs[i*n + o];

                    int ms = coeffs_io.rows(),
                        md = coeffs_io.cols();

                    if (coeffs_io.rows() % 2 == 0 || coeffs_io.cols() % 2 == 0)
                        Throw("Layer::set_fourier_coeffs(): require odd dimensions Fourier coefficient matrix!");

                    int msh = ms / 2,
                        mdh = md / 2;

                    for (int d = -mdh; d <= mdh; ++d) {
                        for (int s = -msh; s <= msh; ++s) {
                            double value = coeffs_io(s + msh, d + mdh);
                            if (std::abs(value) > 1e-10)
                                local_quintets.emplace_back(s, d, o, i, value);
                        }
                    }
                }
                tbb::spin_mutex::scoped_lock lock(mutex);
                quintets.insert(quintets.end(), local_quintets.begin(), local_quintets.end());
            }
        }
    );

    set_quintets(quintets);
    apply_surface_integration_weights(*this);
}

void Layer::add(const Layer &l1, const Layer &l2, Layer &output, bool homogeneous, double eps) {
#if !defined(HAVE_UMFPACK)
    (void) l1; (void) l2; (void) output; (void) homogeneous; (void) eps;
    Throw("Layer::add: You need to recompile with support for UMFPACK!");
#else
    if (output.resolution() != l1.resolution() ||
        output.resolution() != l2.resolution() ||
        output.fourier_orders().second != l1.fourier_orders().second ||
        output.fourier_orders().second != l2.fourier_orders().second) {
        Throw("Layer::add(): incompatible sizes!");
    }

    if (homogeneous) {
        const MatrixS &Rt_1 = l1.reflection_top();
        const MatrixS &Rb_1 = l1.reflection_bottom();
        const MatrixS &Tbt_1 = l1.transmission_bottom_top();
        const MatrixS &Ttb_1 = l1.transmission_top_bottom();
        const MatrixS &Rt_2 = l2.reflection_top();
        const MatrixS &Ttb_2 = l2.transmission_top_bottom();

        MatrixS I(Rt_1.rows(), Rt_1.cols());
        I.setIdentity();

        MatrixS RR = I - mmul(Rb_1, Rt_2, 0.0);
        RR.prune(1.0, eps);

        MatrixS tmp = UMFPack(RR).solve_sparse(Ttb_1, eps);

        MatrixS Rt  = Rt_1 + mmul(Tbt_1, mmul(Rt_2, tmp, eps), eps);
        MatrixS Ttb = mmul(Ttb_2, tmp);

        output.reflection_top() = Rt;
        output.reflection_bottom() = Rt;
        output.transmission_top_bottom() = Ttb;
        output.transmission_bottom_top() = Ttb;
    } else {
        const MatrixS &Rt_1 = l1.reflection_top();
        const MatrixS &Rb_1 = l1.reflection_bottom();
        const MatrixS &Tbt_1 = l1.transmission_bottom_top();
        const MatrixS &Ttb_1 = l1.transmission_top_bottom();
        const MatrixS &Rt_2 = l2.reflection_top();
        const MatrixS &Rb_2 = l2.reflection_bottom();
        const MatrixS &Tbt_2 = l2.transmission_bottom_top();
        const MatrixS &Ttb_2 = l2.transmission_top_bottom();

        MatrixS I(Rt_1.rows(), Rt_1.cols());
        I.setIdentity();

        MatrixS RR1 = I - mmul(Rb_1, Rt_2, 0.0);
        MatrixS RR2 = I - mmul(Rt_2, Rb_1, 0.0);
        RR1.prune(1.0, eps);
        RR2.prune(1.0, eps);

        MatrixS tmp0 = UMFPack(RR1).solve_sparse(Ttb_1, eps);
        MatrixS tmp1 = UMFPack(RR2).solve_sparse(Tbt_2, eps);

        MatrixS Rt = Rt_1 + mmul(Tbt_1, mmul(Rt_2, tmp0, eps), eps);
        MatrixS Rb = Rb_2 + mmul(Ttb_2, mmul(Rb_1, tmp1, eps), eps);

        MatrixS Ttb = mmul(Ttb_2, tmp0, eps);
        MatrixS Tbt = mmul(Tbt_1, tmp1, eps);

        output.reflection_top() = Rt;
        output.reflection_bottom() = Rb;
        output.transmission_top_bottom() = Ttb;
        output.transmission_bottom_top() = Tbt;
    }
#endif
}

void Layer::subtract(const Layer &ladd, const Layer &l1, Layer &l2, double eps) {
#if !defined(HAVE_UMFPACK)
    (void) ladd; (void) l1; (void) l2; (void) eps;
    (void) remove_surface_integration_weights;
    Throw("Layer::subtract: You need to recompile with support for UMFPACK!");
#else
    const MatrixS &Rt_add = ladd.reflection_top();
    const MatrixS &Rt_1 = l1.reflection_top();
    const MatrixS &Rb_1 = l1.reflection_bottom();
    const MatrixS &Tbt_1 = l1.transmission_bottom_top();
    const MatrixS &Ttb_1 = l1.transmission_top_bottom();

    double drop_threshold = 1e-5;

    MatrixS I(Rt_1.rows(), Rt_1.cols());
    I.setIdentity();

    MatrixS W = eps * I;

    /* Invert transmission from top layer */
    MatrixS Ttb_1_t = Ttb_1.transpose();
    MatrixS Tbt_1_t = Tbt_1.transpose();

    MatrixS Ttb_1_i = UMFPack(mmul(Ttb_1_t, Ttb_1) + W).solve_sparse(Ttb_1_t, drop_threshold);
    MatrixS Tbt_1_i = UMFPack(mmul(Tbt_1_t, Tbt_1) + W).solve_sparse(Tbt_1_t, drop_threshold);

    /* Compute top reflection of new bottom layer */
    MatrixS X = mmul(Tbt_1_i, mmul((Rt_add - Rt_1), Ttb_1_i));

    MatrixS Y = UMFPack(I + mmul(Rb_1, X)).solve_sparse(I, drop_threshold);
    MatrixS Z = mmul(X, Y);

    l2.reflection_top() = Z;
    remove_surface_integration_weights(l2);

    Z = l2.reflection_top();

    /* The rows/columns close to normal (mu ~= 1) or grazing (mu ~= 0) incidence are especially tricky */
    int ms, md;
    std::tie(ms, md) = l1.fourier_orders();
    int n = l1.resolution();
    int size = n / 2;
    int blocks = ms + md - 1;

    /* At normal incidence, we extrapolate the last few columns and then copy them to the rows */
    int nn = 4;

    for (int b = 0; b < blocks; ++b) {
        int start_col = b * size;

        for (int j = 0; j < nn; ++j) {
            Z.col(start_col + size - nn + j) = Z.col(start_col + size - nn - 1);
        }
    }

    MatrixS Zt = Z.transpose();
    for (int b = 0; b < blocks; ++b) {
        int start_col = b * size;

        for (int j = 0; j < nn; ++j) {
            Zt.col(start_col + size - nn + j) = Zt.col(start_col + size - nn - 1);
        }
    }

    Z = Zt.transpose();

    /* For grazing incidence, we copy the first few rows to the columns */
    nn = 2;
    for (int b = 0; b < blocks; ++b) {
        int start_col = b * size;

        for (int j = 0; j < nn; ++j) {
            Z.col(start_col + j) = Zt.col(start_col + j);
        }
    }

    Z.prune(1.0, drop_threshold);

    l2.reflection_top() = Z;
    apply_surface_integration_weights(l2);
#endif
}

void Layer::expand(double target_tau) {
    /* Heuristic for choosing the initial width of a layer based on
       "Discrete Space Theory of Radiative Transfer" by Grant and Hunt
       Proc. R. Soc. London 1969 */
    double tau = std::min(m_nodes.cwiseAbs().minCoeff() * 2, std::pow(2.0, -15.0));

    size_t doublings = (size_t) std::ceil(std::log(target_tau / tau) / std::log(2.0));
    tau = target_tau * std::pow(2.0, -(double) doublings);

    size_t ml = fourier_orders().first + fourier_orders().second - 1;
    size_t n = resolution() / 2;

    MatrixS I(ml * n, ml * n);
    I.setIdentity();

    MatrixS row_scale = extended_sparse_diagonal(m_nodes.tail(n).cwiseInverse() * tau, ml);

    MatrixS Rt = row_scale * m_reflection_top;
    MatrixS Ttb = I + row_scale * (m_transmission_top_bottom - I);

    m_reflection_top = Rt;
    m_reflection_bottom = Rt;
    m_transmission_top_bottom = Ttb;
    m_transmission_bottom_top = Ttb;

    for (size_t i = 0; i < (size_t) doublings; ++i) {
        add(*this, *this, *this, true);
    }
}

double Layer::scatter_coeff(int s, int d, size_t o, size_t i) const {
    size_t n = resolution() / 2;

    int ms = (int) fourier_orders().first,
        md = (int) fourier_orders().second,
        ml = ms + md - 1;
    int mlh = ml / 2;

    int block_i = d + s + mlh,
        block_j = d - s + mlh;

    if (o < n && i < n)
        return m_transmission_bottom_top.coeff(block_i*n + o, block_j*n + i);
    else if (o >= n && i >= n)
        return m_transmission_top_bottom.coeff(block_i*n + o-n, block_j*n + i-n);
    else if (o < n && i >= n)
        return m_reflection_top.coeff(block_i*n + o, block_j*n + i-n);
    else if (o >= n && i < n)
        return m_reflection_bottom.coeff(block_i*n + o-n, block_j*n + i);
    else
        Log(Error, "Layer::coeff(): out of bounds!");
        return -1.0;
}

void Layer::reverse() {
    m_reflection_top.swap(m_reflection_bottom);
    m_transmission_top_bottom.swap(m_transmission_bottom_top);
}

double Layer::eval(double mu_o, double mu_i, double phi_s, double phi_d, bool clamp) const {
    int n = m_nodes.size(), h = n / 2;

    double knot_weights_o[4], knot_weights_i[4];
    ssize_t knot_offset_o, knot_offset_i;

    if (mu_o < 0 || (mu_o == 0 && mu_i > 0)) {
        auto spline_weights_o = spline::eval_spline_weights<true>(m_nodes.data() + h, h, -mu_o, knot_weights_o);
        knot_offset_o = spline_weights_o.second;
    } else {
        auto spline_weights_o = spline::eval_spline_weights<true>(m_nodes.data() + h, h, mu_o, knot_weights_o);
        knot_offset_o = spline_weights_o.second + h;
    }

    if (mu_i < 0 || (mu_i == 0 && mu_o > 0)) {
        auto spline_weights_i = spline::eval_spline_weights<true>(m_nodes.data() + h, h, -mu_i, knot_weights_i);
        knot_offset_i = spline_weights_i.second;
    } else {
        auto spline_weights_i = spline::eval_spline_weights<true>(m_nodes.data() + h, h, mu_i, knot_weights_i);
        knot_offset_i = spline_weights_i.second + h;
    }

    int msh = (int) fourier_orders().first / 2,
        mdh = (int) fourier_orders().second / 2;

    std::complex<double> exp_s = std::exp(1i * phi_s);
    std::complex<double> exp_d = std::exp(1i * phi_d);

    std::complex<double> start_exp_s = std::exp(-1i * (double) msh * phi_s);
    std::complex<double> cur_exp_d = std::exp(-1i * (double) mdh * phi_d);

    double result = 0.0;
    for (int d = -mdh; d <= mdh; ++d) {
        std::complex<double> cur_exp = start_exp_s * cur_exp_d;
        for (int s = -msh; s <= msh; ++s) {

            std::complex<double> coeff(0);
            for (int o = 0; o < 4; ++o) {
                for (int i = 0; i < 4; ++i) {
                    double weight = knot_weights_o[o] * knot_weights_i[i];
                    if (weight == 0)
                        continue;

                    weight /= 2 * math::Pi<double> * std::abs(m_nodes[knot_offset_i + i]) * m_weights[knot_offset_i + i];
                    coeff += weight * scatter_coeff(s, d, knot_offset_o + o, knot_offset_i + i);
                }
            }
            result += std::real(coeff * cur_exp);
            cur_exp *= exp_s;
        }
        cur_exp_d *= exp_d;
    }
    return clamp ? std::max(0.0, result) : result;
}

void Layer::find_truncation(MatrixS::Index o, MatrixS::Index i, double error,
                            size_t &ms_trunc, size_t &md_trunc) const {
    int n = m_nodes.size();
    int h = n / 2;
    i = (MatrixS::Index) (i < h ? (h-i-1) : i);
    o = (MatrixS::Index) (o < h ? (h-o-1) : o);

    ms_trunc = 0, md_trunc = 0;

    int ms = (int) fourier_orders().first;
    int md = (int) fourier_orders().second;
    int msh = ms / 2, mdh = md / 2;

#if 1
    // Cut out rectangle, s.t. all entries are > eps * error
    // This can be done by doing two independent passes through the data
    double ref = std::abs(scatter_coeff(0, 0, o, i)) * error;

    // Check in phi_s dimension
    for (int s = 0; s <= msh; ++s) {
        bool keep = false;
        for (int d = -mdh; d <= mdh; ++d) {
            double value = std::abs(scatter_coeff(s, d, o, i));
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
            double value = std::abs(scatter_coeff(s, d, o, i));
            if (value > ref) keep = true;
        }
        if (keep) {
            md_trunc = 2*d+1;
        } else {
            break;
        }
    }
#else
    // Cut out rectangle that minimizes full truncation error
    // (sum of elements outside)
    // Computed using summed area tables

    double ref = std::abs(scatter_coeff(0, 0, o, i)) * error;

    MatrixX val(ms, md);
    val.setZero();

    // Build summed area table sigma (recursive formulation)
    MatrixX sigma(ms, md);
    sigma.setZero();
    for (int d = -mdh; d <= mdh; ++d) {
        for (int s = -msh; s <= msh; ++s) {
            int os = s + msh;
            int od = d + mdh;

            double value = std::abs(scatter_coeff(s, d, o, i));
            if (value < 1e-6) value = 0;

            val(os, od) = value;
            if (os - 1 >= 0)
                value += sigma(os - 1, od);
            if (od - 1 >= 0)
                value += sigma(os, od - 1);
            if (os - 1 >= 0 && od - 1 >= 0)
                value -= sigma(os - 1, od - 1);

            sigma(os, od) = value;
        }
    }

    // For each possible size, consider truncation error, keep rectangle
    // with minimal number of entries
    ms_trunc = 0;
    md_trunc = 0;

    int min_coeffs = std::numeric_limits<int>::max();
    double total = sigma(ms-1, md-1);

    if (ref != 0) {
        ms_trunc = ms;
        md_trunc = md;

        for (int d = 0; d <= mdh; ++d) {
            for (int s = 0; s <= msh; ++s) {
                int os = s + msh;
                int od = d + mdh;

                int coeffs = (2*s+1)*(2*d+1);

                double value = total;
                value -= sigma(os, od);
                if (-s+msh-1 >= 0)
                    value += sigma(-s+msh-1, od);
                if (-d+mdh-1 >= 0)
                    value += sigma(os, -d+mdh-1);
                if (-s+msh-1 >= 0 && -d+mdh-1 >= 0)
                    value -= sigma(-s+msh-1, -d+mdh-1);


                if (value < ref && coeffs < min_coeffs) {
                    ms_trunc = 2*s+1;
                    md_trunc = 2*d+1;
                    min_coeffs = coeffs;
                }
            }
        }
    }
#endif
}

void Layer::fourier_slice(int o, int i, MatrixX &coeffs) const {
    int n = m_nodes.size();
    int h = n / 2;
    i = (MatrixS::Index) (i < h ? (h-i-1) : i);
    o = (MatrixS::Index) (o < h ? (h-o-1) : o);

    int msh = (int) fourier_orders().first / 2,
        mdh = (int) fourier_orders().second / 2;

    coeffs.resize(2*msh+1, 2*mdh+1);
    coeffs.setZero();

    for (int d = -mdh; d <= mdh; ++d) {
        int od = d + mdh;
        for (int s = -msh; s <= msh; ++s) {
            int os = s + msh;
            coeffs(os, od) = scatter_coeff(s, d, o, i);
        }
    }
}

void Layer::fourier_slice_interpolated(double mu_o, double mu_i, MatrixX &coeffs) const {
    int n = m_nodes.size(), h = n / 2;

    double knot_weights_o[4], knot_weights_i[4];
    ssize_t knot_offset_o, knot_offset_i;

    if (mu_o < 0 || (mu_o == 0 && mu_i > 0)) {
        auto spline_weights_o = spline::eval_spline_weights<true>(m_nodes.data() + h, h, -mu_o, knot_weights_o);
        knot_offset_o = spline_weights_o.second;
    } else {
        auto spline_weights_o = spline::eval_spline_weights<true>(m_nodes.data() + h, h, mu_o, knot_weights_o);
        knot_offset_o = spline_weights_o.second + h;
    }

    if (mu_i < 0 || (mu_i == 0 && mu_o > 0)) {
        auto spline_weights_i = spline::eval_spline_weights<true>(m_nodes.data() + h, h, -mu_i, knot_weights_i);
        knot_offset_i = spline_weights_i.second;
    } else {
        auto spline_weights_i = spline::eval_spline_weights<true>(m_nodes.data() + h, h, mu_i, knot_weights_i);
        knot_offset_i = spline_weights_i.second + h;
    }

    int msh = (int) fourier_orders().first / 2,
        mdh = (int) fourier_orders().second / 2;

    coeffs.resize(2*msh+1, 2*mdh+1);
    coeffs.setZero();

    for (int d = -mdh; d <= mdh; ++d) {
        int od = d + mdh;
        for (int s = -msh; s <= msh; ++s) {
            int os = s + msh;

            for (int o = 0; o < 4; ++o) {
                for (int i = 0; i < 4; ++i) {
                    double weight = knot_weights_o[o] * knot_weights_i[i];
                    if (weight == 0)
                        continue;

                    weight /= 2 * math::Pi<double> * std::abs(m_nodes[knot_offset_i + i]) * m_weights[knot_offset_i + i];
                    coeffs(os, od) = weight * scatter_coeff(s, d, knot_offset_o + o, knot_offset_i + i);
                }
            }
        }
    }
}

std::string Layer::to_string() const {
    size_t nz_max = resolution() * resolution() * fourier_orders().first * fourier_orders().second;
    size_t nz = m_reflection_top.nonZeros() +
                m_reflection_bottom.nonZeros() +
                m_transmission_top_bottom.nonZeros() +
                m_transmission_bottom_top.nonZeros();

    std::ostringstream oss;
    oss.precision(2);
    oss << "Layer[" << std::endl
        << "  resolution = " << resolution() << "x" << resolution() << std::endl
        << "  fourier_orders = " << fourier_orders().first << "x"
                                 << fourier_orders().second << "," << std::endl
        << "  non_zeros=" << nz << "/" << nz_max << " (" << ((float) nz / (float) nz_max * 100) << "%)" << std::endl
        << "]";
    return oss.str();
}

void Layer::set_quintets(const std::vector<Quintet> &quintets) {
    int ms = (int) fourier_orders().first,
        md = (int) fourier_orders().second,
        ml = ms + md - 1;
    int msh = ms / 2, mdh = md / 2, mlh = ml / 2;

    std::vector<Eigen::Triplet<double>> triplets_t,
                                        triplets_b,
                                        triplets_tb,
                                        triplets_bt;

    size_t approx_size = quintets.size() / (4*ms*md);
    triplets_t.reserve(approx_size);
    triplets_b.reserve(approx_size);
    triplets_tb.reserve(approx_size);
    triplets_bt.reserve(approx_size);

    size_t n = resolution() / 2;
    for (auto const &quintet: quintets) {
        typedef MatrixS::Index Index;

        // Only keep & store the Fourier coefficients smaller than fourier_orders()
        int os = msh + quintet.s;
        int od = mdh + quintet.d;
        if (os < 0 || os >= ms) continue;
        if (od < 0 || od >= md) continue;

        int block_i = quintet.d + quintet.s + mlh,
            block_j = quintet.d - quintet.s + mlh;

        // Note the reordering inside the blocks s.t. adding makes sense
        if (quintet.o < n && quintet.i < n)
            triplets_bt.emplace_back(Index(block_i*n + quintet.o), Index(block_j*n + quintet.i), quintet.value);
        else if (quintet.o >= n && quintet.i >= n)
            triplets_tb.emplace_back(Index(block_i*n + quintet.o-n), Index(block_j*n + quintet.i-n), quintet.value);
        else if (quintet.o < n && quintet.i >= n)
            triplets_t.emplace_back(Index(block_i*n + quintet.o), Index(block_j*n + quintet.i-n), quintet.value);
        else if (quintet.o >= n && quintet.i < n)
            triplets_b.emplace_back(Index(block_i*n + quintet.o-n), Index(block_j*n + quintet.i), quintet.value);
        else
            Throw("Layer::setFromQuintets(): internal error!");
    }

    m_reflection_top.setFromTriplets(triplets_t.begin(), triplets_t.end());
    m_reflection_bottom.setFromTriplets(triplets_b.begin(), triplets_b.end());
    m_transmission_top_bottom.setFromTriplets(triplets_tb.begin(), triplets_tb.end());
    m_transmission_bottom_top.setFromTriplets(triplets_bt.begin(), triplets_bt.end());
}

std::tuple<int, int, int> microfacet_parameter_heuristic(double alpha_u, double alpha_v,
                                                         std::complex<double> &eta) {
    /* Isotropic heuristic: need to be able to represent smaller roughness parameter */
    double alpha = std::min(alpha_u, alpha_v);
    alpha = std::min(alpha, 1.0);
    if (eta.real() < 1 && eta.imag() == 0)
        eta = std::complex<double>(1) / eta;

    static const double c[][9] = {
        /* IOR    A_n      B_n     C_n       D_n      A_m      B_m      C_m      D_m                                 */
        {  0.0, 35.275,  14.136,  29.287,  1.8765,   39.814,  88.992, -98.998,  39.261  },  /* Generic conductor     */
        {  1.1, 256.47, -73.180,  99.807,  37.383,  110.782,  57.576,  94.725,  14.001  },  /* Dielectric, eta = 1.1 */
        {  1.3, 100.264, 28.187,  64.425,  14.850,   45.809,  17.785, -7.8543,  12.892  },  /* Dielectric, eta = 1.3 */
        {  1.5, 74.176,  27.470,  42.454,  9.6437,   31.700,  44.896, -45.016,  19.643  },  /* Dielectric, eta = 1.5 */
        {  1.7, 80.098,  17.016,  50.656,  7.2798,   46.549,  58.592, -73.585,  25.473  },  /* Dielectric, eta = 1.7 */
    };

    int i0 = 0, i1 = 0;

    if (eta.imag() == 0) { /* Dielectric case */
        for (int i = 1; i < 4; ++i) {
            if (eta.real() >= c[i][0] && eta.real() <= c[i+1][0]) {
                if (std::abs(eta.real() - c[i][0]) < 0.05) {
                    i1 = i0 = i;
                } else if (std::abs(eta - c[i+1][0]) < 0.05) {
                    i0 = i1 = i+1;
                } else {
                    i0 = i; i1 = i+1;
                }
            }
        }

        if (!i0)
            Throw("Index of refraction is out of bounds (must be between 1.1 and 1.7)!");
    }

    double n0 = std::max(c[i0][1] + c[i0][2]*std::pow(std::log(alpha), 4.0)*alpha, c[i0][3]+c[i0][4]*std::pow(alpha, -1.2));
    double n1 = std::max(c[i1][1] + c[i1][2]*std::pow(std::log(alpha), 4.0)*alpha, c[i1][3]+c[i1][4]*std::pow(alpha, -1.2));
    double d0 = std::max(c[i0][5] + c[i0][6]*std::pow(std::log(alpha), 4.0)*alpha, c[i0][7]+c[i0][8]*std::pow(alpha, -1.2));
    double d1 = std::max(c[i1][5] + c[i1][6]*std::pow(std::log(alpha), 4.0)*alpha, c[i1][7]+c[i1][8]*std::pow(alpha, -1.2));

    int n_i = (int) std::ceil(std::max(n0, n1)),
        d_i = (int) std::ceil(std::max(d0, d1));

    if (n_i % 2 == 1)
        n_i += 1;

    d_i = 2*d_i - 1;

    /* Anisotropic heuristic: based on a linear fit along the ratio of roughness parameters */
    int s_i = 1;
    if (alpha_u != alpha_v) {
        double A =  5.16,
               B = -2.63;


        double ratio = alpha_u / alpha_v;
        if (ratio < 1) ratio = 1.0 / ratio;

        s_i = std::ceil(A*ratio + B);
        if (s_i % 2 == 0)
            s_i += 1;
    }

    return std::make_tuple(n_i, s_i, d_i);
}

std::tuple<int, int, int> henyey_greenstein_parameter_heuristic(double g) {
    g = std::abs(g);
    double d = 5.4 / (1.0 - g) - 1.3;
    double n = 8.6 / (1.0 - g) - 0.2;
    int n_i = (int) std::ceil(n),
        d_i = (int) std::ceil(d);

    if (n_i % 2 == 1)
        n_i += 1;

    d_i = 2*d_i - 1;

    return std::make_tuple(n_i, 1, d_i);
}

NAMESPACE_END(mitsuba)
