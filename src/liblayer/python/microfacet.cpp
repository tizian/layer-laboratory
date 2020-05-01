#include <mitsuba/layer/microfacet.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(microfacet) {
    m.def("smith_G1", &smith_G1, D(smith_G1),
          "v"_a, "m"_a, "alpha_u"_a, "alpha_v"_a);

    m.def("project_roughness", &project_roughness, D(project_roughness),
          "v"_a, "alpha_u"_a, "alpha_v"_a);

    m.def("microfacet", py::vectorize(&microfacet), D(microfacet),
          "mu_o"_a, "mu_i"_a, "phi_s"_a, "phi_d"_a, "alpha_u"_a, "alpha_v"_a, "eta"_a, "isotropic_g"_a = false);
    m.def("microfacet_exp", py::vectorize(&microfacet_exp), D(microfacet_exp),
          "mu_o"_a, "mu_i"_a, "phi_s"_a, "phi_d"_a, "alpha_u"_a, "alpha_v"_a, "eta"_a);
    m.def("microfacet_fresnel", py::vectorize(&microfacet_fresnel), D(microfacet_fresnel),
          "mu_o"_a, "mu_i"_a, "phi_s"_a, "phi_d"_a, "alpha_u"_a, "alpha_v"_a, "eta"_a, "fresnel_only"_a=false);
    m.def("microfacet_G", py::vectorize(&microfacet_G), D(microfacet_G),
          "mu_o"_a, "mu_i"_a, "phi_s"_a, "phi_d"_a, "alpha_u"_a, "alpha_v"_a, "eta"_a, "isotropic_g"_a = false);

    m.def("exp_cos_fourier_series", &exp_cos_fourier_series, D(exp_cos_fourier_series),
          "A"_a, "B"_a, "C"_a, "relerr"_a);

    m.def("fresnel_fourier_series", &fresnel_fourier_series, D(fresnel_fourier_series),
          "mu_o"_a, "mu_i"_a, "alpha_u"_a, "alpha_v"_a, "eta"_a, "m"_a, "phi_max"_a,
          "svd_regularization"_a=false, "fresnel_only"_a=false);

    m.def("smith_G1_fourier_series", &smith_G1_fourier_series, D(smith_G1_fourier_series),
          "mu"_a, "alpha_u"_a, "alpha_v"_a, "order"_a, "n_samples"_a = 30);

    m.def("microfacet_fourier_series",
        [](double mu_o, double mu_i, double alpha_u, double alpha_v, std::complex<double> eta, int ms, int md, double relerr, int component, int n_samples_phi_s) {
            MatrixX result;
            microfacet_fourier_series(mu_o, mu_i, alpha_u, alpha_v, eta, ms, md, relerr, result, component, n_samples_phi_s);
            return result;
        }, D(microfacet_fourier_series),
        "mu_o"_a, "mu_i"_a, "alpha_u"_a, "alpha_v"_a, "eta"_a,
        "ms"_a, "md"_a, "relerr"_a, "component"_a = 0, "n_samples_phi_s"_a = 30);

    m.def("microfacet_reflection_exp_coeffs",
        [](double mu_o, double mu_i, double alpha_u, double alpha_v, double phi_s) {
            VectorX c;
            microfacet_reflection_exp_coeffs(mu_o, mu_i, alpha_u, alpha_v, phi_s, c);
            return c;
        }, D(microfacet_reflection_exp_coeffs),
        "mu_o"_a, "mu_i"_a, "alpha_u"_a, "alpha_v"_a,
        "phi_s"_a);
    m.def("microfacet_refraction_exp_coeffs",
        [](double mu_o, double mu_i, double alpha_u, double alpha_v, double phi_s, double eta) {
            VectorX c;
            microfacet_refraction_exp_coeffs(mu_o, mu_i, alpha_u, alpha_v, phi_s, eta, c);
            return c;
        }, D(microfacet_refraction_exp_coeffs),
        "mu_o"_a, "mu_i"_a, "alpha_u"_a, "alpha_v"_a,
        "phi_s"_a, "eta"_a);
    m.def("microfacet_inside_lowfreq_interval",
        py::vectorize(&microfacet_inside_lowfreq_interval), D(microfacet_inside_lowfreq_interval),
        "mu_o"_a, "mu_i"_a, "phi_s"_a,
        "alpha_u"_a, "alpha_v"_a, "eta"_a,
        "phi_d"_a, "relerr"_a = 1e-3);
}
