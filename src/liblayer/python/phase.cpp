#include <mitsuba/layer/phase.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(phase) {
    m.def("henyey_greenstein", py::vectorize(&henyey_greenstein),
          D(henyey_greenstein), "mu_o"_a, "mu_i"_a, "phi_d"_a, "g"_a);
    m.def("henyey_greenstein_fourier_series", [](double mu_o, double mu_i, double g, int md, double relerr) {
        VectorX result;
        henyey_greenstein_fourier_series(mu_o, mu_i, g, md, relerr, result);
        return result;
    }, D(henyey_greenstein_fourier_series), "mu_o"_a, "mu_i"_a, "g"_a, "md"_a, "relerr"_a);

    m.def("von_mises_fisher", py::vectorize(&von_mises_fisher),
          D(von_mises_fisher), "mu_o"_a, "mu_i"_a, "phi_d"_a, "kappa"_a);
    m.def("von_mises_fisher_fourier_series", [](double mu_o, double mu_i, double kappa, double relerr) {
        VectorX result;
        von_mises_fisher_fourier_series(mu_o, mu_i, kappa, relerr, result);
        return result;
    }, D(von_mises_fisher_fourier_series), "mu_o"_a, "mu_i"_a, "kappa"_a, "relerr"_a);
}
