#include <mitsuba/layer/fourier.h>
#include <mitsuba/python/python.h>
#include <mitsuba/core/math.h>

MTS_PY_EXPORT(fourier) {
    m.def("filon_integrate_exp", [](const py::array_t<std::complex<double>> &values, size_t orders, double a, double b) {
        size_t size = values.size();

        VectorXc coeffs(orders);
        coeffs.setZero();
        filon_integrate_exp(values.data(), size, coeffs.data(), orders, a, b);
        return coeffs;
    }, D(filon_integrate_exp), "values"_a, "orders"_a, "a"_a=0, "b"_a=2*math::Pi<double>);

    m.def("fftw_transform_c2c", [](const py::array_t<std::complex<double>> &values) {
        size_t size = values.size();

        VectorXc coeffs(size);
        coeffs.setZero();
        fftw_transform_c2c(values.data(), size, coeffs.data());
        return coeffs;
    }, D(fftw_transform_c2c), "values"_a);

    m.def("fftw_transform_r2c", [](const py::array_t<double> &values) {
        size_t size = values.size();

        VectorXc coeffs(size);
        coeffs.setZero();
        fftw_transform_r2c(values.data(), size, coeffs.data());
        return coeffs;
    }, D(fftw_transform_r2c), "values"_a);

    m.def("fftw_transform_2d", &fftw_transform_2d,
        D(fftw_transform_2d), "values"_a);

    m.def("eval_fourier_series_2d", [](const py::array_t<float> &coeffs_, const py::array_t<float> phi_s, const py::array_t<float> phi_d) {
        return py::vectorize([&](float phi_s, float phi_d) {
            auto coeffs = coeffs_.unchecked<2>();
            size_t ms = coeffs.shape(0),
                   md = coeffs.shape(1);
            return eval_fourier_series_2d(coeffs_.data(), ms, md, phi_s, phi_d); })(phi_s, phi_d);
        }, D(eval_fourier_series_2d), "coeffs"_a, "phi_s"_a, "phi_d"_a);
}
