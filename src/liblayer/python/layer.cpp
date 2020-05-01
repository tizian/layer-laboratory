#include <mitsuba/layer/layer.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(Layer) {
    py::class_<Layer>(m, "Layer", D(Layer))
        .def(py::init<const VectorX &, const VectorX &, size_t, size_t>(),
             "nodes"_a, "weights"_a, "fourier_orders_s"_a = 1, "fourier_orders_d"_a = 1)
        .def(py::init<const Layer &>())

        .def("reverse", &Layer::reverse, D(Layer, reverse))
        .def("clear", &Layer::clear, D(Layer, clear))
        .def("clear_backside", &Layer::clear_backside, D(Layer, clear_backside))

        .def("set_empty", &Layer::set_empty, D(Layer, set_empty))
        .def("set_diffuse", &Layer::set_diffuse, D(Layer, set_diffuse), "albedo"_a)
        .def("set_microfacet", &Layer::set_microfacet, D(Layer, set_microfacet),
            "eta"_a, "alpha_u"_a, "alpha_v"_a, "fourier_orders_s"_a = 1, "fourier_orders_d"_a = 1,
            "component"_a = 0, "n_samples_phi_s"_a = 30, "svd_reg"_a = false)
        .def("set_isotropic", &Layer::set_isotropic, D(Layer, set_isotropic), "albedo"_a)
        .def("set_henyey_greenstein", &Layer::set_henyey_greenstein, D(Layer, set_henyey_greenstein),
             "albedo"_a, "g"_a)
        .def("set_von_mises_fisher", &Layer::set_von_mises_fisher, D(Layer, set_von_mises_fisher),
             "albedo"_a, "kappa"_a)
        .def("set_fourier_coeffs", &Layer::set_fourier_coeffs, D(Layer, set_fourier_coeffs),
             "coeffs"_a)

        .def_static("add", [](const Layer &l1, const Layer &l2, bool homogeneous, double epsilon) {
            Layer l3(l1.nodes(), l1.weights(), l1.fourier_orders().first, l1.fourier_orders().second);
            Layer::add(l1, l2, l3, homogeneous, epsilon);
            return l3;
        }, D(Layer, add), "l1"_a, "l2"_a, "homogeneous"_a=false, "epsilon"_a=DROP_THRESHOLD)
        .def("add_to_top", [](Layer &l1, const Layer &l2, bool homogeneous, double epsilon) {
            l1.add_to_top(l2, homogeneous, epsilon);
        }, D(Layer, add_to_top), "l"_a, "homogeneous"_a = false, "epsilon"_a=DROP_THRESHOLD)
        .def("add_to_bottom", [](Layer &l1, const Layer &l2, bool homogeneous, double epsilon) {
            l1.add_to_bottom(l2, homogeneous, epsilon);
        }, D(Layer, add_to_bottom), "l"_a, "homogeneous"_a = false, "epsilon"_a=DROP_THRESHOLD)
        .def("expand", &Layer::expand, D(Layer, expand))
        .def_static("subtract", [](const Layer &ladd, const Layer &l1, double eps) {
            Layer l2(l1.nodes(), l1.weights(), l1.fourier_orders().first, l1.fourier_orders().second);
            Layer::subtract(ladd, l1, l2, eps);
            return l2;
        }, D(Layer, subtract), "ladd"_a, "l1"_a, "eps"_a)

        .def("scatter_coeff", &Layer::scatter_coeff, D(Layer, scatter_coeff))
        .def("eval", [](const Layer &l, py::array_t<double> mu_o, py::array_t<double> mu_i, py::array_t<double> phi_s, py::array_t<double> phi_d, bool clamp) {
            return py::vectorize([&](double mu_o, double mu_i, double phi_s, double phi_d) {
                return l.eval(mu_o, mu_i, phi_s, phi_d, clamp); })(mu_o, mu_i, phi_s, phi_d);
            }, D(Layer, eval), "mu_o"_a, "mu_i"_a, "phi_s"_a = 0, "phi_d"_a = 0, "clamp"_a = true)
        .def("find_truncation", [](const Layer &layer, MatrixS::Index i, MatrixS::Index o, double error) {
            size_t ms_trunc, md_trunc;
            layer.find_truncation(i, o, error, ms_trunc, md_trunc);
            return std::make_pair(ms_trunc, md_trunc);
        }, D(Layer, find_truncation), "i"_a, "o"_a, "error"_a)
        .def("fourier_slice", [](const Layer &layer, int o, int i) {
            MatrixX coeffs;
            layer.fourier_slice(o, i, coeffs);
            return coeffs;
        }, D(Layer, fourier_slice))
        .def("fourier_slice_interpolated", [](const Layer &layer, double mu_o, double mu_i) {
            MatrixX coeffs;
            layer.fourier_slice_interpolated(mu_o, mu_i, coeffs);
            return coeffs;
        }, D(Layer, fourier_slice_interpolated))

        .def("__repr__", &Layer::to_string, D(Layer, to_string))
        .def_property_readonly("resolution", &Layer::resolution, D(Layer, resolution))
        .def_property_readonly("fourier_orders", &Layer::fourier_orders, D(Layer, fourier_orders))
        .def_property_readonly("weights", &Layer::weights, D(Layer, weights))
        .def_property_readonly("nodes", &Layer::nodes, D(Layer, nodes))

        .def_property_readonly("reflection_top", [](const Layer &layer) {
            return layer.reflection_top();
        }, D(Layer, reflection_top))
        .def_property_readonly("reflection_bottom", [](const Layer &layer) {
            return layer.reflection_bottom();
        }, D(Layer, reflection_bottom))
        .def_property_readonly("transmission_top_bottom", [](const Layer &layer) {
            return layer.transmission_top_bottom();
        }, D(Layer, transmission_top_bottom))
        .def_property_readonly("transmission_bottom_top", [](const Layer &layer) {
            return layer.transmission_bottom_top();
        }, D(Layer, transmission_bottom_top))

        .def("set_reflection_top", [](Layer &layer, MatrixS m) {
            layer.reflection_top() = m;
        }, D(Layer, reflection_top))
        .def("set_reflection_bottom", [](Layer &layer, MatrixS m) {
            layer.reflection_bottom() = m;
        }, D(Layer, reflection_bottom))
        .def("set_transmission_top_bottom", [](Layer &layer, MatrixS m) {
            layer.transmission_top_bottom() = m;
        }, D(Layer, transmission_top_bottom))
        .def("set_transmission_bottom_top", [](Layer &layer, MatrixS m) {
            layer.transmission_bottom_top() = m;
        }, D(Layer, transmission_bottom_top));

    m.def("microfacet_parameter_heuristic", microfacet_parameter_heuristic,
          "alpha_u"_a, "alpha_v"_a, "eta"_a, D(microfacet_parameter_heuristic));
    m.def("henyey_greenstein_parameter_heuristic", henyey_greenstein_parameter_heuristic,
          "g"_a, D(henyey_greenstein_parameter_heuristic));
}
