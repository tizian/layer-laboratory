#include <mitsuba/layer/storage.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(BSDFStorage) {
    py::class_<BSDFStorage>(m, "BSDFStorage", D(BSDFStorage))
        .def(py::init<mitsuba::filesystem::path &, bool>(), D(BSDFStorage, BSDFStorage))
        .def(py::init<mitsuba::filesystem::path &>(), D(BSDFStorage, BSDFStorage))
        .def("close", &BSDFStorage::close, D(BSDFStorage, close))
        .def_property_readonly("max_orders", &BSDFStorage::max_orders, D(BSDFStorage, max_orders))
        .def_property_readonly("channel_count", &BSDFStorage::channel_count, D(BSDFStorage, channel_count))
        .def_property_readonly("node_count", &BSDFStorage::node_count, D(BSDFStorage, node_count))
        .def_property_readonly("basis_count", &BSDFStorage::basis_count, D(BSDFStorage, basis_count))
        .def_property_readonly("parameter_count", &BSDFStorage::parameter_count, D(BSDFStorage, parameter_count))
        .def_property_readonly("size", &BSDFStorage::size, D(BSDFStorage, size))
        .def_property_readonly("total_coeffs", &BSDFStorage::total_coeffs, D(BSDFStorage, total_coeffs))
        .def_property_readonly("metadata", &BSDFStorage::metadata, D(BSDFStorage, metadata))
        .def_property_readonly("eta", &BSDFStorage::eta, D(BSDFStorage, eta))
        .def("set_eta", &BSDFStorage::set_eta, D(BSDFStorage, set_eta))
        .def_property_readonly("nodes", &BSDFStorage::nodes, D(BSDFStorage, nodes))
        .def("alpha", [](const BSDFStorage &m, size_t i) {
                if (i >= 2)
                    throw py::index_error();
                return m.alpha((int) i);
             }, D(BSDFStorage, alpha))
        .def("set_alpha", [](BSDFStorage &m, size_t i, float v) {
                if (i >= 2)
                    throw py::index_error();
                m.set_alpha((int) i, v);
             }, D(BSDFStorage, set_alpha))
        .def("parameter_sample_count", [](const BSDFStorage &m, size_t i) {
                if (i >= m.parameter_count())
                    throw py::index_error();
                return m.parameter_sample_count(i);
             }, D(BSDFStorage, parameter_sample_count))
        .def("parameter_sample_positions", [](const BSDFStorage &m, size_t i) {
                if (i >= m.parameter_count())
                    throw py::index_error();
                py::list list;
                for (size_t j=0; j<m.parameter_sample_count(i); ++j)
                    list.append(py::float_(m.parameter_sample_positions(i)[j]));
                return list;
             }, D(BSDFStorage, parameter_sample_positions))
        .def("eval", [](const BSDFStorage &m, float mu_i, float mu_o, float phi_i, float phi_o, bool clamp) {
                return m.eval(mu_i, mu_o, phi_i, phi_o, nullptr, clamp);
            }, D(BSDFStorage, eval), "mu_i"_a, "mu_o"_a, "phi_i"_a = 0, "phi_o"_a = 0, "clamp"_a = true)
        .def("eval", [](const BSDFStorage &m, py::array_t<float> mu_i, py::array_t<float> mu_o, py::array_t<float> phi_i, py::array_t<float> phi_o, bool clamp) {
            return py::vectorize([&](float mu_i, float mu_o, float phi_i, float phi_o) {
                return m.eval(mu_i, mu_o, phi_i, phi_o, nullptr, clamp)[0]; })(mu_i, mu_o, phi_i, phi_o);
            }, D(BSDFStorage, eval), "mu_i"_a, "mu_o"_a, "phi_i"_a = 0, "phi_o"_a = 0, "clamp"_a = true)
        .def("pdf", [](const BSDFStorage &m, float mu_i, float mu_o, float phi_i, float phi_o) {
                return m.pdf(mu_i, mu_o, phi_i, phi_o);
            }, D(BSDFStorage, pdf))
        .def("pdf", [](const BSDFStorage &m, float mu_i, py::array_t<float> mu_o_, float phi_i, py::array_t<float> phi_o_) {
            auto mu_o  = mu_o_.unchecked<1>();
            auto phi_o = phi_o_.unchecked<1>();

            py::array_t<float> p(mu_o.shape(0));

            float *p_ptr = (float *) p.request().ptr;

            for (ssize_t i = 0; i < mu_o.shape(0); ++i) {
                p_ptr[i] = m.pdf(mu_i, mu_o[i], phi_i, phi_o[i]);
            }
            return p;
        }, D(BSDFStorage, pdf))

        .def("sample", [](const BSDFStorage &m, float mu_i, float phi_i, const Point2f &sample) {
                return m.sample(mu_i, phi_i, sample);
            }, D(BSDFStorage, sample))
        .def("sample", [](const BSDFStorage &m, float mu_i, float phi_i, py::array_t<float> samples_) {
            auto samples = samples_.unchecked<2>();
            if (samples.shape(1) != 2)
                Throw("BSDFStorage::sample: Expected samples array with dimension 2.");

            py::array_t<float> mu_o(samples.shape(0));
            py::array_t<float> phi_o(samples.shape(0));
            py::array_t<float> pdf(samples.shape(0));
            py::array_t<float> result(samples.shape(0));

            float *mu_o_ptr   = (float *) mu_o.request().ptr,
                  *phi_o_ptr  = (float *) phi_o.request().ptr,
                  *pdf_ptr    = (float *) pdf.request().ptr,
                  *result_ptr = (float *) result.request().ptr;

            for (ssize_t i = 0; i < samples.shape(0); ++i) {
                Color3f weight;
                float mu_o, phi_o, pdf;

                std::tie(weight, mu_o, phi_o, pdf) = m.sample(mu_i, phi_i, Point2f(samples(i, 0), samples(i, 1)));

                result_ptr[i] = weight.g();
                mu_o_ptr[i]   = mu_o;
                phi_o_ptr[i]  = phi_o;
                pdf_ptr[i]    = pdf;
            }
            return std::make_tuple(result, mu_o, phi_o, pdf);
        }, D(BSDFStorage, sample))

        .def("fourier_slice", [](const BSDFStorage &m, int i, int o, int channel) {
            MatrixX coeffs;
            m.fourier_slice(i, o, channel, coeffs);
            return coeffs;
        }, D(BSDFStorage, fourier_slice))
        .def("fourier_slice_interpolated", [](const BSDFStorage &m, float mu_i, float mu_o, int channel) {
            MatrixX coeffs;
            m.fourier_slice_interpolated(mu_i, mu_o, channel, coeffs);
            return coeffs;
        }, D(BSDFStorage, fourier_slice_interpolated))

        .def("__repr__", &BSDFStorage::to_string)
        .def("stats", &BSDFStorage::stats)
        .def_static("from_layer", [](const mitsuba::filesystem::path &path, const Layer *layer, float error) {
                return BSDFStorage::from_layer(path, layer, "", error);
            }, D(BSDFStorage, from_layer))
        .def_static("from_layer_rgb", [](const mitsuba::filesystem::path &path, const Layer *layer_r, const Layer *layer_g, const Layer *layer_b, float error) {
                return BSDFStorage::from_layer_rgb(path, layer_r, layer_g, layer_b, "", error);
            }, D(BSDFStorage, from_layer_rgb))
        .def_static("from_layer_general", [](const mitsuba::filesystem::path &path, std::vector<const Layer *> layers_,
                                             size_t n_channels, size_t n_bases, size_t n_parameters,
                                             std::vector<size_t> param_sample_counts_, std::vector<std::vector<float>> param_sample_positions_,
                                             float error) {
                std::cout << "from_layer_general: " << std::endl;
                const Layer **layers = &layers_[0];
                size_t *param_sample_counts = &param_sample_counts_[0];
                const float **parameter_sample_positions = (const float **) alloca(sizeof(float *) * n_parameters);
                for (size_t i = 0; i < n_parameters; ++i) {
                    parameter_sample_positions[i] = &(param_sample_positions_[i][0]);
                }

                return BSDFStorage::from_layer_general(path, layers,
                                                       n_channels, n_bases, n_parameters,
                                                       param_sample_counts, parameter_sample_positions,
                                                       "", error);
        });
}
