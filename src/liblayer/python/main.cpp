#include <mitsuba/python/python.h>
#include <mitsuba/layer/fourier.h>
#include <mitsuba/layer/layer.h>

MTS_PY_DECLARE(fourier);
MTS_PY_DECLARE(Layer);
MTS_PY_DECLARE(microfacet);
MTS_PY_DECLARE(phase);
MTS_PY_DECLARE(sparse);
MTS_PY_DECLARE(BSDFStorage);

PYBIND11_MODULE(layer_ext, m) {
    // Temporarily change the module name (for pydoc)
    m.attr("__name__") = "mitsuba.layer";

    MTS_PY_IMPORT(fourier);
    MTS_PY_IMPORT(Layer);
    MTS_PY_IMPORT(microfacet);
    MTS_PY_IMPORT(phase);
    MTS_PY_IMPORT(sparse);
    MTS_PY_IMPORT(BSDFStorage);

    // Change module name back to correct value
    m.attr("__name__") = "mitsuba.layer_ext";
}
