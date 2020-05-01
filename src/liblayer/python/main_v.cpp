#include <mitsuba/python/python.h>

#define MODULE_NAME MTS_MODULE_NAME(layer, MTS_VARIANT_NAME)

#define PY_TRY_CAST(Type)                                         \
    if (auto tmp = dynamic_cast<Type *>(o); tmp)                  \
        return py::cast(tmp);

/// Helper routine to cast Mitsuba plugins to their underlying interfaces
static py::object caster(Object * /* o */) {
    MTS_PY_IMPORT_TYPES()

    return py::object();
}

PYBIND11_MODULE(MODULE_NAME, m) {
    // Temporarily change the module name (for pydoc)
    m.attr("__name__") = "mitsuba.layer";

    /// Register the variant-specific caster with the 'core_ext' module
    auto casters = (std::vector<void *> *) (py::capsule)(
        py::module::import("mitsuba.core_ext").attr("casters"));
    casters->push_back((void *) caster);

    // Change module name back to correct value
    m.attr("__name__") = "mitsuba." ENOKI_TOSTRING(MODULE_NAME);
}
