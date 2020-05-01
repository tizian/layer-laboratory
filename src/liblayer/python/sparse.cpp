#include <mitsuba/layer/sparse.h>
#include <mitsuba/python/python.h>

MTS_PY_EXPORT(sparse) {
    m.def("mmul", &mmul, D(mmul), "A"_a, "B"_a, "epsilon"_a = 0);

#if defined HAVE_UMFPACK
    py::class_<UMFPack>(m, "UMFPack")
        .def(py::init([](const MatrixS &value) { return new UMFPack(value, true); }))
        .def("solve_dense", &UMFPack::solve_dense, "b"_a, "epsilon"_a = 0.0)
        .def("solve_sparse", &UMFPack::solve_sparse, "b"_a, "epsilon"_a = 0.0);
#endif
}
