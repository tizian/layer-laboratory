#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/layer/fwd.h>
#include <memory>

#if defined HAVE_UMFPACK
    #include <umfpack.h>
#endif

NAMESPACE_BEGIN(mitsuba)

extern MTS_EXPORT_LAYER MatrixS mmul(const MatrixS &A, const MatrixS &B, double epsilon = 0.0);

#if defined HAVE_UMFPACK

class MTS_EXPORT_LAYER UMFPack {
public:
    /// copy_A=true is only needed if the 'A' matrix may be deallocated in the meantime (e.g. Python)
    UMFPack(const MatrixS &A, bool copy_A = false);
    ~UMFPack();

    MatrixS solve_dense(const MatrixS &b_, double epsilon = 0.0);
    MatrixS solve_sparse(const MatrixS &b_, double epsilon = 0.0);

private:
    const MatrixS *m_A;
    MatrixS m_A_storage;
    void *m_numeric = nullptr;
    double m_control[UMFPACK_CONTROL];
    double m_info[UMFPACK_INFO];
    long m_lnz = 0, m_unz = 0, do_recip = 0;

    std::unique_ptr<long[]>   Lp, Li, Up, Ui, P, Q;
    std::unique_ptr<double[]> L, U, R;
    std::unique_ptr<long[]>   Pi;
};

#endif

NAMESPACE_END(mitsuba)
