#include <mitsuba/layer/sparse.h>
#include <mitsuba/layer/layer.h>
#include <mitsuba/core/logger.h>
#include <tbb/tbb.h>
#include <mutex>

NAMESPACE_BEGIN(mitsuba)

MatrixS mmul(const MatrixS &A, const MatrixS &B, double epsilon) {
    if (!(A.rows() == B.rows() && A.cols() == B.cols() && A.rows() == A.cols()))
        Throw("Invalid input matrix dimensions");

    MatrixS result(A.rows(), A.cols());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve((A.nonZeros() + B.nonZeros()) * 2);
    tbb::spin_mutex mutex;

    tbb::parallel_for(
        tbb::blocked_range<ssize_t>(0, A.cols(), 1),
        [&](const tbb::blocked_range<ssize_t> &range) {
            std::unique_ptr<bool[]>   mask(new bool[A.rows()]);
            std::unique_ptr<long[]>   indices(new long[A.rows()]);
            std::unique_ptr<double[]> values(new double[A.rows()]);
            memset(mask.get(), 0, sizeof(bool) * A.rows());
            for (ssize_t j = range.begin(); j != range.end(); ++j) {
                int nnz = 0;
                for (auto B_it = MatrixS::InnerIterator(B, j); B_it; ++B_it) {
                    int Bi = B_it.index();
                    double Bv = B_it.value();
                    for (auto A_it = MatrixS::InnerIterator(A, Bi); A_it; ++A_it) {
                        int Ai = A_it.index();
                        double Av = A_it.value();

                        if (!mask[Ai]) {
                            mask[Ai] = true;
                            values[Ai] = Av * Bv;
                            indices[nnz] = Ai;
                            nnz++;
                        } else {
                            values[Ai] += Av * Bv;
                        }
                    }
                }

                std::lock_guard<tbb::spin_mutex> lock(mutex);
                for (int i = 0; i < nnz; ++i) {
                    int k = indices[i];
                    mask[k] = false;
                    if (std::abs(values[k]) > epsilon)
                        triplets.emplace_back(k, j, values[k]);
                }
            }
        }
    );
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

#if defined HAVE_UMFPACK

void report_umpfack_error(int rv, double *info, double *control) {
    std::cout << "return value: " << rv;
    if (rv == UMFPACK_WARNING_singular_matrix) {
        std::cout << "   (Singular matrix)" << std::endl;
    } else if (rv == UMFPACK_ERROR_out_of_memory) {
        std::cout << "   (Out of memory)" << std::endl;
    } else {
        std::cout << "   (Unkown code)" << std::endl;
    }

    std::cout << "Info:" << std::endl;
#ifdef _WIN64
    umfpack_di_report_info(control, info);
#else
    umfpack_dl_report_info(control, info);
#endif
    Throw("UMFPack factorization failed");
}

UMFPack::UMFPack(const MatrixS &A, bool store_A)  {
    if (store_A) {
        m_A_storage = A;
        m_A = &m_A_storage;
    } else {
        m_A = &A;
    }
    int n = m_A->rows();

#ifdef _WIN64
    umfpack_di_defaults(m_control);
#else
    umfpack_dl_defaults(m_control);
#endif

    m_control[UMFPACK_IRSTEP] = 0;
    m_control[UMFPACK_PRL] = 0;
    m_control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
    m_control[UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
    void *symbolic = nullptr;

#ifdef _WIN64
    int rv = umfpack_di_symbolic(n, n, (int *) m_A->outerIndexPtr(), (int *) m_A->innerIndexPtr(), m_A->valuePtr(), &symbolic, m_control, m_info);
#else
    int rv = umfpack_dl_symbolic(n, n, m_A->outerIndexPtr(), m_A->innerIndexPtr(), m_A->valuePtr(), &symbolic, m_control, m_info);
#endif

    if (rv != UMFPACK_OK) {
        Log(Warn, "umfpack_d*_symbolic failed");
        report_umpfack_error(rv, m_info, m_control);
    }

#ifdef _WIN64
    rv = umfpack_di_numeric((int *) m_A->outerIndexPtr(), (int *) m_A->innerIndexPtr(), m_A->valuePtr(), symbolic, &m_numeric, m_control, m_info);
#else
    rv = umfpack_dl_numeric(m_A->outerIndexPtr(), m_A->innerIndexPtr(), m_A->valuePtr(), symbolic, &m_numeric, m_control, m_info);
#endif

    if (rv != UMFPACK_OK) {
        Log(Warn, "umfpack_d*_numeric failed");
        report_umpfack_error(rv, m_info, m_control);
    }

#ifdef _WIN64
    umfpack_di_free_symbolic(&symbolic);
    int unused;
    rv = umfpack_di_get_lunz((int *)&m_lnz, (int *)&m_unz, &unused, &unused, &unused, m_numeric);
    umfpack_di_report_info(m_control, m_info);
#else
    umfpack_dl_free_symbolic(&symbolic);
    long unused;
    rv = umfpack_dl_get_lunz(&m_lnz, &m_unz, &unused, &unused, &unused, m_numeric);
    umfpack_dl_report_info(m_control, m_info);
#endif

    if (rv != UMFPACK_OK) {
        Throw("umfpack_d*_get_lunz failed: %i", rv);
    }

    Lp = std::unique_ptr<long[]>   (new long[n + 1]);
    Up = std::unique_ptr<long[]>   (new long[n + 1]);
    Li = std::unique_ptr<long[]>   (new long[m_lnz]);
    Ui = std::unique_ptr<long[]>   (new long[m_unz]);
    L  = std::unique_ptr<double[]> (new double[m_lnz]);
    U  = std::unique_ptr<double[]> (new double[m_unz]);
    P  = std::unique_ptr<long[]>   (new long[n]);
    Pi = std::unique_ptr<long[]>   (new long[n]);
    Q  = std::unique_ptr<long[]>   (new long[n]);
    R  = std::unique_ptr<double[]> (new double[n]);

#ifdef _WIN64
    rv = umfpack_di_get_numeric((int *) Lp.get(), (int *)Li.get(), L.get(), (int *)Up.get(),
                                (int *) Ui.get(), U.get(), (int *)P.get(), (int *)Q.get(),
                                nullptr, (int *) &do_recip, R.get(), m_numeric);
#else
    rv = umfpack_dl_get_numeric(Lp.get(), Li.get(), L.get(), Up.get(),
                                Ui.get(), U.get(), P.get(), Q.get(),
                                nullptr, &do_recip, R.get(), m_numeric);
#endif

    if (rv != UMFPACK_OK) {
        Throw("umfpack_d*_get_numeric: failed!");
    }

    for (int i = 0; i < n; ++i) {
        Pi[P[i]] = i;
    }

    std::unique_ptr<double[]> L2(new double[m_lnz]);
    std::unique_ptr<long[]>   Lp2(new long[n + 1]);
    std::unique_ptr<long[]>   Li2(new long[m_lnz]);

#ifdef _WIN64
    rv = umfpack_di_transpose(n, n, (int *) Lp.get(), (int *) Li.get(), L.get(), nullptr,
                              nullptr, (int *) Lp2.get(), (int *) Li2.get(), L2.get());
#else
    rv = umfpack_dl_transpose(n, n, Lp.get(), Li.get(), L.get(), nullptr,
                              nullptr, Lp2.get(), Li2.get(), L2.get());
#endif

    if (rv != UMFPACK_OK) {
        Throw("umfpack_d*_transpose failed: %i", rv);
    }

    Lp = std::move(Lp2); Li = std::move(Li2); L = std::move(L2);
}

UMFPack::~UMFPack() {
#ifdef _WIN64
    umfpack_di_free_numeric(&m_numeric);
#else
    umfpack_dl_free_numeric(&m_numeric);
#endif
}

MatrixS UMFPack::solve_dense(const MatrixS &b, double epsilon) {
    std::vector<Eigen::Triplet<double>> triplets;
    std::mutex mutex;
    tbb::parallel_for(
        tbb::blocked_range<int>(0, b.cols(), 1),
        [&](const tbb::blocked_range<int> &range) {
            std::unique_ptr<double[]> in(new double[b.rows()]),
                                      out(new double[b.rows()]);
            std::vector<Eigen::Triplet<double>> triplets_local;
            for (int k = range.begin(); k != range.end(); ++k) {
                memset(in.get(), 0, sizeof(double) * b.rows());
                for (auto it = MatrixS::InnerIterator(b, k); it; ++it)
                    in[it.index()] = it.value();

#ifdef _WIN64
                umfpack_di_solve(UMFPACK_A, (int *) m_A->outerIndexPtr(),
                                 (int *) m_A->innerIndexPtr(), m_A->valuePtr(),
                                 out.get(), in.get(), m_numeric, m_control, m_info);
#else
                umfpack_dl_solve(UMFPACK_A, m_A->outerIndexPtr(),
                                 m_A->innerIndexPtr(), m_A->valuePtr(),
                                 out.get(), in.get(), m_numeric, m_control, m_info);
#endif


                for (int l = 0; l < b.rows(); ++l) {
                    if (std::abs(out[l]) > epsilon)
                        triplets_local.push_back(
                            Eigen::Triplet<double>(l, k, out[l]));
                }
                mutex.lock();
                triplets.insert(triplets.end(), triplets_local.begin(), triplets_local.end());
                mutex.unlock();
                triplets_local.clear();
            }
        }
    );
    MatrixS result(b.rows(), b.rows());
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

MatrixS UMFPack::solve_sparse(const MatrixS &B, double epsilon) {
    int n = m_A->rows();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(2 * (m_A->nonZeros() + B.nonZeros()));
    tbb::spin_mutex mutex;

    tbb::parallel_for(
        tbb::blocked_range<ssize_t>(0, n, 1),
        [&](const tbb::blocked_range<ssize_t> &range) {
            std::unique_ptr<double[]> x(new double[n]);
            std::unique_ptr<bool[]>   x_m(new bool[n]);
            std::unique_ptr<long[]>   x_i(new long[n]);
            memset(x_m.get(), 0, sizeof(bool) * n);

            for (int j = range.begin(); j != range.end(); ++j) {
                int max_idx = std::numeric_limits<int>::min(),
                    min_idx = std::numeric_limits<int>::max(),
                    nnz = 0;

                for (MatrixS::InnerIterator it(B, j); it; ++it) {
                    int k = it.index(), i = Pi[k];
                    x[i] = it.value() * (do_recip ? R[k] : 1.0 / R[k]);
                    Assert(!x_m[i]);
                    x_m[i] = true;
                    x_i[nnz++] = i;
                    max_idx = std::max(i, max_idx);
                    min_idx = std::min(i, min_idx);
                }

                for (int i = min_idx; i <= max_idx; ++i) {
                    if (!x_m[i])
                        continue;
                    double v = x[i];
                    if (std::abs(v) < epsilon)
                        continue;
                    int offs = Lp[i], offs_n = Lp[i+1];
                    Assert(Li[offs] == i);

                    for (int k = offs + 1; k != offs_n; ++k) {
                        int ki = Li[k];
                        double kv = L[k];

                        if (x_m[ki]) {
                            x[ki] -= v * kv;
                        } else {
                            x[ki] = -v * kv;
                            x_m[ki] = true;
                            x_i[nnz++] = ki;
                            max_idx = std::max(max_idx, ki);
                        }
                    }
                }

                for (int i = max_idx; i >= min_idx; --i) {
                    if (!x_m[i])
                        continue;
                    int offs = Up[i], offs_n = Up[i+1];

                    Assert(Ui[offs_n-1] == i);
                    if (Ui[offs_n-1] != i)
                        Throw("Internal error! %i %i", i, Ui[offs_n-1]);

                    x[i] /= U[offs_n - 1];
                    double v = x[i];
                    if (std::abs(v) < epsilon)
                        continue;

                    for (int k = offs; k != offs_n - 1; ++k) {
                        int ki = Ui[k];
                        double kv = U[k];

                        if (x_m[ki]) {
                            x[ki] -= v * kv;
                        } else {
                            x[ki] = -v * kv;
                            x_m[ki] = true;
                            x_i[nnz++] = ki;
                            min_idx = std::min(min_idx, ki);
                        }
                    }
                }

                std::lock_guard<tbb::spin_mutex> lock(mutex);
                for (int k = 0; k < nnz; ++k) {
                    int i  = x_i[k];
                    triplets.push_back(Eigen::Triplet<double>(Q[i], j, x[i]));
                    x_m[i] = false;
                }
            }
        }
    );
    MatrixS result(n, n);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

#endif

NAMESPACE_END(mitsuba)
