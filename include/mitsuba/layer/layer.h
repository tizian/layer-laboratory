#pragma once

#include <mitsuba/layer/fwd.h>

NAMESPACE_BEGIN(mitsuba)

#define ERROR_GOAL     1e-3
#define DROP_THRESHOLD 1e-9

class MTS_EXPORT_LAYER Layer {
public:
    /**
     * \brief Create a new layer with the given discretization in zenith angles
     *
     * \param nodes
     *    A vector with the zenith angle cosines of the chosen discretization
     *
     * \param weights
     *    Associated weights for each angle. Usually, the 'nodes' and 'weights'
     *    are generated using some kind of quadrature rule (e.g. Gauss-Legendre,
     *    Gauss-Lobatto, etc.)
     *
     * \param fourier_orders_s
     *    Specifies the number of coefficients to use for the azimuthal Fourier
     *    expansion of phi_s (default: 1)
     *
     * \param fourier_orders_d
     *    Specifies the number of coefficients to use for the azimuthal Fourier
     *    expansion of phi_d (default: 1)
     */
    Layer(const VectorX &nodes, const VectorX &weights, size_t fourier_orders_s = 1,
                                                        size_t fourier_orders_d = 1);

    /// Return the number of Fourier orders
    std::pair<size_t, size_t> fourier_orders() const {
        return std::make_pair(m_fourier_orders_s, m_fourier_orders_d);
    }

    /// Return the number of nodes (i.e. the number of discretizations in \mu)
    size_t resolution() const { return m_nodes.size(); }

    /// Clear all scattering data
    void clear();

    /// Clear scattering data related to the reflection from the bottom
    void clear_backside();

    /// Initialize the layer with an empty (non-scattering) layer
    void set_empty();

    /**
     * \brief Initialize the layer with a diffuse base layer
     *
     * \param albedo
     *    The layer's diffuse reflectance (in <tt>[0, 1]</tt>)
     */
    void set_diffuse(double albedo);


    /**
     * \brief Initialize the layer with a microfacet model (dielectric or conductor)
     *
     * \param eta
     *    Relative index of refraction (complex)
     *
     * \param alpha_u
     *    Beckmann roughness parameter in U direction
     *
     * \param alpha_v
     *    Beckmann roughness parameter in V direction
     *
     * \param fourier_orders_s
     *    Number of phi_s Fourier orders that should be used internally
     *    in the computation. Defaults to the value returned by
     *    \ref fourier_orders()
     *
     * \param fourier_orders_d
     *    Number of phi_d Fourier orders that should be used internally
     *    in the computation. Defaults to the value returned by
     *    \ref fourier_orders()
     *
     * \param component
     *    Indicates whether which component should be considered:
     *    The full model (default, 0), exponential (1), shadowing-masking (2),
     *    Fresnel + remainder (3)
     *
     * \param n_samples_phi_s
     *    Number of samples to use for the Filon integration along phi_s.
     *    Defaults to the same value as used for \ref fourier_orders_s
     *
     * \param isotropic_g
     *    Indicates whether an isotropic version of the shadowing-masking function
     *    should be used. (Default: false)
     */
    void set_microfacet(std::complex<double> eta, double alpha_u, double alpha_v,
                        size_t fourier_orders_s = 1, size_t fourier_orders_d = 1,
                        int component = 0, int n_samples_phi_s = 30, bool svd_reg=false);

    /**
     * \brief Initialize the layer with an isotropic phase function
     *
     * \param albedo
     *    The layer's single scattering albedo reflectance (in <tt>[0, 1]</tt>)
     */
    void set_isotropic(double albedo);

    /**
     * \brief Initialize the layer with a Henyey-Greenstein phase function
     *
     * \param albedo
     *    The layer's single scattering albedo reflectance (in <tt>[0, 1]</tt>)
     *
     * \param g
     *    The layer's HG "anisotropy" parameter (in <tt>[-1, 1]</tt>)
     */
    void set_henyey_greenstein(double albedo, double g);

    /**
     * \brief Initialize the layer with a von Mises-Fisher phase function
     *
     * \param albedo
     *    The layer's single scattering albedo reflectance (in <tt>[0, 1]</tt>)
     *
     * \param kappa
     *    The layer's kappa "anisotropy" parameter
     */
    void set_von_mises_fisher(double albedo, double kappa);

     /**
     * \brief Initialize the layer with explicit Fourier series coefficients
     * for all pairs of elevation angles.
     * Useful to compute a layer from external data such as measurements.
     *
     * \param coeffs
     *   List of matrices, each containing the 2D Fourier series coefficients
     *   for a specific pair of elevation angles.
     *   The size of the list should be resolution()*resolution().
     */
    void set_fourier_coeffs(std::vector<MatrixX> coeffs);

    /**
     * \brief Additive composition of two layers
     *
     * \param l1
     *    Input layer 1
     *
     * \param l2
     *    Input layer 2
     *
     * \param output
     *    Used to return the resulting layer
     *
     * \param homogeneous
     *    When both layers are homogenous, (i.e. if their two sides are
     *    indistinguishable, this flag should be set to \c true to get a
     *    speed-up). Default: \c false
     *
     * \param epsilon
     *    Drop threshold. Only Fourier coefficients larger than this are
     *    considered. This greatly affects both the accuracy and the performance
     *    of the adding process.
     *
     * \remark
     *    In the Python API, the \c output parameter is directly returned
     */
    static void add(const Layer &l1, const Layer &l2, Layer &output,
                    bool homogeneous = false, double epsilon = DROP_THRESHOLD);

    /**
     * \brief Append a layer above the current one
     *
     * This is just a convenience wrapper around \ref Layer::add()
     *
     * \param l
     *    The layer to be appended
     *
     * \param homogeneous
     *    When the layers are homogenous, (i.e. if their two sides are
     *    indistinguishable, this flag should be set to \c true to get a
     *    speed-up). Default: \c false
     *
     * \param epsilon
     *    Drop threshold. Only Fourier coefficients larger than this are
     *    considered. This greatly affects both the accuracy and the performance
     *    of the adding process.
     */
    void add_to_top(const Layer &l, bool homogeneous = false, double epsilon = DROP_THRESHOLD) {
        Layer::add(l, *this, *this, homogeneous, epsilon);
    }

    /**
     * \brief Append a layer below the current one
     *
     * This is just a convenience wrapper around \ref Layer::add()
     *
     * \param l
     *    The layer to be appended
     *
     * \param homogeneous
     *    When the layers are homogenous, (i.e. if their two sides are
     *    indistinguishable, this flag should be set to \c true to get a
     *    speed-up). Default: \c false
     *
     * \param epsilon
     *    Drop threshold. Only Fourier coefficients larger than this are
     *    considered. This greatly affects both the accuracy and the performance
     *    of the adding process.
     */
    void add_to_bottom(const Layer &l, bool homogeneous = false, double epsilon = DROP_THRESHOLD) {
        Layer::add(*this, l, *this, homogeneous, epsilon);
    }

    /**
     * \brief Subtractive composition of two layers. Currently, only recovering "Rt_2"
     * underneath a transmissive layer (Rt_1, Rb_1, Ttb_1, Tbt_1) is implemented.
     *
     * \param ladd
     *    Input combined layer
     *
     * \param l1
     *    Input top layer to be subtracted
     *
     * \param l2
     *    Used to return the resulting (bottom) layer
     *
     * \param eps
     *    Tikhonov regularization parameter
     *
     * \remark
     *    In the Python API, the \c l2 parameter is directly returned
     */
    static void subtract(const Layer &ladd, const Layer &l1, Layer &l2, double eps);

    /// Solve for the transport matrix of a layer with the given optical thickness (using Adding-Doubling)
    void expand(double tau);

    /// Access a scattering matrix entry
    double scatter_coeff(int s, int d, size_t o, size_t i) const;

    /// Return the used integration weights
    const VectorX &weights() const { return m_weights; }

    /// Return the used integration nodes
    const VectorX &nodes() const { return m_nodes; }

    /// Return the scattering matrix for reflection on the top
    MatrixS &reflection_top() { return m_reflection_top; }

    /// Return the scattering matrix for reflection on the bottom
    MatrixS &reflection_bottom() { return m_reflection_bottom; }

    /// Return the scattering matrix for transmission from top to bottom
    MatrixS &transmission_top_bottom() { return m_transmission_top_bottom; }

    /// Return the scattering matrix for transmission from bottom to top
    MatrixS &transmission_bottom_top() { return m_transmission_bottom_top; }

    /// Return the scattering matrix for reflection on the top (const version)
    const MatrixS &reflection_top() const { return m_reflection_top; }

    /// Return the scattering matrix for reflection on the bottom (const version)
    const MatrixS &reflection_bottom() const { return m_reflection_bottom; }

    /// Return the scattering matrix for transmission from top to bottom (const version)
    const MatrixS &transmission_top_bottom() const { return m_transmission_top_bottom; }

    /// Return the scattering matrix for transmission from bottom to top (const version)
    const MatrixS &transmission_bottom_top() const { return m_transmission_bottom_top; }

    /// Reverse the layer (Swap top and bottom interfaces)
    void reverse();

    /**
     * \brief Evaluate the BSDF stored in the layer.
     *
     * \param mu_o
     *      The outgoing elevation angle cosine
     *
     * \param mu_i
     *      The incident elevation angle cosine
     *
     * \param phi_s
     *      The azimuthal angle sum
     *
     * \param phi_d
     *      The azimuthal angle difference
     *
     * \param clamp
     *      Choose whether the result should be clamped to the range [0, inf)
     *      (default: true)
     *
     * \return
     *      The BSDF value for given angles
     */
    double eval(double mu_o, double mu_i, double phi_s = 0.0, double phi_d = 0.0, bool clamp=true) const;

    /**
     * \brief Find optimal box for a given maximal truncation error.
     *
     * \param o
     *      Outgoing elevation angle node index
     *
     * \param i
     *      Incident elevation angle node index
     *
     * \param error
     *      Maximal truncation error
     *
     * \param[out] ms_trunc
     *      The number of required Fourier coefficients in phi_s
     *
     * \param[out] md_trunc
     *      The number of required Fourier coefficients in phi_d
     */
    void find_truncation(MatrixS::Index o, MatrixS::Index i, double error,
                         size_t &ms_trunc, size_t &md_trunc) const;

    /// For debugging: return a 2D Fourier series for the given parameters (in phi_s, phi_d format)
    void fourier_slice(int o, int i, MatrixX &coeffs) const;

    /// For debugging: return a 2D Fourier series for the given parameters (in phi_s, phi_d format)
    void fourier_slice_interpolated(double mu_o, double mu_i, MatrixX &coeffs) const;

    /// Return a human-readable summary
    std::string to_string() const;

protected:
    /// Helper struct for sparse matrix construction
    struct Quintet {
        int32_t s, d;
        uint32_t o, i;
        double value;

        Quintet(int32_t s, int32_t d, uint32_t o, uint32_t i, double value)
         : s(s), d(d), o(o), i(i), value(value) {}
    };

    /// Initialize from a list of quintets
    void set_quintets(const std::vector<Quintet> &quintets);

protected:
    /// Integration nodes
    VectorX m_nodes;

    /// Integration weights
    VectorX m_weights;

    /// Number of Fourier modes
    size_t m_fourier_orders_s, m_fourier_orders_d;

    /// Scattering matrices
    MatrixS m_reflection_top,
            m_reflection_bottom,
            m_transmission_top_bottom,
            m_transmission_bottom_top;
};

/// Heuristic to guess a suitable number of parameters (Microfacet model)
extern MTS_EXPORT_LAYER std::tuple<int, int, int> microfacet_parameter_heuristic(double alpha_u, double alpha_v,
                                                                                 std::complex<double> &eta);

/// Heuristic to guess a suitable number of parameters (HG model)
extern MTS_EXPORT_LAYER std::tuple<int, int, int> henyey_greenstein_parameter_heuristic(double g);

NAMESPACE_END(mitsuba)
