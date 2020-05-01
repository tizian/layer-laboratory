#pragma once

#include <mitsuba/layer/layer.h>
#include <mitsuba/core/mmap.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/spline.h>
#include <mitsuba/layer/fourier.h>

NAMESPACE_BEGIN(mitsuba)

using FloatP = Packet<float>;
using Point2f = Point<float, 2>;

/**
 * \brief Storage class for anisotropic BSDFs
 *
 * This class implements sparse storage support for anisotropic BSDFs which are
 * point-sampled as a function of the incident and exitant zenith angles and
 * expanded into Fourier coefficients as a function of the azimuthal
 * sum and difference angles.
 */
class MTS_EXPORT_LAYER BSDFStorage : public Object {
public:
    typedef uint32_t OffsetType;

    /// Map an existing BSDF storage file into memory
    BSDFStorage(const fs::path &filename, bool read_only = true);

    /// Return the maximal number of Fourier orders
    std::pair<size_t, size_t> max_orders() const {
        return std::make_pair((size_t) m_header->n_max_order_s,
                              (size_t) m_header->n_max_order_d);
    }

    /// Return the number of color channels
    size_t channel_count() const { return (size_t) m_header->n_channels; }

    /// Return the resolution of the discretization in \mu_i and \mu_o
    size_t node_count() const { return (size_t) m_header->n_nodes; }

    /// Return the number of basis functions stored in this file (usually just 1)
    size_t basis_count() const { return (size_t) m_header->n_bases; }

    /// Return the number of Fourier orders for the marginals (pdf and cdf) over \mu_o
    size_t marginal_order_count() const { return (size_t) m_header->n_max_order_s / 2 + 1; }

    /// Return the number of model parameters
    size_t parameter_count() const { return (size_t) m_header->n_parameters; }

    /// Return the number of samples associated with parameter \c i
    size_t parameter_sample_count(size_t i) const { return (size_t) m_param_sample_counts[i]; }

    /// Return the sample positions associated with parameter \c i
    const float *parameter_sample_positions(size_t i) const { return m_param_sample_positions_nested[i]; }

    /// Return the size of the underlying representation in bytes
    size_t size() const;

    /// Return the total number of stored coefficients
    size_t total_coeffs() const;

    /// Return metadata attached to the BSDF file (if any)
    const std::string &metadata() const { return m_metadata; }

    /// Return the relative index of refraction
    float eta() const { return m_header->eta; }

    /// Set the relative index of refraction
    void set_eta(float eta) { m_header->eta = eta; }

    /// Return the Beckmann-equivalent roughness (0: bottom, 1: top surface)
    float alpha(int index) const { assert(index >= 0 && index <= 1); return m_header->alpha[index]; }

    /// Set the Beckmann-equivalent roughness (0: bottom, 1: top surface)
    void set_alpha(int index, float alpha) { assert(index >= 0 && index <= 1); m_header->alpha[index] = alpha; }

    /// Return the nodes of the underlying discretization in \mu_i and \mu_o
    const float *nodes() const { return m_nodes; }

    /// Return a pointer to the coefficients of the CDF associated with the incident angle \c o
    float *cdf(size_t o) { return m_cdf_mu + o*marginal_order_count()*node_count()*basis_count(); }

    /// Return a pointer to the coefficients of the CDF associated with the incident angle \c o (const version)
    const float *cdf(size_t o) const { return m_cdf_mu + o*marginal_order_count()*node_count()*basis_count(); }

    /**
     * \brief Evaluate the model.
     *
     * \param mu_i
     *      The incident elevation angle cosine
     * \param mu_o
     *      The outgoing elevation angle cosine
     * \param phi_i
     *      The incident azimuthal angle
     * \param phi_o
     *      The outgoing azimuthal angle
     * \param basis_coeffs
     *      Basis coefficients for parameter interpolation
     *      (default: \c nullptr)
     * \param clamp
     *      Choose whether the result should be clamped to the range [0, inf)
     *      (default: true)
     * \remark
     *      The Python API lacks the \c basis_coeffs parameter.
     * \return
     *      The BSDF value for given angles
     */
    Color3f eval(float mu_i, float mu_o, float phi_i, float phi_o,
                 const float *basis_coeffs = nullptr, bool clamp=true) const;

    /**
     * \brief Importance sample the model.
     *
     * \param mu_i
     *      The incident elevation angle cosine
     * \param phi_i
     *      The incident azimuthal angle
     * \param sample
     *      A pair of uniformly distributed random samples in
     *      the interval <tt>[0,1]</tt>
     * \param basis_coeffs
     *      Basis coefficients for parameter interpolation
     *      (default: \c nullptr)
     *
     * /return (weight, mu_o, phi_o, pdf)
     *     weight:
     *         Importance weight (i.e. the value of the BSDF divided by
     *         the pdf) for the RGB channels.
     *     mu_o:
     *         The sampled outgoing elevation angle cosine
     *     phi_o:
     *         The sampled outoing azimuthal angle
     *     pdf:
     *         The probability density function of the sampling scheme
     *
     * \remark From Python, this function is called using the syntax
     *         <tt>weight, mu_o, phi_o, pdf = storage.sample(mu_i, phi_i, sample)</tt>
     *         where <tt>sample</tt> can be an array.
     *         The Python API lacks the \c basis_coeffs parameter.
     */
    std::tuple<Color3f, float, float, float> sample(float mu_i, float phi_i, const Point2f &sample,
                                                    const float *basis_coeffs = nullptr) const;

    /**
     * \brief Evaluate the probability density of the sampling scheme implemented by \ref sample().
     *
     * \param mu_i
     *      The incident elevation angle cosine
     * \param mu_o
     *      The outgoing elevation angle cosine
     * \param phi_i
     *      The incident azimuthal angle
     * \param phi_o
     *      The outgoing azimuthal angle
     * \param basis_coeffs
     *      Basis coefficients for parameter interpolation
     *      (default: \c nullptr)
     * \remark
     *      The Python API lacks the \c basis_coeffs parameter.
     * \return
     *      The pdf value for given angles.
     */
    float pdf(float mu_i, float mu_o, float phi_i, float phi_o,
              const float *basis_coeffs = nullptr) const;

    /// For debugging: return a 2D Fourier series for the given parameters (in phi_s, phi_d format)
    void fourier_slice(int i, int o, int channel, MatrixX &coeffs) const;

    /// For debugging: return a interpolated 2D Fourier series for the given parameters (in phi_s, phi_d format)
    void fourier_slice_interpolated(float mu_i, float mu_o, int channel, MatrixX &coeffs) const;

    /// Forcefully release all resources
    void close() { m_mmap = nullptr; m_header = nullptr; m_coeffs = nullptr; m_cdf_mu = nullptr; m_nodes = nullptr; }

    /// Return statistics
    std::string stats() const;

    /// Return a string representation
    std::string to_string() const override;

    /// Virtual destructor
    virtual ~BSDFStorage();

    /// Create a BSDF storage file from a Layer data structure (monochromatic)
    static BSDFStorage *from_layer(const fs::path &filename,
                                   const Layer *layer,
                                   const std::string &metadata = "", float error = 1e-3) {
        const Layer *layers[1] = { layer };
        return BSDFStorage::from_layer_general(filename, layers, 1, 1, 0, NULL, NULL, metadata, error);
    }

    /// Create a BSDF storage file from three Layer data structures (RGB)
    static BSDFStorage *from_layer_rgb(const fs::path &filename,
                                       const Layer *layer_r, const Layer *layer_g, const Layer *layer_b,
                                       const std::string &metadata = "", float error = 1e-3) {
        const Layer *layers[3] = { layer_r, layer_g, layer_b };
        return BSDFStorage::from_layer_general(filename, layers, 3, 1, 0, NULL, NULL, metadata, error);
    }

    /// Create a BSDF storage file from three Layer data structures (most general interface)
    static BSDFStorage *from_layer_general(const fs::path &filename,
                                           const Layer **layers,
                                           size_t n_channels, size_t b_bases = 1, size_t n_parameters = 0,
                                           const size_t *param_sample_counts = NULL, const float **param_sample_positions = NULL,
                                           const std::string &metadata = "", float error = 1e-3);

protected:
    struct Header {
        uint8_t  identifier[7];      // Set to 'SCATFUN'
        uint8_t  version;            // Currently version is 2
        uint32_t flags;              // 0x01: file contains a BSDF
        uint32_t n_nodes;            // Number of samples in the elevational discretization

        uint32_t n_coeffs;           // Total number of Fourier series coefficients stored in the file
        uint32_t n_max_order_s;      // Coeff. count for the longest series occuring in the file (phi_s dimension)
        uint32_t n_max_order_d;      // Coeff. count for the longest series occuring in the file (phi_d dimension)
        uint32_t n_channels;         // Number of color channels (usually 1 or 3)
        uint32_t n_bases;            // Number of BSDF basis functions (relevant for texturing)

        uint32_t n_metadata_bytes;   // Size of descriptive metadata that follows the BSDF data
        uint32_t n_parameters;       // Number of textured material parameters
        uint32_t n_parameter_values; // Total number of BSDF samples for all textured parameters
        float    eta;                // Relative IOR through the material (eta(bottom) / eta(top))

        float    alpha[2];           // Beckmann-equiv. roughness on the top (0) and bottom (1) side
        float    unused[1];          // Unused fields to pad the header to 64 bytes

        float    data[0];            // BSDF data starts here
    };

    /// Create a new BSDF storage file for the given amount of coefficients etc
    BSDFStorage(const fs::path &filename,
                size_t n_nodes, size_t n_channels,
                size_t n_max_order_s, size_t n_max_order_d, size_t n_coeffs,
                size_t n_bases = 1, size_t n_parameters = 0,
                const size_t *param_sample_counts = nullptr,
                const float **param_sample_positions = nullptr,
                const std::string &metadata = "");

    /// Return a pointer to the underlying sparse offset table
    OffsetType *offset_table(size_t o = 0, size_t i = 0) {
        return m_offset_table + 3*(i * node_count() + o);
    }

    /// Return a pointer to the underlying sparse offset table (const version)
    const OffsetType *offset_table(size_t o = 0, size_t i = 0) const {
        return m_offset_table + 3*(i * node_count() + o);
    }

    /// Return the sparse data offset for the given incident and exitant angle pair
    float *coeff_data(size_t o, size_t i) {
        return m_coeffs + offset_table(o, i)[0];
    }

    /// Return the sparse data offset for the given incident and exitant angle pair (const version)
    const float *coeff_data(size_t o, size_t i) const {
        return m_coeffs + offset_table(o, i)[0];
    }

    /// Return the sparse data offset and Fourier dimensions for the given incident and exitant angle pair
    std::tuple<float *, size_t, size_t> coeff_data_and_sizes(size_t o, size_t i, size_t basis = 0, size_t channel = 0) {
        OffsetType *offset_table_ptr = offset_table(o, i);
        OffsetType offset = offset_table_ptr[0],
                   ms     = offset_table_ptr[1],
                   md     = offset_table_ptr[2];
        float *coeff_ptr = m_coeffs + offset + basis*ms*md + channel*basis_count()*ms*md;
        return std::make_tuple(coeff_ptr, ms, md);
    }

    /// Return the sparse data offset and Fourier dimensions for the given incident and exitant angle pair (const version)
    std::tuple<const float *, size_t, size_t> coeff_data_and_sizes(size_t o, size_t i, size_t basis = 0, size_t channel = 0) const {
        const OffsetType *offset_table_ptr = offset_table(o, i);
        OffsetType offset = offset_table_ptr[0],
                   ms     = offset_table_ptr[1],
                   md     = offset_table_ptr[2];
        const float *coeff_ptr = m_coeffs + offset + basis*ms*md + channel*basis_count()*ms*md;
        return std::make_tuple(coeff_ptr, ms, md);
    }

    /// Return the Fourier dimensions (in s and d) for the given incident and exitant angle pair
    std::pair<size_t, size_t> fourier_dimensions(size_t o = 0, size_t i = 0) const {
        const OffsetType *offset_table_ptr = offset_table(o, i);
        size_t ms = offset_table_ptr[1],
               md = offset_table_ptr[2];
        return std::make_pair(ms, md);
    }

    /// Return the number of Fourier modes in the sparse \mu_i and \mu_o storage
    void interpolated_fourier_dimensions(ssize_t offset_o, ssize_t offset_i, float weights_o[4], float weights_i[4],
                                         size_t ms[16], size_t md[16],
                                         size_t *ms_max, size_t *md_max) const {
        *ms_max = 0; *md_max = 0;
        for (int i = 0; i < 4; ++i) {
            for (int o = 0; o < 4; ++o) {
                float weight = weights_o[o] * weights_i[i];
                if (weight == 0) continue;

                size_t s, d;
                std::tie(s, d) = fourier_dimensions(offset_o + o, offset_i + i);
                ms[4*i + o] = s;
                md[4*i + o] = d;
                *ms_max = std::max(*ms_max, s);
                *md_max = std::max(*md_max, d);
            }
        }
    }

    /// Use spline interpolation over coefficient tables of a 2D Fourier series
    void interpolate_fourier_series(int offset_o, int offset_i, float weights_o[4], float weights_i[4],
                                    size_t modes_s[16], size_t modes_d[16],
                                    size_t ms_max, size_t md_max,
                                    float *out[3], size_t n_channels, const float *basis_coeffs) const {
        size_t md_max_padded = pad<FloatP::Size>(md_max) * FloatP::Size;
        size_t ms_max_half = ms_max / 2;

        for (int i = 0; i < 4; ++i) {
            for (int o = 0; o < 4; ++o) {
                float weight = weights_o[o] * weights_i[i];
                if (weight == 0)
                    continue;

                size_t ms = modes_s[4*i + o], md = modes_d[4*i + o];
                if (ms*md <= 0)
                    continue;
                int msh = (int) ms / 2;

                const float *source = coeff_data(offset_o + o, offset_i + i);

                size_t n_bases = basis_count();
                for (size_t channel = 0; channel < n_channels; ++channel) {
                    float *target = out[channel];
                    for (size_t basis = 0; basis < n_bases; ++basis) {
                        float interpolation_weight = weight * basis_coeffs[channel*n_bases + basis];
                        if (interpolation_weight == 0) {
                            source += ms*md;
                            continue;
                        }

                        for (int s = -msh; s <= msh; ++s) {
                            int os = s + (int)ms_max_half;
                            for (size_t d = 0; d < md; ++d) {
                                target[os*md_max_padded + d] += *source++ * interpolation_weight;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Use spline interpolation over coefficients of a Cosine series
    void interpolate_marginal_cosine_series(int offset_i, float weights_i[4], size_t index, float *marginal_coeffs,
                                            float *out, const float *basis_coeffs) const {
        size_t n_bases = basis_count(),
               n_nodes = node_count(),
               n_marginals = marginal_order_count();

        for (int i = 0; i < 4; ++i) {
            float weight = weights_i[i];
            if (weight == 0)
                continue;

            for (size_t basis = 0; basis < n_bases; ++basis) {
                float interpolation_weight = weight * basis_coeffs[basis];
                float *marginal_ptr = marginal_coeffs + index*n_nodes*n_bases*n_marginals +
                                                       (offset_i + i)*n_bases*n_marginals +
                                                                        basis*n_marginals;
                for (size_t j = 0; j < n_marginals; ++j) {
                    out[j] += interpolation_weight * marginal_ptr[j];
                }
            }
        }
    }

    /**
     * \brief Importance sample outgoing elevation angle cosine.
     *
     * \param mu_i
     *      The incident elevation angle cosine
     * \param phi_i
     *      The incident azimuth angle
     * \param sample
     *      A uniformly distributed random sample in the interval <tt>[0,1]</tt>
     * \basis_coeffs
     *      Basis coefficients for parameter interpolation
     * \return
     *      1. The sampled angle mu_o
     *      2. The probability density of the sampled angle
     */
    std::tuple<float, float>
    sample_elevation_angle(float mu_i, float phi_i,
                           float sample, const float *basis_coeffs) const;

    /**
     * \brief Importance sample azimuthal difference angle.
     *
     * \param coeffs
     *    Coefficient storage
     * \param ms
     *    Denotes the size of \c coeffs in phi_s dimension
     * \param md
     *    Denotes the size of \c coeffs in phi_d dimension
     * \param phi_i
     *      The incident azimuth angle
     * \param sample
     *      A uniformly distributed random sample in the interval <tt>[0,1]</tt>
     * \return
     *      1. The importance weight (i.e. the value of the
     *         Fourier series divided by pdf)
     *      2. The sampled angle phi_d
     *      3. The probability density of the sampled angle
     */
    std::tuple<float, float, float>
    sample_azimuthal_difference_angle(const float *coeffs, size_t ms, size_t md,
                                      float phi_i, float sample) const;

    MTS_DECLARE_CLASS()

protected:
    fs::path m_filename;
    ref<MemoryMappedFile> m_mmap;
    Header *m_header;
    float *m_nodes;
    uint32_t *m_param_sample_counts;
    float *m_param_sample_positions;
    float *m_cdf_mu;
    OffsetType *m_offset_table;
    float *m_coeffs;
    std::string m_metadata;

    float **m_param_sample_positions_nested;
    float *m_pdf_mu;
    float *m_reciprocals;
};

NAMESPACE_END(mitsuba)
