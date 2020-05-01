#include <mitsuba/layer/storage.h>
#include <mitsuba/layer/fourier.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/spline.h>
#include <mitsuba/core/filesystem.h>
#include <tbb/tbb.h>
#include <fstream>

NAMESPACE_BEGIN(mitsuba)

#define BSDF_STORAGE_HEADER_ID          "SCATFUN"
#define BSDF_STORAGE_VERSION            2
#define BSDF_STORAGE_FLAGS_BSDF         1
#define BSDF_STORAGE_HEADER_SIZE        64

static const float __basis_coeffs_default[3] = { 1.0, 1.0, 1.0 };

BSDFStorage::BSDFStorage(const fs::path &filename, bool read_only)
        : m_filename(filename), m_header(nullptr),
          m_param_sample_positions_nested(nullptr), m_pdf_mu(nullptr), m_reciprocals(nullptr) {

    static_assert(sizeof(Header) == BSDF_STORAGE_HEADER_SIZE, "Header size mismatch!");

    m_mmap = new MemoryMappedFile(filename, read_only);
    if (m_mmap->size() < sizeof(Header))
        Throw("BSDF storage file \"%s\" has a truncated header!", filename);

    m_header = (Header *) m_mmap->data();
    const char *id = BSDF_STORAGE_HEADER_ID;
    const size_t len = strlen(BSDF_STORAGE_HEADER_ID);
    if (memcmp(id, m_header->identifier, len) != 0)
        Throw("BSDF storage file \"%s\" has a corrupt header!", filename.string().c_str());

    size_t
        n_nodes = m_header->n_nodes,
        n_max_order_s = m_header->n_max_order_s,
        n_max_order_d = m_header->n_max_order_d,
        n_channels = m_header->n_channels,
        n_bases = m_header->n_bases,
        n_parameters = m_header->n_parameters,
        n_coeffs = m_header->n_coeffs,
        n_parameter_values = m_header->n_parameter_values,
        n_metadata_bytes = m_header->n_metadata_bytes;

    size_t n_marginal_orders = n_max_order_s / 2 + 1;
    size_t n_total_marginal_coeffs = n_marginal_orders*n_nodes*n_nodes*n_bases;

    size_t size = BSDF_STORAGE_HEADER_SIZE +    // Header
        sizeof(float)*n_nodes +                 // Node locations
        sizeof(uint32_t)*n_parameters +         // Parameter sample counts
        sizeof(float)*n_parameter_values +      // Parameter sample positions
        sizeof(float)*n_total_marginal_coeffs + // CDF Fourier coefficients in \mu
        sizeof(OffsetType)*n_nodes*n_nodes*3 +  // Offset + size table
        sizeof(float)*n_coeffs +                // Fourier coefficients
        n_metadata_bytes;                       // Metadata

    size_t uncompressed_size = size - sizeof(float)*n_coeffs
        + n_nodes*n_nodes*n_channels*n_bases*n_max_order_s*n_max_order_d*sizeof(float);

    if (m_mmap->size() != size)
        Throw("BSDF storage file \"%s\" has an invalid size! (it is potentially truncated)", filename);

    Log(Debug, "Mapped sparse BSDF storage file \"%s\" into memory:", filename);
    Log(Debug, "  Discretizations in mu  : %d", n_nodes);
    Log(Debug, "  Max. Fourier orders    : %d x %d", n_max_order_s, n_max_order_d);
    Log(Debug, "  Color channels         : %d", n_channels);
    Log(Debug, "  Textured parameters    : %d", n_parameters);
    Log(Debug, "  Basis functions        : %d", n_bases);
    Log(Debug, "  Uncompressed size      : %s", util::mem_string(uncompressed_size));
    Log(Debug, "  Actual size            : %s (reduced to %.2f%%)",
        util::mem_string(size), 100 * size / (float) uncompressed_size);

    m_nodes = m_header->data;
    m_param_sample_counts = (uint32_t *) (m_nodes + n_nodes);
    m_param_sample_positions = (float *) (m_param_sample_counts + n_parameters);
    m_cdf_mu = (float *) m_param_sample_positions + n_parameter_values;
    m_offset_table = (OffsetType *) (m_cdf_mu + n_total_marginal_coeffs);
    m_coeffs = (float *) (m_offset_table + n_nodes*n_nodes*3);

    m_metadata.resize(n_metadata_bytes);
    memcpy(&m_metadata[0], m_coeffs + n_coeffs, n_metadata_bytes);

    size_t idx = 0;
    m_param_sample_positions_nested = new float*[n_parameters];
    for (size_t i = 0; i < n_parameters; ++i) {
        m_param_sample_positions_nested[i] = m_param_sample_positions + idx;
        idx += m_param_sample_counts[i];
    }

    m_pdf_mu = new float[n_total_marginal_coeffs];
    memset(m_pdf_mu, 0, n_total_marginal_coeffs*sizeof(float));

    for (size_t o = 0; o < n_nodes; ++o) {
        for (size_t basis = 0; basis < n_bases; ++basis) {
            for (size_t i = 0; i < n_nodes; ++i) {

                const float *coeffs;
                OffsetType ms, md;
                std::tie(coeffs, ms, md) = coeff_data_and_sizes(o, i, basis);

                int msh = (int) ms / 2;
                int orders = (int) ms / 2 + 1;

                float *pdf_ptr = m_pdf_mu + o*n_nodes*n_bases*n_marginal_orders +
                                                    i*n_bases*n_marginal_orders +
                                                        basis*n_marginal_orders;
                for (int j = 0; j < orders; ++j) {
                    int os = msh - j;
                    int od = j;

                    float value;
                    if (ms * md > 0) {
                        /* This 1D Fourier series has only real coefficients and
                           can be rewritten into a pure Cosine series */
                        value = ((j == 0) ? 1 : 2) * coeffs[os * md + od];
                    } else {
                        value = 0;
                    }
                    pdf_ptr[j] = value;
                }
            }
        }
    }

    size_t n_reciprocals = (n_max_order_s + 2*(n_max_order_d - 1)) / 2 + 1;
    size_t n_reciprocals_padded = (1 + pad<Float32P::Size>(n_reciprocals)) * Float32P::Size;
    m_reciprocals = new float[n_reciprocals_padded];
    std::fill(m_reciprocals, m_reciprocals + n_reciprocals_padded, 1.0f);
    for (uint32_t i = 0; i < n_reciprocals; ++i) {
        m_reciprocals[i] = 1.0f / (float) i;
    }
}

BSDFStorage::BSDFStorage(const fs::path &filename,
                         size_t n_nodes, size_t n_channels,
                         size_t n_max_order_s, size_t n_max_order_d, size_t n_coeffs,
                         size_t n_bases, size_t n_parameters,
                         const size_t *param_sample_counts, const float **param_sample_positions,
                         const std::string &metadata_string)
        : m_filename(filename), m_header(nullptr),
          m_param_sample_positions_nested(nullptr), m_pdf_mu(nullptr), m_reciprocals(nullptr) {

    if (n_channels != 1 && n_channels != 3)
        Throw("Only 1 and 3-channel files are supported!");

    size_t n_bases_pred = 1, n_parameter_values = 0;

    for (size_t i = 0; i < n_parameters; ++i) {
        n_parameter_values += param_sample_counts[i];
        n_bases_pred *= param_sample_counts[i];
    }

    if (n_bases_pred != n_bases)
        Throw("BSDFStorage::BSDFStorage(): provided an invalid number of basis functions!");

    size_t n_marginal_orders = n_max_order_s / 2 + 1;
    size_t n_total_marginal_coeffs = n_marginal_orders*n_nodes*n_nodes*n_bases;

    size_t size = BSDF_STORAGE_HEADER_SIZE +    // Header
        sizeof(float)*n_nodes +                 // Node locations
        sizeof(uint32_t)*n_parameters +         // Parameter sample counts
        sizeof(float)*n_parameter_values +      // Parameter sample positions
        sizeof(float)*n_total_marginal_coeffs + // CDF Fourier coefficients in \mu
        sizeof(OffsetType)*n_nodes*n_nodes*3 +  // Offset + size table
        sizeof(float)*n_coeffs +                // Fourier coefficients
        metadata_string.size();                 // Metadata

    size_t uncompressed_size = size - sizeof(float)*n_coeffs
        + n_nodes*n_nodes*n_channels*n_bases*n_max_order_s*n_max_order_d*sizeof(float);

    Log(Debug, "Creating sparse BSDF storage file \"%s\":", filename);
    Log(Debug, "  Discretizations in mu  : %d", n_nodes);
    Log(Debug, "  Max. Fourier orders    : %d x %d", n_max_order_s, n_max_order_d);
    Log(Debug, "  Color channels         : %d", n_channels);
    Log(Debug, "  Textured parameters    : %d", n_parameters);
    Log(Debug, "  Basis functions        : %d", n_bases);
    Log(Debug, "  Uncompressed size      : %s", util::mem_string(uncompressed_size));
    Log(Debug, "  Actual size            : %s (reduced to %.2f%%)",
        util::mem_string(size), 100 * size / (float) uncompressed_size);

    m_mmap = new MemoryMappedFile(filename, size);
    m_header = (Header *) m_mmap->data();

    const char *id = BSDF_STORAGE_HEADER_ID;

    const size_t len = strlen(BSDF_STORAGE_HEADER_ID);
    for (size_t i = 0; i < len; ++i)
        m_header->identifier[i] = id[i];

    m_header->version = BSDF_STORAGE_VERSION;
    m_header->flags = 0;
    m_header->flags |= BSDF_STORAGE_FLAGS_BSDF;
    m_header->n_nodes = (uint32_t) n_nodes;
    m_header->n_parameters = (uint16_t) n_parameters;
    m_header->n_max_order_s = (uint32_t) n_max_order_s;
    m_header->n_max_order_d = (uint32_t) n_max_order_d;
    m_header->n_channels = (uint32_t) n_channels;
    m_header->n_bases = (uint32_t) n_bases;
    m_header->n_parameter_values = (uint16_t) n_parameter_values;
    m_header->n_coeffs = (uint32_t) n_coeffs;
    m_header->n_metadata_bytes = (uint32_t) metadata_string.size();
    m_header->eta = 1.0f; // default

    m_nodes = m_header->data;
    m_param_sample_counts = (uint32_t *) (m_nodes + n_nodes);
    m_param_sample_positions = (float *) (m_param_sample_counts + n_parameters);
    m_cdf_mu = (float *) m_param_sample_positions + n_parameter_values;
    m_offset_table = (OffsetType *) (m_cdf_mu + n_total_marginal_coeffs);
    m_coeffs = (float *) (m_offset_table + n_nodes*n_nodes*3);

    memcpy(m_coeffs + n_coeffs, metadata_string.c_str(), metadata_string.size());
    m_metadata = metadata_string;

    size_t idx = 0;
    m_param_sample_positions_nested = new float*[n_parameters];
    for (size_t i = 0; i < n_parameters; ++i) {
        m_param_sample_counts[i] = (uint32_t) param_sample_counts[i];
        m_param_sample_positions_nested[i] = m_param_sample_positions + idx;
        for (size_t j = 0; j < m_param_sample_counts[i]; ++j)
            m_param_sample_positions[idx++] = (float) param_sample_positions[i][j];
    }

    m_pdf_mu = new float[n_total_marginal_coeffs];
    memset(m_pdf_mu, 0, n_total_marginal_coeffs*sizeof(float));

    size_t n_reciprocals = (n_max_order_s + 2*(n_max_order_d - 1)) / 2 + 1;
    size_t n_reciprocals_padded = (1 + pad<Float32P::Size>(n_reciprocals)) * Float32P::Size;
    m_reciprocals = new float[n_reciprocals_padded];
    std::fill(m_reciprocals, m_reciprocals + n_reciprocals_padded, 1.0f);
    for (uint32_t i = 0; i < n_reciprocals; ++i) {
        m_reciprocals[i] = 1.0f / (float) i;
    }
}

BSDFStorage::~BSDFStorage() {
    if (m_param_sample_positions_nested)
        delete[] m_param_sample_positions_nested;
    if (m_pdf_mu)
        delete[] m_pdf_mu;
    if (m_reciprocals)
        delete[] m_reciprocals;
}

Color3f BSDFStorage::eval(float mu_i, float mu_o, float phi_i, float phi_o,
                          const float *basis_coeffs, bool clamp) const {
    if (!basis_coeffs) {
        assert(basis_count() == 1);
        basis_coeffs = __basis_coeffs_default;
    }

    /* Determine offsets and weights for \mu_i and \mu_o */
    float knot_weights_o[4], knot_weights_i[4];
    auto spline_weights_o = spline::eval_spline_weights(m_nodes, m_header->n_nodes, mu_o, knot_weights_o);
    auto spline_weights_i = spline::eval_spline_weights(m_nodes, m_header->n_nodes, mu_i, knot_weights_i);
    if (!spline_weights_o.first || !spline_weights_i.first)
        return 0;

    ssize_t knot_offset_o = spline_weights_o.second,
            knot_offset_i = spline_weights_i.second;
    size_t n_channels = channel_count();

    /* Allocate storage buffer to accumulate Fourier coefficients */
    size_t ms[16], md[16];
    size_t ms_max, md_max;
    interpolated_fourier_dimensions(knot_offset_o, knot_offset_i, knot_weights_o, knot_weights_i,
                                    ms, md, &ms_max, &md_max);
    if (ms_max * md_max <= 0)
        return Color3f(0);

    size_t md_max_padded = pad<Float32P::Size>(md_max);
    float *coeffs[3];
    for (size_t c = 0; c < n_channels; ++c)
        coeffs[c] = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, ms_max * md_max_padded, true);

    /* Accumulate weighted sums of nearby Fourier series over \mu_i and \mu_o discretization */
    interpolate_fourier_series(knot_offset_o, knot_offset_i, knot_weights_o, knot_weights_i,
                               ms, md, ms_max, md_max,
                               coeffs, n_channels, basis_coeffs);

    /* Evaluate Fourier series for angles \phi_s, \phi_d */
    float phi_s = phi_o + phi_i;
    float phi_d = phi_o - phi_i;
    Color3f result;
    if (n_channels == 1) {
        result = Color3f(eval_fourier_series_2d(coeffs[0], ms_max, md_max, phi_s, phi_d));
    } else {
        result = eval_3_fourier_series_2d(coeffs, ms_max, md_max, phi_s, phi_d);
    }
    return clamp ? max(0, result) : result;
}

/// Importance sample the model
std::tuple<Color3f, float, float, float> BSDFStorage::sample(float mu_i, float phi_i, const Point2f &sample,
                                                             const float *basis_coeffs) const {
    if (!basis_coeffs) {
        assert(basis_count() == 1);
        basis_coeffs = __basis_coeffs_default;
    }

    /* Sample outgoing elevation angle */
    float mu_o, pdf_mu;
    std::tie(mu_o, pdf_mu) = sample_elevation_angle(mu_i, phi_i, sample.y(), basis_coeffs);

    /* Determine offsets and weights for \mu_i and \mu_o */
    float knot_weights_o[4], knot_weights_i[4];
    auto spline_weights_o = spline::eval_spline_weights(m_nodes, m_header->n_nodes, (float)mu_o, knot_weights_o);
    auto spline_weights_i = spline::eval_spline_weights(m_nodes, m_header->n_nodes, (float)mu_i, knot_weights_i);
    if (!spline_weights_o.first || !spline_weights_i.first)
        return std::make_tuple(Color3f(0), mu_o, 0, 0);

    ssize_t knot_offset_o = spline_weights_o.second,
            knot_offset_i = spline_weights_i.second;
    size_t n_channels = channel_count();

    /* Allocate storage buffer to accumulate Fourier coefficients */
    size_t ms[16], md[16];
    size_t ms_max, md_max;
    interpolated_fourier_dimensions(knot_offset_o, knot_offset_i, knot_weights_o, knot_weights_i, ms, md, &ms_max, &md_max);

    size_t md_max_padded = pad<Float32P::Size>(md_max);
    float *coeffs[3];
    for (size_t c = 0; c < n_channels; ++c)
        coeffs[c] = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, ms_max * md_max_padded, true);

    /* Accumulate weighted sums of nearby Fourier series over \mu_i and \mu_o discretization */
    interpolate_fourier_series(knot_offset_o, knot_offset_i, knot_weights_o, knot_weights_i,
                               ms, md, ms_max, md_max,
                               coeffs, n_channels, basis_coeffs);

    /* Importance sample the luminance Fourier expansion to find phi_d */
    float y, phi_d, pdf_phi;
    std::tie(y, phi_d, pdf_phi) = sample_azimuthal_difference_angle(coeffs[0], ms_max, md_max,
                                                                    phi_i, sample.x());

    float phi_o = phi_i + phi_d;
    float pdf = std::max(0.0f, pdf_mu * pdf_phi);
    if (pdf == 0.0f)
        return std::make_tuple(Color3f(0), mu_o, phi_o, pdf);

    Color3f result;
    if (n_channels == 1) {
        result = Color3f(y);
    } else {
        /* Evaluate remaining Fourier series for angles \phi_s, \phi_d */
        float phi_s = phi_i + phi_o;
        float r = eval_fourier_series_2d(coeffs[1], ms_max, md_max, phi_s, phi_d);
        float b = eval_fourier_series_2d(coeffs[2], ms_max, md_max, phi_s, phi_d);
        float g = 1.39829f*y - 0.100913f*b - 0.297375f*r;
        result = Color3f(r, g, b);
    }
    Color3f weight = max(0, result / pdf);
    return std::make_tuple(weight, mu_o, phi_o, pdf);
}

float BSDFStorage::pdf(float mu_i, float mu_o, float phi_i, float phi_o, const float *basis_coeffs) const {
    if (!basis_coeffs) {
        assert(basis_count() == 1);
        basis_coeffs = __basis_coeffs_default;
    }

    /* Determine offsets and weights for \mu_i and \mu_o */
    float knot_weights_o[4], knot_weights_i[4];
    auto spline_weights_o = spline::eval_spline_weights(m_nodes, m_header->n_nodes, mu_o, knot_weights_o);
    auto spline_weights_i = spline::eval_spline_weights(m_nodes, m_header->n_nodes, mu_i, knot_weights_i);
    if (!spline_weights_o.first || !spline_weights_i.first)
        return 0;

    ssize_t knot_offset_o = spline_weights_o.second,
            knot_offset_i = spline_weights_i.second;

    /* Allocate storage buffer to accumulate Fourier coefficients */
    size_t ms[16], md[16];
    size_t ms_max, md_max;
    interpolated_fourier_dimensions(knot_offset_o, knot_offset_i, knot_weights_o, knot_weights_i, ms, md, &ms_max, &md_max);

    size_t md_max_padded = pad<Float32P::Size>(md_max);
    float *coeffs = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, ms_max * md_max_padded, true);

    /* Accumulate weighted sums of nearby Fourier series over \mu_i and \mu_o discretization */
    interpolate_fourier_series(knot_offset_o, knot_offset_i, knot_weights_o, knot_weights_i,
                               ms, md, ms_max, md_max,
                               &coeffs, 1, basis_coeffs);

    /* Evaluate Fourier series for angles \phi_s, \phi_d */
    float phi_s = phi_o + phi_i;
    float phi_d = phi_o - phi_i;
    float p = eval_fourier_series_2d(coeffs, ms_max, md_max, phi_s, phi_d);

    /* Compute normalization factor \rho */
    size_t n_marginal_orders = marginal_order_count();
    size_t n_marginal_orders_padded = pad<Float32P::Size>(n_marginal_orders);
    float *marginal_coeffs = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, n_marginal_orders_padded, true);
    interpolate_marginal_cosine_series(knot_offset_i, knot_weights_i,
                                       node_count()-1, m_cdf_mu,
                                       marginal_coeffs, basis_coeffs);

    float rho = 2 * math::Pi<float> * eval_cosine_series_1d(marginal_coeffs, n_marginal_orders, 2.0f * phi_i); // Account for different periodicity
    return std::max(0.0f, p / rho);
}

std::tuple<float, float>
BSDFStorage::sample_elevation_angle(float mu_i, float phi_i,
                                    float sample, const float *basis_coeffs) const {
    /* Determine offset and coefficients for mu_i */
    float knot_weights_i[4];
    auto spline_weights_i = spline::eval_spline_weights(m_nodes, m_header->n_nodes, (float)mu_i, knot_weights_i);
    if (!spline_weights_i.first)
        return std::make_tuple(0, 0);

    ssize_t knot_offset_i = spline_weights_i.second;

    /* Allocate storage to accumulate coefficients */

    size_t n_marginal_orders = marginal_order_count();
    size_t n_marginal_orders_padded = pad<Float32P::Size>(n_marginal_orders);
    float *marginal_coeffs = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, n_marginal_orders_padded, false);

    /* Define a lambda function to interpolate table entries */
    auto interpolate = [&](float *array, int index) {
        /* Compute Fourier series coefficients in phi parameter */
        memset(marginal_coeffs, 0, n_marginal_orders_padded * sizeof(Float32P));

        interpolate_marginal_cosine_series(knot_offset_i, knot_weights_i,
                                           index, array,
                                           marginal_coeffs, basis_coeffs);

        return std::max(0.0f, eval_cosine_series_1d(marginal_coeffs, n_marginal_orders, 2.0f * phi_i));   // Account for different periodicity
    };

    const float eps        = 1e-5f,
                last       = interpolate(m_cdf_mu, m_header->n_nodes - 1),
                eps_domain = std::abs(eps * (interpolate(m_pdf_mu, m_header->n_nodes - 1) - interpolate(m_pdf_mu, 0))),
                eps_value  = std::abs(eps * last),
                last_rcp   = 1.0f / last;
    /* Scale by the definite integral of the function (in case it is not normalized) */
    sample *= last;

    if (last <= 0)
        return std::make_tuple(0, 0);

    /* Map sample to a spline interval by searching through the monotonic cdf array */
    size_t idx = math::find_interval(m_header->n_nodes,
        [&](size_t idx) {
            return interpolate(m_cdf_mu, idx) <= sample;
        }
    );

    /* Look up node positions and interpolated function values */
    float  f0    = interpolate(m_pdf_mu, idx),
           f1    = interpolate(m_pdf_mu, idx+1),
           x0    = m_nodes[idx],
           x1    = m_nodes[idx+1],
           width = x1 - x0;

    /* Approximate derivatives using finite differences of the interpolant */
    float d0, d1;
    if (idx > 0)
        d0 = width * (f1 - interpolate(m_pdf_mu, idx-1)) / (x1 - m_nodes[idx-1]);
    else
        d0 = f1 - f0;

    if (idx + 2 < m_header->n_nodes)
        d1 = width * (interpolate(m_pdf_mu, idx+2) - f0) / (m_nodes[idx+2] - x0);
    else
        d1 = f1 - f0;

    /* Re-scale the sample after having choosen the interval */
    sample = (sample - interpolate(m_cdf_mu, idx)) / width;

    /* Importance sample linear interpolant as initial guess for t */
    float t;
    if (f0 != f1)
        t = (f0 - safe_sqrt(f0 * f0 + 2 * sample * (f1 - f0))) / (f0 - f1);
    else
        t = sample / f0;

    float a = 0, b = 1, value, deriv;
    int iterations = 0;
    do {
        /* Fall back to a bisection step when t is out of bounds */
        if (!(t > a && t < b)) t = 0.5f * (a + b);

        /* Evaluate the definite integral and its derivative
           (i.e. the spline) */
        std::tie(value, deriv) = spline::eval_spline_i(f0, f1, d0, d1, t);

        value -= sample;

        /* Stop the iteration if converged */
        if (abs(value) <= eps_value || (b - a) <= eps_domain || iterations > 10000) break;

        /* Update the bisection bounds */
        if (value <= 0)
            a = t;
        else
            b = t;

        /* Perform a Newton step */
        t -= value / deriv;
        iterations++;
    } while (true);

    /* Return the value and PDF if requested */
    return std::make_tuple(
        x0 + width * t,
        deriv * last_rcp);
}

std::tuple<float, float, float>
BSDFStorage::sample_azimuthal_difference_angle(const float *coeffs, size_t ms, size_t md,
                                               float phi_i, float sample) const {
    int msh = (int) ms / 2,
        mdh = (int) md - 1;

    /* Compute Fourier series coefficients in \phi_d parameter given a fixed \phi_i */
    int mh = msh + mdh;
    int m = 2*mh+1;
    if (m <= 0)
        return std::make_tuple(0, 0, 0);

    float *coeffs_re = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, 1+pad<Float32P::Size>(m), true);
    float *coeffs_im = (float *) ENOKI_ALIGNED_ALLOCA(Float32P, 1+pad<Float32P::Size>(m), true);
    coeffs_re += mh;
    coeffs_im += mh;

    auto sc_i = enoki::sincos(phi_i * float(-2 * msh));
    auto inc_i = enoki::sincos(2*phi_i);

    size_t pd = pad<Float32P::Size>(md);

    const float *source = coeffs;
    for (int s = -msh; s <= msh; ++s) {
        for (size_t d = 0; d < pd; ++d) {
            size_t od = d * Float32P::Size;

            Float32P reP     = enoki::load_unaligned<Float32P>(coeffs_re + s + od);
            Float32P imP     = enoki::load_unaligned<Float32P>(coeffs_im + s + od);
            Float32P coeffsP = enoki::load_unaligned<Float32P>(source);

            reP += sc_i.second * coeffsP;
            imP += sc_i.first * coeffsP;

            enoki::store_unaligned<Float32P>(coeffs_re + s + od, reP);
            enoki::store_unaligned<Float32P>(coeffs_im + s + od, imP);
            source += Float32P::Size;
        }

        sc_i = trig_addition(sc_i, inc_i);
    }

    sc_i = trig_subtraction(sc_i, inc_i);

    auto flip = -enoki::arange<Int32P>() + Int32P(Float32P::Size-1);

    for (int s = msh; s > 0; --s) {
        size_t os = -s + msh;
        size_t pd_sym = pad<Float32P::Size>(s);
        source = coeffs + os * pd * Float32P::Size + 1;

        for (size_t d = 0; d < pd_sym; ++d) {
            Float32P reP     = enoki::load_unaligned<Float32P>(coeffs_re + s - (d+1) * Float32P::Size);
            Float32P imP     = enoki::load_unaligned<Float32P>(coeffs_im + s - (d+1) * Float32P::Size);
            Float32P coeffsP = enoki::gather<Float32P>(source, flip);

            reP += sc_i.second * coeffsP;
            imP += sc_i.first * coeffsP;

            enoki::store_unaligned<Float32P>(coeffs_re + s - (d+1) * Float32P::Size, reP);
            enoki::store_unaligned<Float32P>(coeffs_im + s - (d+1) * Float32P::Size, imP);
            source += Float32P::Size;
        }

        sc_i = trig_subtraction(sc_i, inc_i);
    }

    /* Declare bisection variables */
    int iterations = 0;
    float a      = 0.0f,
          b      = 2.0f*math::Pi<float>,
          phi_d  = math::Pi<float>,
          value  = 0.0f,
          deriv  = 0.0f,
          coeff0 = coeffs_re[0],
          y      = 2.0f*math::Pi<float>*coeff0*sample;

    while (true) {
        auto sc_d = enoki::sincos(phi_d * (1 + enoki::arange<Float32P>()));
        auto inc_d = enoki::sincos(Float32P::Size * phi_d);

        Float32P valueP = enoki::zero<Float32P>();
        Float32P derivP = enoki::zero<Float32P>();

        for (int d = 1; d <= mh; d += Float32P::Size) {
            auto reP    = enoki::load_unaligned<Float32P>(coeffs_re + d);
            auto imP    = enoki::load_unaligned<Float32P>(coeffs_im + d);
            auto recipP = enoki::load_unaligned<Float32P>(m_reciprocals + d);

            derivP += reP * sc_d.second - imP * sc_d.first;
            valueP += recipP * (imP * (sc_d.second - 1.0f) + reP * sc_d.first);

            sc_d = trig_addition(sc_d, inc_d);
        }
        deriv = 2.0f*enoki::hsum(derivP) + coeff0;
        value = 2.0f*enoki::hsum(valueP) + coeff0 * phi_d;

        value -= y;

        if (std::abs(value) <= 1e-5f * coeff0 || ++iterations > 20)
            break;
        else if (value > 0)
            b = phi_d;
        else
            a = phi_d;

        phi_d -= value / deriv;

        if (!(phi_d > a && phi_d < b))
            phi_d = 0.5f * (a + b);
    }

    float Y = deriv;
    float pdf = math::InvTwoPi<float> * Y / coeff0;

    return std::make_tuple(Y, phi_d, pdf);
}

size_t BSDFStorage::size() const {
    if (!m_mmap)
        return 0;
    return m_mmap->size();
}

size_t BSDFStorage::total_coeffs() const {
    size_t n_coeffs = m_header->n_coeffs;
    return n_coeffs;
}

std::string BSDFStorage::stats() const {
    std::ostringstream oss;

    size_t
        n_nodes = m_header->n_nodes,
        n_max_order_s = m_header->n_max_order_s,
        n_max_order_d = m_header->n_max_order_d,
        n_channels = m_header->n_channels,
        n_bases = m_header->n_bases,
        n_parameters = m_header->n_parameters,
        n_coeffs = m_header->n_coeffs,
        n_parameter_values = m_header->n_parameter_values,
        n_metadata_bytes = m_header->n_metadata_bytes;

    size_t n_marginal_orders = n_max_order_s / 2 + 1;
    size_t n_total_marginal_coeffs = n_marginal_orders*n_nodes*n_nodes*n_bases;

    size_t size = BSDF_STORAGE_HEADER_SIZE +    // Header
        sizeof(float)*n_nodes +                 // Node locations
        sizeof(uint32_t)*n_parameters +         // Parameter sample counts
        sizeof(float)*n_parameter_values +      // Parameter sample positions
        sizeof(float)*n_total_marginal_coeffs + // CDF Fourier coefficients in \mu
        sizeof(OffsetType)*n_nodes*n_nodes*3 +  // Offset + size table
        sizeof(float)*n_coeffs +                // Fourier coefficients
        n_metadata_bytes;                       // Metadata

    size_t uncompressed_size = size - sizeof(float)*n_coeffs
        + n_nodes*n_nodes*n_channels*n_bases*n_max_order_s*n_max_order_d*sizeof(float);

    oss.precision(2);
    oss << " Discretizations in mu  : " << n_nodes << std::endl;
    oss << " Max. Fourier orders    : " << n_max_order_s << " x " << n_max_order_d << std::endl;
    oss << " Color channels         : " << n_channels << std::endl;
    oss << " Textured parameters    : " << n_parameters << std::endl;
    oss << " Basis functions        : " << n_bases << std::endl;
    oss << " Uncompressed size      : " << util::mem_string(uncompressed_size) << std::endl;
    oss << " Actual size            : " << util::mem_string(size);
    oss << " (reduced to " << (100 * size / (float) uncompressed_size) << "%)";
    return oss.str();
}

std::string BSDFStorage::to_string() const {
    std::ostringstream oss;
    oss << "BSDFStorage[" << std::endl
        << "  mmap = " << string::indent(m_mmap->to_string()) << "," << std::endl;
    if (m_header) {
        oss << "  n_nodes = " << m_header->n_nodes << "," << std::endl
            << "  n_max_orders = " << m_header->n_max_order_s << "x"
                                   << m_header->n_max_order_d << "," << std::endl
            << "  n_channels = " << m_header->n_channels << "," << std::endl
            << "  n_bases = " << m_header->n_bases << "," << std::endl
            << "  eta = " << m_header->eta << std::endl;
    }
    oss << "]";
    return oss.str();
}

BSDFStorage *BSDFStorage::from_layer_general(const fs::path &filename,
                                             const Layer **layers,
                                             size_t n_channels, size_t n_bases, size_t n_parameters,
                                             const size_t *param_sample_counts, const float **param_sample_positions,
                                             const std::string &metadata, float error) {
    const Layer &layer = *layers[0];
    size_t n = layer.resolution(), h = n / 2;

    /* Insert an explicit mu=0 node to simplify the evaluation / sampling code */
    VectorX nodes = (VectorX(n + 2)
                     << layer.nodes().head(h).reverse(),
                     0, 0,
                     layer.nodes().tail(h)).finished();

    Log(Debug, "BSDFStorage::from_layer_general(): merging %d layers into \"%s\" "
        "- analyzing sparsity pattern..", n_bases * n_channels, filename);

    /* Determine how many coefficients would be needed for a dense storage */
    size_t n_coeffs_dense = nodes.size()*nodes.size()
                            *layer.fourier_orders().first*layer.fourier_orders().second
                            *n_bases*n_channels;

    /* Determine coefficient counts for offset table */
    size_t n_nodes = (size_t) nodes.size();
    std::atomic<size_t> ms_max_storage(0), md_max_storage(0);

    OffsetType *offset_table_data = new OffsetType[3*n_nodes*n_nodes];
    memset(offset_table_data, 0, 3*n_nodes*n_nodes*sizeof(OffsetType));

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n_nodes),
        [&](const tbb::blocked_range<size_t> &range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
            // for (size_t i = 0; i < n_nodes; ++i) {
                for (size_t o = 0; o < n_nodes; ++o) {
                    MatrixS::Index ip, op;
                    size_t offset = 3*(o + i * n_nodes);

                    if (i == h || i == h+1 || o == h || o == h+1) {
                        offset_table_data[offset + 0] = 0;
                        offset_table_data[offset + 1] = 0;
                        offset_table_data[offset + 2] = 0;
                        continue;
                    }

                    ip = (MatrixS::Index) (i > h ? (i-2) : i);
                    op = (MatrixS::Index) (o > h ? (o-2) : o);

                    size_t ms_max_io = 0, md_max_io = 0;
                    for (size_t basis = 0; basis < n_bases; ++basis) {
                        for (size_t ch = 0; ch < n_channels; ++ch) {
                            size_t ms_trunc, md_trunc;
                            layers[basis*n_channels + ch]->find_truncation(op, ip, (double)error, ms_trunc, md_trunc);

                            ms_max_io = std::max(ms_max_io, ms_trunc);
                            md_max_io = std::max(md_max_io, md_trunc);
                        }
                    }

                    /* Never use more coefficients than available in Layer */
                    assert(ms_max_io <= layer.fourier_orders().first);
                    assert(md_max_io <= layer.fourier_orders().second);

                    while (true) {
                        size_t tmp = ms_max_storage;
                        if (ms_max_io <= tmp || ms_max_storage.compare_exchange_strong(tmp, ms_max_io))
                            break;
                    }

                    while (true) {
                        size_t tmp = md_max_storage;
                        if (md_max_io <= tmp || md_max_storage.compare_exchange_strong(tmp, md_max_io))
                            break;
                    }

                    offset_table_data[offset + 0] = 0;   // offset into coeffs array, to be computed later.

                    if (ms_max_io * md_max_io == 0) {
                        offset_table_data[offset + 1] = 0;
                        offset_table_data[offset + 2] = 0;
                    }
                    else {
                        offset_table_data[offset + 1] = (OffsetType) ms_max_io;
                        offset_table_data[offset + 2] = (OffsetType) md_max_io / 2 + 1;  // Can only save part of phi_d dimension and then use symmetries
                    }
                }
            }
        });

    md_max_storage = md_max_storage / 2 + 1;    // Can only save part of phi_d dimension and then use symmetries

    /* Compute the offset table */
    size_t n_coeffs_sparse = 0;
    for (size_t i = 0; i < n_nodes*n_nodes; ++i) {
        offset_table_data[3*i] = (OffsetType) n_coeffs_sparse;
        n_coeffs_sparse += offset_table_data[3*i + 1]*offset_table_data[3*i + 2] * n_bases*n_channels;
    }

    Log(Debug, "Done. Number of coeff: %d" " / %d" ", sparsity=%.2f%%",
        n_coeffs_sparse, n_coeffs_dense, 100 * (float) n_coeffs_sparse / (float) n_coeffs_dense);

    BSDFStorage *storage = new BSDFStorage(filename,
                                           n_nodes, n_channels,
                                           ms_max_storage, md_max_storage, n_coeffs_sparse,
                                           n_bases, n_parameters,
                                           param_sample_counts,
                                           param_sample_positions,
                                           metadata);

    Log(Debug, "Copying data into sparse BSDF file ..");

    for (size_t i = 0; i < n_nodes; ++i)
        storage->m_nodes[i] = (float) nodes[i];

    memcpy(storage->offset_table(), offset_table_data, 3*n_nodes*n_nodes*sizeof(OffsetType));

    /* Do a huge matrix transpose */
    for (size_t i = 0; i < n_nodes; ++i) {
        for (size_t o = 0; o < n_nodes; ++o) {
            MatrixS::Index ip, op;

            float *coeffs;
            OffsetType ms, md;
            std::tie(coeffs, ms, md) = storage->coeff_data_and_sizes(o, i);

            if (i == h || o == h) {
                assert(ms*md == 0);
                continue;
            }

            if (ms*md == 0) {
                continue;
            }

            ip = (MatrixS::Index) (i < h ? (h-i-1) : (i-2));
            op = (MatrixS::Index) (o < h ? (h-o-1) : (o-2));

            double weight = std::abs(nodes[o] / (2.0 * math::Pi<double> * nodes[i] * layer.weights()[ip]));

            int msh = ms / 2;

            if (n_channels == 1) {
                for (size_t basis = 0; basis < n_bases; ++basis) {
                    for (int s = -msh; s <= msh; ++s) {
                        for (int d = 0; d < (int) md; ++d) {  // Can only save part of phi_d dimension and then use symmetries
                            float value = (float) ((*layers[basis]).scatter_coeff(s, d, op, ip) * weight);
                            if (!std::isfinite(value))
                                Log(Warn, "Encountered invalid data: %f", value);

                            *coeffs++ = value;
                        }
                    }
                }
            } else if (n_channels == 3) {
                float *coeffs_y = coeffs;
                float *coeffs_r = coeffs_y + ms*md*n_bases;
                float *coeffs_b = coeffs_r + ms*md*n_bases;

                for (size_t basis = 0; basis < n_bases; ++basis) {
                    for (int s = -msh; s <= msh; ++s) {
                        for (int d = 0; d < (int) md; ++d) {  // Can only save part of phi_d dimension and then use symmetries

                            float r = (float) ((*layers[basis*n_channels + 0]).scatter_coeff(s, d, op, ip) * weight);
                            float g = (float) ((*layers[basis*n_channels + 1]).scatter_coeff(s, d, op, ip) * weight);
                            float b = (float) ((*layers[basis*n_channels + 2]).scatter_coeff(s, d, op, ip) * weight);

                            float y = r * 0.212671f + g * 0.715160f + b * 0.072169f;
                            if (!std::isfinite(y))
                                Log(Warn, "Encountered invalid data: %f", y);

                            *coeffs_y++ = y; *coeffs_r++ = r; *coeffs_b++ = b;
                        }
                    }
                }
            }
        }
    }

    Log(Debug, "Computing cumulative distributions for importance sampling ..");

    /* Create an importance sampling "CDF Fourier series" coefficient table */

// #define CDF_DEBUG

#ifdef CDF_DEBUG
    MatrixX debug_pdf(n_nodes, n_nodes);
    MatrixX debug_cdf(n_nodes, n_nodes);
#endif

    int n_marginal_orders = ms_max_storage / 2 + 1; // Store real Cosine series for each pair (i,o)

    MatrixX spline_values(n_nodes, n_marginal_orders), spline_cdf(n_nodes, n_marginal_orders);
    for (size_t i = 0; i < n_nodes; ++i) {
        spline_values.setZero();
        for (size_t basis = 0; basis < n_bases; ++basis) {
            for (size_t o = 0; o < n_nodes; ++o) {

                const float *coeffs;
                OffsetType ms, md;
                std::tie(coeffs, ms, md) = storage->coeff_data_and_sizes(o, i, basis);

                OffsetType msh = ms / 2;
                int n_cdf_orders = ms / 2 + 1;  // Number of marginal orders for this specific pair (i,o)

                for (int j = 0; j < n_cdf_orders; ++j) {
                    int os = msh - j;
                    int od = j;

                    if (ms * md > 0) {
                        if (j == 0) {
                            spline_values(o, j) = (double) coeffs[os * md + od];
                        } else {
                            spline_values(o, j) = 2 * (double) coeffs[os * md + od];
                        }
                    } else {
                        spline_values(o, j) = 0;
                    }
                }
            }

            for (int j = 0; j < n_marginal_orders; ++j) {
                VectorX in(n_nodes), out(n_nodes);
                in = spline_values.col(j);
                spline::integrate_1d(nodes.data(), in.data(), (uint32_t) n_nodes, out.data());
                spline_cdf.col(j) = out;
            }

#ifdef CDF_DEBUG
            debug_pdf.col(i) = spline_values.col(0);
            debug_cdf.col(i) = spline_cdf.col(0);
#endif
            for (size_t o = 0; o < n_nodes; ++o) {
                float *cdf_ptr = storage->cdf(o) + i*n_bases*n_marginal_orders +
                                                       basis*n_marginal_orders;
                for (int j = 0; j < n_marginal_orders; ++j) {
                    cdf_ptr[j] = (float) spline_cdf(o, j);
                }
            }
        }
    }

#ifdef CDF_DEBUG
    Log(Debug, "BSDFStorage::from_layer_general(): CDF Debug mode enabled, writing out pdf and cdf tables into .csv files..");
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream pdf_file("debug_pdf.csv");
    pdf_file << debug_pdf.format(CSVFormat);

    std::ofstream cdf_file("debug_cdf.csv");
    cdf_file << debug_cdf.format(CSVFormat);
#endif

    Log(Debug, "BSDFStorage::from_layer_general(): Done.");

    return storage;
}

void BSDFStorage::fourier_slice(int i, int o, int channel, MatrixX &coeffs) const {
    coeffs.resize(1, 1);
    coeffs(0, 0) = 0;

    size_t ms, md;
    std::tie(ms, md) = fourier_dimensions(o, i);
    int msh = (int) ms / 2;

    if (ms*md == 0) {
        coeffs.resize(1, 1);
        coeffs(0, 0) = 0;
        return;
    }

    coeffs.resize(ms, md);
    coeffs.setZero();

    const float *source = coeff_data(o, i) + channel*ms*md;
    for (int s = -msh; s <= msh; ++s) {
        int os = s + msh;
        for (size_t d = 0; d < md; ++d) {
            coeffs(os, d) += (double) (*source++);
        }
    }
}

void BSDFStorage::fourier_slice_interpolated(float mu_i, float mu_o, int channel, MatrixX &coeffs) const {
    coeffs.resize(1, 1);
    coeffs(0, 0) = 0;

    /* Determine offsets and weights for \mu_i and \mu_o */
    float knot_weights_o[4], knot_weights_i[4];
    auto spline_weights_o = spline::eval_spline_weights(m_nodes, m_header->n_nodes, (float)mu_o, knot_weights_o);
    auto spline_weights_i = spline::eval_spline_weights(m_nodes, m_header->n_nodes, (float)mu_i, knot_weights_i);
    if (!spline_weights_o.first || !spline_weights_i.first)
        return;

    ssize_t knot_offset_o = spline_weights_o.second; ssize_t knot_offset_i = spline_weights_i.second;

    /* Lookup size of 2D Fourier expansion */
    size_t modes_s[16], modes_d[16];
    size_t ms_max, md_max;
    interpolated_fourier_dimensions(knot_offset_o, knot_offset_i, knot_weights_o, knot_weights_i, modes_s, modes_d, &ms_max, &md_max);

    coeffs.resize(ms_max, md_max);
    coeffs.setZero();
    int ms_max_half = (int) ms_max / 2;

    for (int i = 0; i < 4; ++i) {
        for (int o = 0; o < 4; ++o) {
            float weight = knot_weights_o[o] * knot_weights_i[i];
            if (weight == 0)
                continue;

            size_t ms = modes_s[4*i + o], md = modes_d[4*i + o];
            if (ms*md == 0) continue;

            int msh = (int) ms / 2;
            const float *source = coeff_data(knot_offset_o + o, knot_offset_i + i) + channel*ms*md;
            for (int s = -msh; s <= msh; ++s) {
                for (size_t d = 0; d < md; ++d) {
                    coeffs(s + ms_max_half, d) += (double) (*source++ * weight);
                }
            }
        }
    }
}

MTS_IMPLEMENT_CLASS(BSDFStorage, Object)
NAMESPACE_END(mitsuba)
