#include <mitsuba/render/bsdf.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/layer/storage.h>
#include <mitsuba/render/srgb.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Vector3,
          typename Value = value_t<Vector3>>
Value phi(const Vector3 &v) {
    Value p = atan2(v.y(), v.x());
    masked(p, p < 0) += 2.f*math::Pi<Value>;
    return p;
}

template <typename Float, typename Spectrum>
class Fourier final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES()

    Fourier(const Properties &props) : Base(props) {
        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_name = file_path.filename().string();

        m_storage = new BSDFStorage(file_path);

        m_flags = BSDFFlags::GlossyReflection |
                  BSDFFlags::GlossyTransmission |
                  BSDFFlags::FrontSide |
                  BSDFFlags::BackSide;
        if (m_storage->max_orders().first > 1)
            m_flags = m_flags | BSDFFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /* sample1 */,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        BSDFSample3f bs = zero<BSDFSample3f>();
        if (!ctx.is_enabled(BSDFFlags::Glossy))
            return { bs, 0 };

        Float mu_i  = Frame3f::cos_theta(si.wi),
              phi_i = phi(si.wi);

        const float *basis_coeffs = nullptr;    // TODO: Compute basis_coeffs from textures
        auto [weight_rgb, mu_o, phi_o, p] = m_storage->sample(mu_i, phi_i, sample2, basis_coeffs);

        Float phi_d = phi_o - phi_i;
        Float sin_theta_2_o = max(0, 1 - mu_o*mu_o);
        Float norm = sqrt(sin_theta_2_o / Frame3f::sin_theta_2(si.wi));
        masked(norm, isinf(norm)) = 0;

        auto [sin_phi_d, cos_phi_d] = sincos(phi_d);
        bs.wo = -normalize(Vector3f(norm * (cos_phi_d * si.wi.x() - sin_phi_d * si.wi.y()),
                                    norm * (sin_phi_d * si.wi.x() + cos_phi_d * si.wi.y()),
                                    mu_o));

        Float n_dot_wi = Frame3f::cos_theta(si.wi),
              n_dot_wo = Frame3f::cos_theta(bs.wo);

        Float factor = select(n_dot_wo < 0, m_storage->eta(), rcp(m_storage->eta()));
        bs.eta = select(n_dot_wi*n_dot_wo > 0, 1.f, factor);

        bs.sampled_type = select(n_dot_wi*n_dot_wo > 0, +BSDFFlags::GlossyReflection, +BSDFFlags::GlossyTransmission);
        bs.sampled_component = 0;
        bs.pdf = p;

        Spectrum weight;
        if constexpr (is_spectral_v<Spectrum>) {
            Float lum = luminance(weight_rgb);
            weight_rgb *= select(lum > 0.f, rcp(lum), 0.f);
            weight = lum*srgb_model_eval<Spectrum>(srgb_model_fetch(weight_rgb), si.wavelengths);
        } else {
            weight = weight_rgb;
        }

        return { bs, weight };
    }

    Spectrum eval(const BSDFContext &ctx,
                  const SurfaceInteraction3f &si,
                  const Vector3f &wo_,
                  Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::Glossy))
            return 0.f;

        Vector3f wi = si.wi;
        Vector3f wo = -wo_;

        Float mu_i  = Frame3f::cos_theta(wi),
              mu_o  = Frame3f::cos_theta(wo),
              phi_i = phi(wi),
              phi_o = phi(wo);

        const float *basis_coeffs = nullptr;
        auto value_rgb = m_storage->eval(mu_i, mu_o, phi_i, phi_o, basis_coeffs);

        Spectrum value;
        if constexpr (is_spectral_v<Spectrum>) {
            Float lum = luminance(value_rgb);
            value_rgb *= select(lum > 0.f, rcp(lum), 0.f);
            value = lum*srgb_model_eval<Spectrum>(srgb_model_fetch(value_rgb), si.wavelengths);
        } else {
            value = value_rgb;
        }

        Float n_dot_wi = Frame3f::cos_theta(si.wi),
              n_dot_wo = Frame3f::cos_theta(wo_);

        Float factor = select(n_dot_wo < 0, m_storage->eta(), rcp(m_storage->eta()));
        masked(value, n_dot_wi*n_dot_wo <= 0) *= factor*factor;

        return value;
    }

    Float pdf(const BSDFContext &ctx,
              const SurfaceInteraction3f &si,
              const Vector3f &wo_,
              Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::Glossy))
            return 0.f;

        Vector3f wi = si.wi;
        Vector3f wo = -wo_;

        Float mu_i  = Frame3f::cos_theta(wi),
              mu_o  = Frame3f::cos_theta(wo),
              phi_i = phi(wi),
              phi_o = phi(wo);

        const float *basis_coeffs = nullptr;
        return m_storage->pdf(mu_i, mu_o, phi_i, phi_o, basis_coeffs);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Fourier[" << std::endl
            << "  name = \"" << m_name << "\"," << std::endl
            << "  storage = " << string::indent(m_storage->to_string()) <<  "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    std::string m_name;
    ref<BSDFStorage> m_storage;
};

MTS_IMPLEMENT_CLASS_VARIANT(Fourier, BSDF)
MTS_EXPORT_PLUGIN(Fourier, "Tabulated Fourier/Layerlab BRDF")
NAMESPACE_END(mitsuba)
