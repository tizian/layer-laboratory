#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class TangentRotation final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture)

    TangentRotation(const Properties &props) : Base(props) {
        for (auto &kv : props.objects()) {
            auto *bsdf = dynamic_cast<Base *>(kv.second.get());
            if (bsdf) {
                if (m_nested_bsdf)
                    Throw("Cannot specify more than one child BSDF");
                m_nested_bsdf = bsdf;
            }
        }
        if (!m_nested_bsdf)
           Throw("Child BSDF not specified");

        m_angle_encoding = AngleEncoding::None;
        if (props.has_property("angles_direct")) {
            m_angle_encoding = AngleEncoding::Direct;
            m_texture = props.texture<Texture>("angles_direct", 0.f);
        } else if (props.has_property("angles_vector")) {
            m_angle_encoding = AngleEncoding::Vector;
            m_texture = props.texture<Texture>("angles_vector");
        } else if (props.has_property("angles_mesh")) {
            m_angle_encoding = AngleEncoding::Mesh;
            m_texture = props.texture<Texture>("angles_mesh");
        }

        m_flip_orientation = props.bool_("flip_orientation", false);

        parameters_changed({});
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        m_components.clear();
        for (size_t i = 0; i < m_nested_bsdf->component_count(); ++i)
            m_components.push_back(m_nested_bsdf->flags(i));

        m_flags = m_nested_bsdf->flags();
    }

    SurfaceInteraction3f rotate_frame(const SurfaceInteraction3f &si_, Mask active) const {
        SurfaceInteraction3f si(si_);

         if (m_angle_encoding == AngleEncoding::Mesh) {
            Vector3f v_object = Vector3f(m_texture->eval_3(si_, active));
            // std::cout << "v_object: " << v_object << std::endl;
            Vector3f v_local  = si_.to_local(v_object);
            // std::cout << "v_local: " << v_local << std::endl;
            v_local.z() = 0.f;
            v_local = normalize(v_local);
            // std::cout << "v_local: " << v_local << std::endl;

            si.sh_frame.s = si_.sh_frame.to_world(v_local);
            si.sh_frame.t = cross(si.sh_frame.n, si.sh_frame.s);
        } else {
            Float phi = 0.f;

            if (m_angle_encoding == AngleEncoding::Direct) {
                phi = 2.f*math::Pi<Float>*m_texture->eval_1(si, active);
            } else if (m_angle_encoding == AngleEncoding::Vector) {
                Color3f v = m_texture->eval_3(si, active);
                Vector2f vec = normalize(Vector2f(
                    2.f*(v.x() - 0.5f),
                    2.f*(v.y() - 0.5f)
                ));
                masked(phi, active) = atan2(vec.y(), vec.x());
            }

            Transform4f rot = Transform4f::rotate(si.sh_frame.n, rad_to_deg(phi));
            si.sh_frame.t = rot * si.sh_frame.t;
            si.sh_frame.s = rot * si.sh_frame.s;
        }

        if (m_flip_orientation)
            std::swap(si.sh_frame.t, si.sh_frame.s);

        return si;
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si_,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        SurfaceInteraction si = rotate_frame(si_, active);
        si.wi = si.to_local(si_.to_world(si_.wi));
        auto [bs, weight] = m_nested_bsdf->sample(ctx, si, sample1, sample2, active);
        bs.wo = si_.to_local(si.to_world(bs.wo));
        return { bs, weight };
    }

    Spectrum eval(const BSDFContext &ctx,
                  const SurfaceInteraction3f &si_,
                  const Vector3f &wo_,
                  Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        SurfaceInteraction3f si = rotate_frame(si_, active);
        si.wi = si.to_local(si_.to_world(si_.wi));
        Vector3f wo = si.to_local(si_.to_world(wo_));
        return m_nested_bsdf->eval(ctx, si, wo, active);
    }

    Float pdf(const BSDFContext &ctx,
              const SurfaceInteraction3f &si_,
              const Vector3f &wo_,
              Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        SurfaceInteraction3f si = rotate_frame(si_, active);
        si.wi = si.to_local(si_.to_world(si_.wi));
        Vector3f wo = si.to_local(si_.to_world(wo_));
        return m_nested_bsdf->pdf(ctx, si, wo, active);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("texture", m_texture.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "TangentRotation[" << std::endl
            << "  nested_bsdf = " << string::indent(m_nested_bsdf->to_string()) << "," << std::endl
            << "  texture = " << (m_texture ? string::indent(m_texture->to_string()) : "null") << "," << std::endl
            << "  encoding = " << (m_angle_encoding == 0 ? "direct" : "vector") << "," << std::endl
            << "  flip_orientation = " << std::boolalpha << m_flip_orientation << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Base> m_nested_bsdf;
    ref<Texture> m_texture;

    enum AngleEncoding {
        None,            // Nothing to do
        Direct,          // Monochromatic texture, encoding directly the rotation angle. [0,1] will be mapped to [0,2pi]
        Vector,          // RGB texture, encoding the tangent direction with a 2D vector (with R & G channels)
        Mesh,            // Mesh attribute texture, encoding 3D orientation vector (in object space) at each vertex
    } m_angle_encoding;
    bool m_flip_orientation;
};

MTS_IMPLEMENT_CLASS_VARIANT(TangentRotation, BSDF)
MTS_EXPORT_PLUGIN(TangentRotation, "Tangent rotation material")
NAMESPACE_END(mitsuba)
