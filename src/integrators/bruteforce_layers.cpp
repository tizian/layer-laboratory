#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/math.h>
#include <fstream>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class DepthIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(SamplingIntegrator)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, BSDF)

    DepthIntegrator(const Properties &props) : Base(props) {
        m_num_samples = props.int_("num_samples", 10000000);

        Float theta_i = deg_to_rad(props.float_("theta_i", 30.f));
        Float phi_i   = deg_to_rad(props.float_("phi_i", 0.f));
        m_wi = math::sphdir(theta_i, phi_i);

        FileResolver* fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_filename = file_path.filename().string();

        for (auto &kv : props.objects()) {
            auto *bsdf = dynamic_cast<BSDF *>(kv.second.get());
            if (bsdf) {
                m_layers.push_back(bsdf);
            }
        }
    }

    std::pair<Spectrum, Mask> sample(const Scene * /* scene */,
                                     Sampler *sampler,
                                     const RayDifferential3f & /* ray */,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        SurfaceInteraction3f si = enoki::zero<SurfaceInteraction3f>();
        si.t = 1.f;

        auto sample_layers = [&](Vector3f wi_) {
            BSDFContext ctx;

            int layer = 0;
            Spectrum spec = 1.f;
            Vector3f wi = wi_;
            Vector3f wo;

            while (layer >= 0 && layer < int(m_layers.size())) {
                si.wi = wi;
                auto [bs, weight] = m_layers[layer]->sample(ctx, si,
                                                            sampler->next_1d(),
                                                            sampler->next_2d(),
                                                            true);
                wo = bs.wo;
                spec *= weight;

                layer += wo.z() > 0 ? -1.f : 1.f;
                wi = -wo;
            }

            return std::make_pair(wo, luminance(spec));
        };

        size_t histogram_res = 128;
        float *Rt = new float[histogram_res*histogram_res];
        memset(Rt, 0, histogram_res*histogram_res*sizeof(float));

        std::ofstream file(m_filename);

        int N = 0;
        Float mean = 0;
        for (int s = 0; s < m_num_samples; ++s) {
            Vector3f wo;
            Float weight;
            std::tie(wo, weight) = sample_layers(m_wi);
            if (wo.z() < 0) continue;

            Float theta_o = acos(wo.z()),
                  phi_o   = atan2(wo.y(), wo.x());
            if (phi_o < 0.f)
                phi_o += 2.f*math::Pi<Float>;

            int i = int((theta_o / (0.5f * math::Pi<Float>)) * histogram_res);
            int j = int((phi_o / (2.f * math::Pi<Float>)) * histogram_res);
            i = max(0, min(i, int(histogram_res - 1)));
            j = max(0, min(j, int(histogram_res - 1)));

            Float jacobian = sin(deg_to_rad(((float)i + 0.5f) * 90.f / histogram_res));
            Float tmp = weight / jacobian;

            Rt[i*histogram_res + j] += tmp;
            mean += weight;

            N++;
        }
        mean /= N;

        for (size_t i = 0; i < histogram_res; ++i) {
            for (size_t j = 0; j < histogram_res; ++j) {
                file << Rt[i*histogram_res + j] << "\n";
            }
        }
        file.close();

        return { 0.f, true };
    }

    MTS_DECLARE_CLASS()
protected:
    std::string m_filename;
    std::vector<ref<BSDF>> m_layers;
    int m_num_samples;
    Vector3f m_wi;
};

MTS_IMPLEMENT_CLASS_VARIANT(DepthIntegrator, SamplingIntegrator)
MTS_EXPORT_PLUGIN(DepthIntegrator, "Depth integrator");
NAMESPACE_END(mitsuba)
