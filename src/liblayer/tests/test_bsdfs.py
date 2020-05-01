import mitsuba
import mitsuba.layer
import pytest
import enoki as ek
import numpy as np
import os

def test01_diffuse(variant_scalar_rgb):
    from mitsuba.core.xml import load_string
    from mitsuba.render import BSDF, BSDFContext, SurfaceInteraction3f
    from mitsuba.core import Frame3f

    thetas = np.linspace(0, np.pi / 2, 20)
    phi = np.pi

    values_ref = []

    # Create diffuse reference BSDF
    bsdf = load_string("""<bsdf version="2.0.0" type="diffuse">
                              <spectrum name="reflectance" value="0.5"/>
                          </bsdf>""")

    theta_i = np.radians(30.0)
    si    = SurfaceInteraction3f()
    si.p  = [0, 0, 0]
    si.n  = [0, 0, 1]
    si.wi = [np.sin(theta_i), 0, np.cos(theta_i)]
    si.sh_frame = Frame3f(si.n)
    ctx = BSDFContext()

    for theta in thetas:
        wo = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
        values_ref.append(bsdf.eval(ctx, si, wo=wo)[0])

    # Create same BSDF as layer representation
    n  = 100
    ms = 1
    md = 1
    mu, w = mitsuba.core.quad.gauss_lobatto(n)
    layer = mitsuba.layer.Layer(mu, w, ms, md)
    layer.set_diffuse(0.5)

    for i, theta in enumerate(thetas):
        l_eval = layer.eval(-np.cos(theta), np.cos(theta_i))*np.abs(np.cos(theta))
        # Values should be close (except if they are insignificantly small).
        # We have less precision at grazing angles because of Fourier representation.
        assert np.allclose(values_ref[i], l_eval, rtol=0.01)

    # Convert into BSDF storage representation
    base_path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = base_path + "diffuse.bsdf"
    storage = mitsuba.layer.BSDFStorage.from_layer(path, layer, 1e-5)

    for i, theta in enumerate(thetas):
        s_eval = storage.eval(np.cos(theta_i), -np.cos(theta))[0]
        # Values should be close (except if they are insignificantly small).
        # We have less precision at grazing angles because of Fourier representation.
        assert np.allclose(values_ref[i], s_eval, rtol=0.01)
    storage.close()

    # And load via the "fourier" BSDF plugin
    fourier = load_string("""<bsdf version="2.0.0" type="fourier">
                                 <string name="filename" value="{}"/>
                             </bsdf>""".format(path))

    for i, theta in enumerate(thetas):
        wo = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
        f_eval = fourier.eval(ctx, si, wo=wo)[0]
        assert np.allclose(values_ref[i], f_eval, rtol=0.02)
    del fourier

def test02_roughconductor(variant_scalar_rgb):
    from mitsuba.core.xml import load_string
    from mitsuba.render import BSDF, BSDFContext, SurfaceInteraction3f
    from mitsuba.core import Frame3f

    for alpha in [(0.3, 0.3), (0.3+1e-5, 0.3-1e-5), (0.2, 0.4)]:
        alpha_u = alpha[0]
        alpha_v = alpha[1]

        thetas = np.linspace(0, np.pi / 2, 20)
        phi = np.pi

        values_ref = []

        # Create conductor reference BSDF
        bsdf = load_string("""<bsdf version="2.0.0" type="roughconductor">
                                  <float name="alpha_u" value="{}"/>
                                  <float name="alpha_v" value="{}"/>
                                  <string name="distribution" value="beckmann"/>
                                  <spectrum name="eta" value="0.0"/>
                                  <spectrum name="k" value="1.0"/>
                              </bsdf>""".format(alpha_u, alpha_v))

        theta_i = np.radians(30.0)
        si    = SurfaceInteraction3f()
        si.p  = [0, 0, 0]
        si.n  = [0, 0, 1]
        si.wi = [np.sin(theta_i), 0, np.cos(theta_i)]
        si.sh_frame = Frame3f(si.n)
        ctx = BSDFContext()

        for theta in thetas:
            wo = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
            values_ref.append(bsdf.eval(ctx, si, wo=wo)[0])

        # Create same BSDF as layer representation
        n, ms, md = mitsuba.layer.microfacet_parameter_heuristic(alpha_u, alpha_v, 0+1j)
        mu, w = mitsuba.core.quad.gauss_lobatto(n)
        layer = mitsuba.layer.Layer(mu, w, ms, md)
        layer.set_microfacet(0+1j, alpha_u, alpha_v)

        for i, theta in enumerate(thetas):
            l_eval = layer.eval(-np.cos(theta), np.cos(theta_i))*np.abs(np.cos(theta))
            # Values should be close (except if they are insignificantly small).
            # We have less precision at grazing angles because of Fourier representation.
            print(values_ref[i], l_eval)
            assert values_ref[i] < 1e-5 or np.allclose(values_ref[i], l_eval, rtol=0.05/(np.abs(np.cos(theta))))

        # Convert into BSDF storage representation
        base_path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path + "roughconductor.bsdf"
        storage = mitsuba.layer.BSDFStorage.from_layer(path, layer, 1e-8)

        for i, theta in enumerate(thetas):
            s_eval = storage.eval(np.cos(theta_i), -np.cos(theta))[0]
            # Values should be close (except if they are insignificantly small).
            # We have less precision at grazing angles because of Fourier representation.
            assert values_ref[i] < 1e-5 or np.allclose(values_ref[i], s_eval, rtol=0.05/(np.abs(np.cos(theta))))
        storage.close()

        # And load via the "fourier" BSDF plugin
        fourier = load_string("""<bsdf version="2.0.0" type="fourier">
                                     <string name="filename" value="{}"/>
                                 </bsdf>""".format(path))

        for i, theta in enumerate(thetas):
            wo = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
            f_eval = fourier.eval(ctx, si, wo=wo)[0]
            assert values_ref[i] < 1e-5 or np.allclose(values_ref[i], f_eval, rtol=0.05/(np.abs(np.cos(theta))))
        del fourier

def test03_roughdielectric(variant_scalar_rgb):
    from mitsuba.core.xml import load_string
    from mitsuba.render import BSDF, BSDFContext, SurfaceInteraction3f, TransportMode
    from mitsuba.core import Frame3f

    for alpha in [(0.3, 0.3), (0.3+1e-5, 0.3-1e-5), (0.2, 0.4)]:
        alpha_u = alpha[0]
        alpha_v = alpha[1]

        thetas = np.linspace(0, np.pi, 20)
        phi = np.pi

        values_ref = []

        # Create dielectric reference BSDF
        bsdf = load_string("""<bsdf version="2.0.0" type="roughdielectric">
                                  <float name="alpha_u" value="{}"/>
                                  <float name="alpha_v" value="{}"/>
                                  <string name="distribution" value="beckmann"/>
                                  <float name="int_ior" value="1.5"/>
                                  <float name="ext_ior" value="1.0"/>
                              </bsdf>""".format(alpha_u, alpha_v))

        theta_i = np.radians(30.0)
        si    = SurfaceInteraction3f()
        si.p  = [0, 0, 0]
        si.n  = [0, 0, 1]
        si.wi = [np.sin(theta_i), 0, np.cos(theta_i)]
        si.sh_frame = Frame3f(si.n)
        ctx = BSDFContext()
        ctx.mode = TransportMode.Importance

        for theta in thetas:
            wo = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
            values_ref.append(bsdf.eval(ctx, si, wo=wo)[0])

        # Create same BSDF as layer representation
        n, ms, md = mitsuba.layer.microfacet_parameter_heuristic(alpha_u, alpha_v, 1.5)
        mu, w = mitsuba.core.quad.gauss_lobatto(n)
        layer = mitsuba.layer.Layer(mu, w, ms, md)
        layer.set_microfacet(1.5, alpha_u, alpha_v)

        for i, theta in enumerate(thetas):
            l_eval = layer.eval(-np.cos(theta), np.cos(theta_i))*np.abs(np.cos(theta))
            # Values should be close (except if they are insignificantly small).
            # We have less precision at grazing angles because of Fourier representation.
            assert values_ref[i] < 1e-5 or np.allclose(values_ref[i], l_eval, rtol=0.05/(np.abs(np.cos(theta))))

        # Convert into BSDF storage representation
        base_path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path + "roughdielectric.bsdf"
        storage = mitsuba.layer.BSDFStorage.from_layer(path, layer, 1e-5)

        for i, theta in enumerate(thetas):
            s_eval = storage.eval(np.cos(theta_i), -np.cos(theta))[0]
            # Values should be close (except if they are insignificantly small).
            # We have less precision at grazing angles because of Fourier representation.
            assert values_ref[i] < 1e-5 or np.allclose(values_ref[i], s_eval, rtol=0.05/(np.abs(np.cos(theta))))
        storage.close()

        # And load via the "fourier" BSDF plugin
        fourier = load_string("""<bsdf version="2.0.0" type="fourier">
                                     <string name="filename" value="{}"/>
                                 </bsdf>""".format(path))

        for i, theta in enumerate(thetas):
            wo = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
            f_eval = fourier.eval(ctx, si, wo=wo)[0]
            assert values_ref[i] < 1e-5 or np.allclose(values_ref[i], f_eval, rtol=0.05/(np.abs(np.cos(theta))))
        del fourier