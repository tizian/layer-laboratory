import mitsuba
import mitsuba.layer
import pytest
import enoki as ek
import numpy as np

def test01_isotropic(variant_scalar_rgb):
    # Reference data from "Multiple Light Scattering" by C. van de Hulst, 1980
    # Volume 1, Table 12 p.260-261 (Isotropic scattering, finite slabs)
    albedo = 0.8
    thickness = 2.0

    mu, w = mitsuba.core.quad.gauss_lobatto(50)
    layer = mitsuba.layer.Layer(mu, w)
    layer.set_isotropic(albedo)
    layer.expand(thickness)
    print(layer)

    # Intensities leaving out at top
    mu_i_top = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    mu_o_top = 0.5
    ref_top = [0.28616, 0.30277, 0.34188, 0.39102, 0.45353, 0.53214]
    for i in range(len(mu_i_top)):
        table = ref_top[i]
        computed = layer.eval(mu_o_top, -mu_i_top[i])*np.pi
        assert np.allclose(table, computed, rtol=0.001)

    # Intensities leaving out at bottom
    mu_i_bot = [1.0, 0.9, 0.7, 0.3, 0.1]
    mu_o_bot = -0.5
    ref_bot = [0.14864, 0.14731, 0.13963, 0.09425, 0.06601]
    for i in range(len(mu_i_bot)):
        table = ref_bot[i]
        computed = layer.eval(mu_o_bot, -mu_i_bot[i])*np.pi
        assert np.allclose(table, computed, rtol=0.003)


def test02_hg(variant_scalar_rgb):
    # Reproduce same results from test01_isotropic using the HG implementation
    albedo = 0.8
    thickness = 2.0
    g = 0.0001

    n, ms, md = mitsuba.layer.henyey_greenstein_parameter_heuristic(g)
    n = 100
    mu, w = mitsuba.core.quad.gauss_lobatto(n)
    layer = mitsuba.layer.Layer(mu, w, ms, md)
    layer.set_henyey_greenstein(albedo, g)
    layer.expand(thickness)

    # Intensities leaving out at top
    mu_i_top = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    mu_o_top = 0.5
    ref_top = [0.28616, 0.30277, 0.34188, 0.39102, 0.45353, 0.53214]
    for i in range(len(mu_i_top)):
        table = ref_top[i]
        computed = layer.eval(mu_o_top, -mu_i_top[i])*np.pi
        assert np.allclose(table, computed, rtol=0.001)

    # Intensities leaving out at bottom
    mu_i_bot = [1.0, 0.9, 0.7, 0.3, 0.1]
    mu_o_bot = -0.5
    ref_bot = [0.14864, 0.14731, 0.13963, 0.09425, 0.06601]
    for i in range(len(mu_i_bot)):
        table = ref_bot[i]
        computed = layer.eval(mu_o_bot, -mu_i_bot[i])*np.pi
        assert np.allclose(table, computed, rtol=0.003)
