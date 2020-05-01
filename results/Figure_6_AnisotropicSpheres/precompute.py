try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
import mitsuba.layer

import numpy as np
from cie import *
from materials import gold

eta_top = 1.5
eta_bot = get_rgb(gold)

parameters = {
    'hor_hor': [(0.30, 0.05), (0.30, 0.05)],
    'hor_iso': [(0.30, 0.05), (0.15, 0.15)],
    'hor_ver': [(0.30, 0.05), (0.05, 0.30)],
    'iso_hor': [(0.15, 0.15), (0.30, 0.05)],
    'iso_iso': [(0.15, 0.15), (0.15, 0.15)],
    'iso_ver': [(0.15, 0.15), (0.05, 0.30)],
    'ver_hor': [(0.05, 0.30), (0.30, 0.05)],
    'ver_iso': [(0.05, 0.30), (0.15, 0.15)],
    'ver_ver': [(0.05, 0.30), (0.05, 0.30)],
}

for key, alphas in parameters.items():
    alpha_u_top = alphas[0][0]
    alpha_v_top = alphas[0][1]
    alpha_u_bot = alphas[1][0]
    alpha_v_bot = alphas[1][1]

    print(alpha_u_top, alpha_v_top)
    print("+")
    print(alpha_u_bot, alpha_v_bot)

    n1, ms1, md1 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_top, alpha_v_top, eta_top)
    n2, ms2, md2 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_bot, alpha_v_bot, eta_bot[0])
    n = max(n1, n2)
    ms = max(ms1, ms2)
    md = md2
    mu, w = mitsuba.core.quad.gauss_lobatto(n)

    print("   dielectric top")
    dielectric = mitsuba.layer.Layer(mu, w, ms, md)
    dielectric.set_microfacet(eta_top, alpha_u_top, alpha_v_top)

    channels = []
    for i in range(3):
        print("   gold base [%d]" % i)
        layer = mitsuba.layer.Layer(mu, w, ms, md)
        layer.set_microfacet(eta_bot[i], alpha_u_bot, alpha_v_bot)
        layer.clear_backside()

        layer.add_to_top(dielectric, epsilon=1e-6)

        channels.append(layer)

        print("    save")
    filename = "%s.bsdf" % key
    storage = mitsuba.layer.BSDFStorage.from_layer_rgb(filename, *channels, 1, 1e-4)

    print("\n")