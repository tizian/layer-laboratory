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


# Anisotropic gold metal, no layering
print("Anisotropic gold metal, no layering")

eta = get_rgb(gold)
alpha_u = 0.05
alpha_v = 0.3

n, ms, md = mitsuba.layer.microfacet_parameter_heuristic(alpha_u, alpha_v, eta[0])
mu, w = mitsuba.core.quad.gauss_lobatto(n)

channels = []
for i in range(3):
    layer = mitsuba.layer.Layer(mu, w, ms, md)
    layer.set_microfacet(eta[i], alpha_u, alpha_v)
    layer.clear_backside()

    channels.append(layer)

filename = "aniso_gold.bsdf"
storage = mitsuba.layer.BSDFStorage.from_layer_rgb(filename, *channels, 1e-4)


# Aniso. gold metal + HG dust layer
print("Aniso. gold metal + HG dust layer")

eta = get_rgb(gold)
alpha_u = 0.05
alpha_v = 0.3

g = 0.5
albedo = 0.95
tau = 0.2

n1, ms1, md1 = mitsuba.layer.henyey_greenstein_parameter_heuristic(g)
n2, ms2, md2 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u, alpha_v, eta[0])
n = max(n1, n2)
ms = max(ms1, ms2)
md = md2
mu, w = mitsuba.core.quad.gauss_lobatto(n)

dust = mitsuba.layer.Layer(mu, w, ms, md)
dust.set_henyey_greenstein(albedo, g)
dust.expand(tau)

channels = []
for i in range(3):
    layer = mitsuba.layer.Layer(mu, w, ms, md)
    layer.set_microfacet(eta[i], alpha_u, alpha_v)
    layer.clear_backside()

    # Apply dust
    layer.add_to_top(dust, epsilon=1e-5)

    channels.append(layer)

filename = "aniso_gold_dust.bsdf"
storage = mitsuba.layer.BSDFStorage.from_layer_rgb(filename, *channels, 1e-4)


# Iso. gold metal + iso. dielectric coating
print("Iso. gold metal + iso. dielectric coating")

eta_bot = get_rgb(gold)
alpha_u_bot = 0.1
alpha_v_bot = 0.1

eta_top = 1.5
alpha_u_top = 0.1
alpha_v_top = 0.1

n1, ms1, md1 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_top, alpha_v_top, eta_top)
n2, ms2, md2 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_bot, alpha_v_bot, eta_bot[0])
n = max(n1, n2)
ms = max(ms1, ms2)
md = md2
mu, w = mitsuba.core.quad.gauss_lobatto(n)

layer_top = mitsuba.layer.Layer(mu, w, ms, md)
layer_top.set_microfacet(eta_top, alpha_u_top, alpha_v_top)

channels = []
for i in range(3):
    layer_bot = mitsuba.layer.Layer(mu, w, ms, md)
    layer_bot.set_microfacet(eta_bot[i], alpha_u_bot, alpha_v_bot)
    layer_bot.clear_backside()

    layer_bot.add_to_top(layer_top, epsilon=1e-5)

    channels.append(layer_bot)

filename = "iso_gold_iso_dielectric.bsdf"
storage = mitsuba.layer.BSDFStorage.from_layer_rgb(filename, *channels, 1e-4)


# Aniso. gold metal + aniso. dielectric coating
print("Aniso. gold metal + aniso. dielectric coating")

eta_bot = get_rgb(gold)
alpha_u_bot = 0.05
alpha_v_bot = 0.3

eta_top = 1.5
alpha_u_top = 0.3
alpha_v_top = 0.05

n1, ms1, md1 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_top, alpha_v_top, eta_top)
n2, ms2, md2 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_bot, alpha_v_bot, eta_bot[0])
n = max(n1, n2)
ms = max(ms1, ms2)
md = md2
mu, w = mitsuba.core.quad.gauss_lobatto(n)

layer_top = mitsuba.layer.Layer(mu, w, ms, md)
layer_top.set_microfacet(eta_top, alpha_u_top, alpha_v_top)

channels = []
for i in range(3):
    layer_bot = mitsuba.layer.Layer(mu, w, ms, md)
    layer_bot.set_microfacet(eta_bot[i], alpha_u_bot, alpha_v_bot)
    layer_bot.clear_backside()

    layer_bot.add_to_top(layer_top, epsilon=1e-5)

    channels.append(layer_bot)

filename = "aniso_gold_aniso_dielectric.bsdf"
storage = mitsuba.layer.BSDFStorage.from_layer_rgb(filename, *channels, 1e-4)


# Aniso. gold + blue scattering medium + iso. dielectric coating
print("Aniso. gold + blue scattering medium + iso. dielectric coating")

eta_bot = get_rgb(gold)
alpha_u_bot = 0.05
alpha_v_bot = 0.3

g = 0.2
albedo = [0.6, 0.8, 0.95]
tau = 0.3

alpha_u_top = 0.05
alpha_v_top = 0.05
eta_top = 1.5

n1, ms1, md1 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_top, alpha_v_top, eta_top)
n2, ms2, md2 = mitsuba.layer.henyey_greenstein_parameter_heuristic(-g)
n3, ms3, md3 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_bot, alpha_v_bot, eta_bot[0])
n = max(n1, n2, n3)
ms = max(ms1, ms2, ms3)
md = md3
mu, w = mitsuba.core.quad.gauss_lobatto(n)

dielectric = mitsuba.layer.Layer(mu, w, ms, md)
dielectric.set_microfacet(eta_top, alpha_u_top, alpha_v_top)

channels = []
for i in range(3):
    b = mitsuba.layer.Layer(mu, w, ms, md)
    b.set_microfacet(eta_bot[i], alpha_u_bot, alpha_v_bot)
    b.clear_backside()

    m = mitsuba.layer.Layer(mu, w, ms, md)
    m.set_henyey_greenstein(albedo[i], g)
    m.expand(tau)

    c = mitsuba.layer.Layer.add(m, b, epsilon=1e-5)
    c.add_to_top(dielectric, epsilon=1e-5)

    channels.append(c)

filename = "aniso_gold_blue_dielectric.bsdf"
storage = mitsuba.layer.BSDFStorage.from_layer_rgb(filename, *channels, 1e-4)
