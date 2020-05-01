try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
import mitsuba.layer

import numpy as np

# Example 1: Medium anisotropy

alpha_u = 0.1
alpha_v = 0.2
eta = 0+1j

n, ms, md = mitsuba.layer.microfacet_parameter_heuristic(alpha_u, alpha_v, eta)
mu, w = mitsuba.core.quad.gauss_lobatto(n)
print(n, ms, md)
print("")

for s in [1, 3, 9]:
    if s > ms:
        break

    print('ms: ', s)

    layer = mitsuba.layer.Layer(mu, w, s, md)
    layer.set_microfacet(eta, alpha_u, alpha_v)
    layer.clear_backside()

    filename = "%.02f_%.02f_%02d.bsdf" % (alpha_u, alpha_v, s)
    storage = mitsuba.layer.BSDFStorage.from_layer(filename, layer, 1e-8)


# Example 2: Extreme anisotropy

alpha_u = 0.05
alpha_v = 0.4
eta = 0+1j

n, ms, md = mitsuba.layer.microfacet_parameter_heuristic(alpha_u, alpha_v, eta)
mu, w = mitsuba.core.quad.gauss_lobatto(n)
print(n, ms, md)
print("")

for s in [3, 9, 27]:
    if s > ms:
        break

    print('ms: ', s)

    layer = mitsuba.layer.Layer(mu, w, s, md)
    layer.set_microfacet(eta, alpha_u, alpha_v)
    layer.clear_backside()

    filename = "%.02f_%.02f_%02d.bsdf" % (alpha_u, alpha_v, s)
    storage = mitsuba.layer.BSDFStorage.from_layer(filename, layer, 1e-8)