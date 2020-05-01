try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
import mitsuba.layer

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import identity

# External requirement: brew install suite-sparse, pip install sparseqr
import sparseqr

class MyNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin, vmax):
        matplotlib.colors.Normalize.__init__(self)
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, value, clip=False):
        return np.ma.masked_array(np.where(value > 0, 0.5 + 0.5*value/self.vmax, 0.5 - 0.5*value/self.vmin))

def plot_layer(ax1, layer, zenith_i=30.0, azimuth_i=0.0, transmission=False, clamp=True):
    if zenith_i < 90 and not transmission:
        component = 'Rt'
    elif zenith_i < 90 and transmission:
        component = 'Ttb'
    elif zenith_i >= 90 and not transmission:
        component = 'Rb'
    else:
        component = 'Tbt'

    phi_i = np.radians(azimuth_i)
    theta_i = np.radians(zenith_i)
    mu_i = np.cos(theta_i)

    azimuths = np.linspace(0, 360, 200)
    if component == 'Rt' or component == 'Tbt':
        zeniths = np.linspace(0, 90, 200)
        theta_ticks_deg = [10, 30, 50, 70, 90]
        theta_labels = ['0˚', '', '', '', '90˚']
    elif component == 'Rb' or component == 'Ttb':
        zeniths = np.linspace(180, 90, 200)
        theta_ticks_deg = [180, 160, 140, 120, 100]
        theta_labels = ['90˚', '', '', '', '180˚']

    theta_o, phi_o = np.meshgrid(np.radians(zeniths), np.radians(azimuths))
    mu_o = -np.cos(theta_o)

    phi_s = phi_o + phi_i
    phi_d = phi_o - phi_i

    storage = mitsuba.layer.BSDFStorage.from_layer("tmp.bsdf", layer, 1e-8)
    data = storage.eval(mu_i, mu_o, 0.5*(phi_s - phi_d), 0.5*(phi_s + phi_d), clamp)

    ax1.grid(linestyle='-', linewidth=0.6, alpha=0.3, color='w')

    text_col = 'k'
    ax1.set_rgrids(np.radians(theta_ticks_deg), labels=theta_labels, angle=270, color=text_col, fontweight='ultralight', size='10', ha='center', alpha=0.8)

    phi_ticks_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
    phi_labels = [("%d˚" % i) for i in phi_ticks_deg]
    phi_labels[0] = ""; phi_labels[4] = ""
    phi_ticks_deg -= np.degrees(phi_i)
    phi_ticks_deg = np.where(phi_ticks_deg < 0, phi_ticks_deg + 360, phi_ticks_deg)
    phi_ticks_deg = np.where(phi_ticks_deg > 360, phi_ticks_deg - 360, phi_ticks_deg)
    ax1.set_thetagrids(phi_ticks_deg, labels=phi_labels, color='k', fontweight='ultralight', size='10', ha='center', alpha=0.8)

    vmin = np.min(data)
    vmax = np.max(data)
    norm = MyNormalize(vmin, vmax)

    theta_o_plot = theta_o if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_o
    view = ax1.contourf(phi_o-phi_i, theta_o_plot, data, 200, cmap='coolwarm', norm=norm, vmin=vmin, vmax=vmax)
    out_info = (phi_o-phi_i, theta_o_plot, data)
    for c in view.collections:
        c.set_edgecolor("face")
        c.set_rasterized(True)

    theta_i_mark = theta_i
    phi_i_plot = np.pi
    if component == 'Ttb' or component == 'Tbt':
        theta_i_mark = np.pi - theta_i

    theta_i_plot = theta_i if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_i_mark
    xy = (phi_i_plot, np.abs(theta_i_plot))
    xytext = (phi_i_plot-0.3, np.abs(theta_i_plot)+0.1)
    ax1.plot(xy[0], xy[1], 'x', color=text_col, ms='10', mew=2)
    ax1.annotate('$\omega_i$', xy=xy, textcoords='data', color=text_col, fontweight='black', size='14', xytext=xytext)

    dr = 0.16
    if component == 'Rb' or component == 'Ttb':
        dstart = 1.033*np.pi
        orientation_line_radii = [dstart, dstart+dr]
    else:
        dstart = 0.533*np.pi
        orientation_line_radii = [dstart, dstart+dr]
    x,y = np.array([[-phi_i, -phi_i], orientation_line_radii])
    line = mlines.Line2D(x, y, lw=14, color='k')
    line.set_clip_on(False)
    ax1.add_line(line)
    x,y = np.array([[-phi_i-np.pi, -phi_i-np.pi], orientation_line_radii])
    line = mlines.Line2D(x, y, lw=14, color='k')
    line.set_clip_on(False)
    ax1.add_line(line)

    [i.set_linewidth(2) for i in ax1.spines.values()]

    return view

def mk_diagonal(x):
    return scipy.sparse.csc_matrix(scipy.sparse.spdiags(x, 0, x.shape[0], x.shape[0]))

def modify_system(A, b, eps_w=0, eps_laplace=0):
    As = [A]
    bs = [b]

    zero_col = scipy.sparse.csc_matrix(b.shape, dtype=np.float64)

    if eps_w > 0:
        As.append(mk_diagonal(weights) * eps_w)
        bs.append(zero_col)

    if eps_laplace > 0:
        As.append(Lap @ mk_diagonal(weights) * eps_laplace)
        bs.append(zero_col)

    A_mod = scipy.sparse.vstack(As)
    b_mod = scipy.sparse.vstack(bs)
    return A_mod, b_mod

def sparsesolve(A, b, eps_w=0, eps_laplace=0):
    A_mod, b_mod = modify_system(A, b, eps_w, eps_laplace)
    return sparseqr.solve(A_mod, b_mod)


# Dielectric top layer
eta_1 = 1.5
alpha_1 = 0.2

# Conductor bottom layer
eta_2 = 0+1j
alpha_u_2, alpha_v_2 = 0.301, 0.3

n1, ms1, md1 = mitsuba.layer.microfacet_parameter_heuristic(alpha_1, alpha_1, eta_1)
n2, ms2, md2 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_2, alpha_v_2, eta_2)
n = max(n1, n2)
ms = max(ms1, ms2)
md = max(md1, md2)
mu, w = mitsuba.core.quad.gauss_lobatto(n)

base_layer = mitsuba.layer.Layer(mu, w, ms, md)
base_layer.set_microfacet(eta_2, alpha_u_2, alpha_v_2)
# base_layer.set_diffuse(0.5)

# Make double sided to make inversion more reasonable (no TIR!)
t1 = mitsuba.layer.Layer(mu, w, ms, md)
t1.set_microfacet(eta_1, alpha_1, alpha_1)
t2 = mitsuba.layer.Layer(mu, w, ms, md)
t2.set_microfacet(eta_1, alpha_1, alpha_1)
t2.reverse()
top = mitsuba.layer.Layer.add(t1, t2)

print('n: %d, ms: %d, md: %d' % (n, ms, md))
print('scattering matrix size: ', top.reflection_bottom.shape[0])

# Extract scattering matrices
Rb_1 = top.reflection_bottom
Rt_1 = top.reflection_top
Ttb_1 = top.transmission_top_bottom
Tbt_1 = top.transmission_bottom_top

# First part of subtracting equations
I = scipy.sparse.eye(Rt_1.shape[0])

# Regularization
eps_w = 4e-2
weights = mu[n//2:] * w[n//2:]
weights = np.tile(weights, md+ms-1)

Ttb_1_i = scipy.sparse.csc_matrix(sparsesolve(Ttb_1, I, eps_w=eps_w, eps_laplace=0))
Ttb_1_i = scipy.sparse.coo_matrix(Ttb_1_i)

inverse_transmission = mitsuba.layer.Layer(mu, w, ms, md)
inverse_transmission.set_transmission_top_bottom(Ttb_1_i)

# Visualize intermediate result, the "inverse transmission"
plt.figure(figsize=(5,5))

outer = gridspec.GridSpec(1, 1)
gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.25, width_ratios=[1.0, 0.06], subplot_spec = outer[0])

ax1 = plt.subplot(gs[0], projection='polar')
p = plot_layer(ax1, inverse_transmission, transmission=True, clamp=False, zenith_i=20)
ax2 = plt.subplot(gs[1])
cb = plt.colorbar(p, cax=ax2)
ax2.set_aspect(13.9)
cb.set_ticks([-34000, 0, 70000])

plt.savefig("inverse_transmission.pdf", bbox_inches='tight')
plt.show()
