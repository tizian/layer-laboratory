try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
import mitsuba.layer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox
import os

display = False

def plot_microfacet_polar(ax1, zenith_i=30.0, azimuth_i=0.0, alpha_u=0.1, alpha_v=0.3, eta=1.5, transmission=False, markers=None):
    theta_i = np.radians(zenith_i)
    phi_i = np.radians(azimuth_i)

    mu_i = np.cos(theta_i)
    azimuths = np.linspace(0, 360, 200)
    if transmission:
        zeniths = np.linspace(90, 180, 200)
    else:
        zeniths = np.linspace(0, 90, 200)
    theta_o, phi_o = np.meshgrid(np.radians(zeniths), np.radians(azimuths))

    mu_o = -np.cos(theta_o)
    phi_s = phi_o + phi_i
    phi_d = phi_o - phi_i

    data = mitsuba.layer.microfacet(-mu_o, -mu_i, phi_s, phi_d, alpha_u, alpha_v, eta)
    data *= np.abs(mu_o)

    ax1.grid(linestyle='-', linewidth=0.6, alpha=0.3, color='w')

    r_start = np.radians(zeniths[0]); r_end = np.radians(zeniths[-1])
    ax1.set_rlim(r_start, r_end)
    if transmission:
        theta_ticks_deg = [100, 120, 140, 160, 180]
    else:
        theta_ticks_deg = [10, 30, 50, 70, 90]
    theta_labels = [("%d˚" % i) for i in theta_ticks_deg]
    ax1.set_rgrids(np.radians(theta_ticks_deg), labels=theta_labels, angle=270, color='w', fontweight='ultralight', size='10', ha='center', va='bottom', alpha=0.8)

    phi_ticks_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
    phi_labels = [("%d˚" % i) for i in phi_ticks_deg]
    phi_labels[0] = ""; phi_labels[4] = ""
    phi_ticks_deg -= np.degrees(phi_i)
    phi_ticks_deg = np.where(phi_ticks_deg < 0, phi_ticks_deg + 360, phi_ticks_deg)
    phi_ticks_deg = np.where(phi_ticks_deg > 360, phi_ticks_deg - 360, phi_ticks_deg)
    ax1.set_thetagrids(phi_ticks_deg, labels=phi_labels, color='k', fontweight='ultralight', size='14', ha='center', alpha=0.8, )

    view = ax1.contourf(phi_o-phi_i, theta_o, data, 400, cmap='jet', zorder=-9)
    for c in view.collections:
        c.set_edgecolor("face")
        c.set_rasterized(True)

    phi_i_plot = np.pi
    xy = (phi_i_plot, theta_i)
    xytext = (phi_i_plot-0.3, theta_i+0.1)
    ax1.plot(xy[0], xy[1], 'wx', ms='14', mew=3)
    ax1.annotate('$\omega_i$', xy=xy, textcoords='data', color='w', fontweight='black', size='14', xytext=xytext)

    if markers:
        for m_idx, marker in enumerate(markers):
            marker_xy = (np.radians(marker[1]), np.radians(marker[0]))
            marker_col = 'C%i' % [0, 2, 1][m_idx]
            ax1.plot(marker_xy[0], marker_xy[1], 'x', ms='14', mew=5, color=marker_col)

    dr = 0.16
    if transmission:
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

def plot_microfacet_sd(fig, ax1, zenith_i, zenith_o, alpha_u, alpha_v, eta):
    theta_i = np.radians(zenith_i)
    theta_o = np.radians(zenith_o)
    mu_i = np.cos(theta_i)
    mu_o = -np.cos(theta_o)

    phi_s_ = np.linspace(-np.pi, np.pi, 500)
    phi_d_ = np.linspace(-np.pi, np.pi, 500)
    phi_d, phi_s = np.meshgrid(phi_s_, phi_d_)

    data = mitsuba.layer.microfacet(-mu_o, -mu_i, phi_s, phi_d, alpha_u, alpha_v, eta)
    vmin = np.min(data)
    vmax = np.max(data)

    extent = [-np.pi, np.pi, -np.pi, np.pi]
    im = ax1.imshow(data, cmap='jet', extent=extent, origin='lower')

    ax1.set_xticks([])
    ax1.set_yticks([])

    [i.set_linewidth(2) for i in ax1.spines.values()]

def plot_layer_fourier(fig, ax1, zenith_i, zenith_o, data):
    data = np.abs(data)
    vmin = np.min(coeffs)+0.001
    vmax = np.max(data)

    extent = [-data.shape[1]//2, data.shape[1]//2, -data.shape[0]//2, data.shape[0]//2]
    im = ax1.imshow(data, cmap='jet', origin='lower', extent=extent)
    ax1.set_xticks([])
    ax1.set_yticks([])

    [i.set_linewidth(2) for i in ax1.spines.values()]

eta = 0+1j
alpha_u = 0.05
alpha_v = 0.3

zenith_i = 30
zenith_os = [25, 45, 65]

markers = [
    [zenith_os[0], 0], [zenith_os[1], 0], [zenith_os[2], 0]
]

azimuths = [0, 45, 90, 135]

def plot_bsdf(idx, display=False):
    plt.figure(figsize=(5, 5))
    outer = gridspec.GridSpec(1, 1)

    gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.32, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
    ax1 = plt.subplot(gs[0], projection='polar')
    p = plot_microfacet_polar(ax1, zenith_i, azimuths[idx], alpha_u, alpha_v, eta, False, markers=markers)

    if display:
        plt.show()
    else:
        plt.savefig("bsdf_" + str(idx) + ".pdf", bbox_inches='tight')
        plt.close()

for i in range(4):
    plot_bsdf(i, display)

def plot_slice(idx, display=False):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    plot_microfacet_sd(fig, ax1, zenith_i, zenith_os[idx], alpha_u, alpha_v, eta)

    if display:
        plt.show()
    else:
        plt.savefig("slice_" + str(idx) + ".pdf", bbox_inches='tight')
        plt.close()

for i in range(3):
    plot_slice(i, display)

n, ms, md = mitsuba.layer.microfacet_parameter_heuristic(alpha_u, alpha_v, eta)
mu, w = mitsuba.core.quad.gauss_lobatto(n)
layer = mitsuba.layer.Layer(mu, w, ms, md)
layer.set_microfacet(eta, alpha_u, alpha_v)

storage = mitsuba.layer.BSDFStorage.from_layer("tmp.bsdf", layer, 1e-10)

coeffs = []
ms = 0
mdh = 0
for idx in range(3):
    theta_i = np.radians(zenith_i)
    theta_o = np.radians(zenith_os[idx])
    mu_i = np.cos(theta_i)
    mu_o = -np.cos(theta_o)
    c = storage.fourier_slice_interpolated(mu_i, mu_o, 0)
    coeffs.append(c)
    if c.shape[0] > ms:
        ms = c.shape[0]
    if c.shape[1] > mdh:
        mdh = c.shape[1]

mdh_cut = 40
for idx in range(3):
    tmp = np.zeros((ms, 2*mdh-1))
    dd = coeffs[idx].shape[1]
    tmp[:, mdh-1:mdh+dd-1] = coeffs[idx]
    tmp[:, mdh-dd:mdh-1] = np.flipud(np.fliplr(coeffs[idx][:, 1:]))
    coeffs[idx] = tmp[:, mdh-40+1:mdh+40]

def plot_fourier(idx, display=False):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    plot_layer_fourier(fig, ax1, zenith_i, zenith_os[idx], coeffs[idx])

    if display:
        plt.show()
    else:
        plt.savefig("fourier_" + str(idx) + ".pdf", bbox_inches='tight')
        plt.close()

for i in range(3):
    plot_fourier(i, display)
