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
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.colors as colors

import os

class MyNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin, vmax):
        matplotlib.colors.Normalize.__init__(self)
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, value, clip=False):
        return np.ma.masked_array(np.where(value > 0, 0.5 + 0.5*value/self.vmax, 0.5 - 0.5*value/self.vmin))

def plot_layer(ax1, layer, zenith_i=30.0, azimuth_i=0.0, transmission=False, levels=None, clamp=True, center=True, vmin=None, vmax=None):
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

    data = layer.eval(mu_o, mu_i, phi_s, phi_d, clamp)
#     data *= np.abs(mu_o)
    storage = mitsuba.layer.BSDFStorage.from_layer("tmp.bsdf", layer, 1e-8)
    data = storage.eval(mu_i, mu_o, 0.5*(phi_s - phi_d), 0.5*(phi_s + phi_d), clamp)
    data /= np.abs(mu_o)

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

    if center and not (vmin and vmax) and clamp:
        vmin = 0.001
        vmax = np.max(data)
        norm = None
    if center and not (vmin and vmax):
        vmin = np.min(data)
        vmax = np.max(data)
        norm = MyNormalize(vmin, vmax)
    elif center:
        norm = MyNormalize(vmin, vmax)
    else:
        vmin = np.min(data)
        vmax = np.max(data)
        norm = None

    theta_o_plot = theta_o if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_o
    view = ax1.contourf(phi_o-phi_i, theta_o_plot, data, 200, cmap='coolwarm', norm=norm, levels=levels, vmin=vmin, vmax=vmax)
    out_info = (phi_o-phi_i, theta_o_plot, data)
    for c in view.collections:
        c.set_edgecolor("face")
        c.set_rasterized(True)

    theta_i_mark = theta_i
    phi_i_plot = np.pi
    if component == 'Ttb' or component == 'Tbt':
        theta_i_mark = np.pi - theta_i
    theta_i_plot = theta_i_mark if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_i_mark
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

    return view, out_info

def plot_microfacet(ax1, zenith_i=30.0, azimuth_i=0.0, alpha_u=0.1, alpha_v=0.3, eta=1.5, transmission=False):
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

    data = mitsuba.layer.microfacet_fresnel(-mu_o, -mu_i, phi_s, phi_d, alpha_u, alpha_v, eta, True)
#     data *= np.abs(mu_o)

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

    vmin = 0.001
    vmax = np.max(data)
    norm = MyNormalize(vmin=vmin, vmax=vmax)

    phi_i_plot = np.pi
    theta_o_plot = theta_o if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_o
    view = ax1.contourf(phi_o-phi_i, theta_o_plot, data, 200, cmap='coolwarm', norm=norm, vmin=vmin, vmax=vmax)
    out_info = (phi_o-phi_i_plot, theta_o_plot, data)
    for c in view.collections:
        c.set_edgecolor("face")
        c.set_rasterized(True)

    theta_i_mark = theta_i
    phi_i_plot = np.pi
    if component == 'Ttb' or component == 'Tbt':
        theta_i_mark = np.pi - theta_i

    theta_i_plot = theta_i_mark if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_i_mark
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

    return view, out_info

alpha_u = 0.2
alpha_v = 0.2
eta = 1.5

n, ms, md = mitsuba.layer.microfacet_parameter_heuristic(alpha_u, alpha_v, eta)
mu, w = mitsuba.core.quad.gauss_lobatto(n)

exponential = mitsuba.layer.Layer(mu, w, ms, md)
exponential.set_microfacet(eta, alpha_u, alpha_v, component=1)

fresnel_old = mitsuba.layer.Layer(mu, w, ms, md)
fresnel_old.set_microfacet(eta, alpha_u, alpha_v, component=3, svd_reg=True)

fresnel_new = mitsuba.layer.Layer(mu, w, ms, md)
fresnel_new.set_microfacet(eta, alpha_u, alpha_v, component=3, svd_reg=False)

# Exponential plot

plt.figure(figsize=(5,5))

outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)

gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
ax21 = plt.subplot(gs2[0], projection='polar')
ax22 = plt.subplot(gs2[1])
ax22.set_aspect(14.0)

zenith_i = 150
important_region = 1e-3
transmission = True

p2, out_info = plot_layer(ax21, exponential, zenith_i=zenith_i, clamp=True, center=True, transmission=transmission)
ax21.contour(out_info[0], out_info[1], out_info[2], levels=[important_region], colors='k', linewidths=0.8, alpha=0.4)
cb = plt.colorbar(p2, cax=ax22)
ax21.set_rlim([0, 0.5*np.pi])
cb.set_ticks([0, 0.95])

plt.savefig("exp.pdf", bbox_inches='tight')
plt.show()
# plt.close()


# Fresnel Plot

plt.figure(figsize=(5,5))

outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)

gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
ax31 = plt.subplot(gs3[0], projection='polar')
ax32 = plt.subplot(gs3[1])
ax32.set_aspect(14.0)

zenith_i = 150
transmission = True

p3, _ = plot_microfacet(ax31, zenith_i=zenith_i, alpha_u=alpha_u, alpha_v=alpha_v, eta=eta, transmission=transmission)
ax31.contour(out_info[0], out_info[1], out_info[2], levels=[important_region], colors='k', linewidths=0.8, alpha=0.4)
cb = plt.colorbar(p3, cax=ax32)
ax31.set_rlim([0, 0.5*np.pi])
cb.set_ticks([0.0, 0.8])

plt.savefig("fresnel.pdf", bbox_inches='tight')
plt.show()
# plt.close()


# QR Regularization

plt.figure(figsize=(5,5))

outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)

gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
ax11 = plt.subplot(gs1[0], projection='polar')
ax12 = plt.subplot(gs1[1])
ax12.set_aspect(14.0)

p1, out_info_2 = plot_layer(ax11, fresnel_new, zenith_i=zenith_i, clamp=False, center=True, transmission=transmission)
ax11.contour(out_info[0], out_info[1], out_info[2], levels=[important_region], colors='k', linewidths=0.8, alpha=0.4)
cb = plt.colorbar(p1, cax=ax12)
cb.set_ticks([0.8, 0, -0.25])
ax11.set_rlim([0, 0.5*np.pi])

plt.savefig("qr.pdf", bbox_inches='tight')
plt.show()
# plt.close()


# SVD Regularization

plt.figure(figsize=(5,5))

outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)

gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
ax01 = plt.subplot(gs0[0], projection='polar')
ax02 = plt.subplot(gs0[1])
ax02.set_aspect(14.0)

p0, _ = plot_layer(ax01, fresnel_old, zenith_i=zenith_i, clamp=False, center=True, transmission=transmission, levels=p1.levels,
                   vmin=np.min(out_info_2[2]), vmax=np.max(out_info_2[2]))
ax01.contour(out_info[0], out_info[1], out_info[2], levels=[important_region], colors='k', linewidths=0.8, alpha=0.4)
cb = plt.colorbar(p0, cax=ax02)
cb.set_ticks([0.8, 0, -0.25])
ax01.set_rlim([0, 0.5*np.pi])

plt.savefig("svd.pdf", bbox_inches='tight')
plt.show()
# plt.close()
