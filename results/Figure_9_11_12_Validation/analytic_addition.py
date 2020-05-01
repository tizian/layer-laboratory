import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.colors as colors
import matplotlib.cm as cm

try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
import mitsuba.layer

try:
    os.mkdir('analytic_addition')
except:
    pass

# First, we need to compute the reference distributions for the layered material
# with brute force MC. This will take a while ...

zenith_i = 30
N = 50000000 # MC samples

azimuths = [0, 45, 90, 135]
for azimuth_i in azimuths:
    print("")
    print("Compute reference distribution with brute-force MC ({}˚) ...".format(azimuth_i))
    print("\n")

    args = "-t1 -Dzenith={} -Dazimuth={} -Dsamples={}".format(zenith_i, azimuth_i, N)
    os.system("mitsuba anisotropic_addition_mc_reference.xml {}".format(args))
    name = "aniso_addition_ref_{}".format(azimuth_i)
    os.system("mv {}.txt analytic_addition/{}.txt".format(name, name))
    os.system("rm anisotropic_addition_mc_reference.exr".format(name))

# Load the data, normalize it properly and reshape
mc_data = []
for azimuth in azimuths:
    data = np.genfromtxt(('analytic_addition/aniso_addition_ref_%d.txt' % azimuth), delimiter=' ')
    n = int(np.sqrt(data.shape[0]))
    data = data.reshape((n, n)).T / N
    data *= (n*n)/(np.pi*np.pi)
    mc_data.append(data)

# Add anisotropic conductor and dielectric layers
print("")
print("Precompute layers ...")

alpha_u_top = 0.1
alpha_v_top = 0.2
eta_top = 1.5

alpha_u_bot = 0.2
alpha_v_bot = 0.1
eta_bot = 0+1j

n1, ms1, md1 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_top, alpha_v_top, eta_top)
n2, ms2, md2 = mitsuba.layer.microfacet_parameter_heuristic(alpha_u_bot, alpha_v_bot, eta_bot)
n = max(n1, n2)
ms = max(ms1, ms2)
md = md1

mu, w = mitsuba.core.quad.gauss_lobatto(n)

top = mitsuba.layer.Layer(mu, w, ms, md)
top.set_microfacet(eta_top, alpha_u_top, alpha_v_top)

base = mitsuba.layer.Layer(mu, w, ms, md)
base.set_microfacet(eta_bot, alpha_u_bot, alpha_v_bot)

added = mitsuba.layer.Layer.add(top, base, 1e-6)

filename = "analytic_addition/aniso_addition_base.bsdf"
storage = mitsuba.layer.BSDFStorage.from_layer(filename, base, 1e-4)
storage.close()

filename = "analytic_addition/aniso_addition_added.bsdf"
storage = mitsuba.layer.BSDFStorage.from_layer(filename, added, 1e-4)
storage.close()

filename = "analytic_addition/aniso_addition_top.bsdf"
storage = mitsuba.layer.BSDFStorage.from_layer(filename, top, 1e-4)
storage.close()

# Plotting
def plot_layer(ax1, storage, zenith_i=30.0, azimuth_i=0.0, transmission=False, clamp=True, levels=None, center=False, vmin=None, vmax=None, black_text=False):
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

    data = storage.eval(mu_i, mu_o, 0.5*(phi_s - phi_d), 0.5*(phi_s + phi_d), clamp)

    ax1.grid(linestyle='-', linewidth=0.6, alpha=0.3, color='w')

    text_col = 'w'
    ax1.set_rgrids(np.radians(theta_ticks_deg), labels=theta_labels, angle=270, color=text_col, fontweight='ultralight', size='10', ha='center', alpha=0.8)

    phi_ticks_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
    phi_labels = [("%d˚" % i) for i in phi_ticks_deg]
    phi_labels[0] = ""; phi_labels[4] = ""
    phi_ticks_deg -= np.degrees(phi_i)
    phi_ticks_deg = np.where(phi_ticks_deg < 0, phi_ticks_deg + 360, phi_ticks_deg)
    phi_ticks_deg = np.where(phi_ticks_deg > 360, phi_ticks_deg - 360, phi_ticks_deg)
    ax1.set_thetagrids(phi_ticks_deg, labels=phi_labels, color='k', fontweight='ultralight', size='10', ha='center', alpha=0.8)

    phi_i_plot = np.pi
    theta_o_plot = theta_o if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_o
    view = ax1.contourf(phi_o-phi_i, theta_o_plot, data, 200, cmap='jet', vmax=np.max(data))
    out_info = (phi_o-phi_i_plot, theta_o_plot, data)
    for c in view.collections:
        c.set_edgecolor("face")
        c.set_rasterized(True)

    if component == 'Rt' or component == 'Rb':
        theta_i_plot = theta_i if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_i
        xy = (phi_i_plot, np.abs(theta_i_plot))
        xytext = (phi_i_plot-0.2, np.abs(theta_i_plot)+0.1)
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

def plot_data(ax1, data, zenith_i=30.0, azimuth_i=0.0, transmission=False, markers=None, vmax=None):
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

    azimuths = np.linspace(0, 360, 128)
    if component == 'Rt' or component == 'Tbt':
        zeniths = np.linspace(0, 90, 128)
        theta_ticks_deg = [10, 30, 50, 70, 90]
        theta_labels = ['0˚', '', '', '', '90˚']
    elif component == 'Rb' or component == 'Ttb':
        zeniths = np.linspace(180, 90, 128)
        theta_ticks_deg = [180, 160, 140, 120, 100]
        theta_labels = ['90˚', '', '', '', '180˚']

    theta_o, phi_o = np.meshgrid(np.radians(zeniths), np.radians(azimuths))
    mu_o = -np.cos(theta_o)

    phi_s = phi_o + phi_i
    phi_d = phi_o - phi_i

    ax1.grid(linestyle='-', linewidth=0.6, alpha=0.3, color='w')

    text_col = 'w'
    ax1.set_rgrids(np.radians(theta_ticks_deg), labels=theta_labels, angle=270, color=text_col, fontweight='ultralight', size='10', ha='center', alpha=0.8)

    phi_ticks_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
    phi_labels = [("%d˚" % i) for i in phi_ticks_deg]
    phi_labels[0] = ""; phi_labels[4] = ""
    phi_ticks_deg -= np.degrees(phi_i)
    phi_ticks_deg = np.where(phi_ticks_deg < 0, phi_ticks_deg + 360, phi_ticks_deg)
    phi_ticks_deg = np.where(phi_ticks_deg > 360, phi_ticks_deg - 360, phi_ticks_deg)
    ax1.set_thetagrids(phi_ticks_deg, labels=phi_labels, color='k', fontweight='ultralight', size='10', ha='center', alpha=0.8)

    phi_i_plot = np.pi
    theta_o_plot = theta_o if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_o
    view = ax1.contourf(phi_o-phi_i-np.pi, theta_o_plot, data, 200, cmap='jet', vmin=0, vmax=vmax)
    out_info = (phi_o-phi_i_plot, theta_o_plot, data)
    for c in view.collections:
        c.set_edgecolor("face")
        c.set_rasterized(True)

    if component == 'Rt' or component == 'Rb':
        theta_i_plot = theta_i if component == 'Rt' or component == 'Tbt' else np.radians(270)-theta_i
        xy = (phi_i_plot, np.abs(theta_i_plot))
        xytext = (phi_i_plot-0.2, np.abs(theta_i_plot)+0.1)
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

print("Generate plots ...")

layers = [mitsuba.layer.BSDFStorage("analytic_addition/aniso_addition_base.bsdf"),
          mitsuba.layer.BSDFStorage("analytic_addition/aniso_addition_added.bsdf")]
names = ['base', 'added']

for i in range(len(layers)):
    plt.figure(figsize=(5,5))
    outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
    ax1 = plt.subplot(gs[0], projection='polar')
    p2, out_info = plot_layer(ax1, layers[i], zenith_i=zenith_i, clamp=True)

    plt.savefig("analytic_addition/" + names[i] + "_Rt.pdf", bbox_inches='tight')
    plt.close()

top = mitsuba.layer.BSDFStorage("analytic_addition/aniso_addition_top.bsdf")
transmissions = [False, False, True, True]
zeniths = [zenith_i, 180-zenith_i, zenith_i, 180-zenith_i]
components = ['Rt', 'Rb', 'Ttb', 'Tbt']

for i in range(4):
    plt.figure(figsize=(5,5))
    outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
    ax1 = plt.subplot(gs[0], projection='polar')
    p2, out_info = plot_layer(ax1, top, zenith_i=zeniths[i], clamp=True, transmission=transmissions[i])

    plt.savefig("analytic_addition/top_" + components[i] + ".pdf", bbox_inches='tight')
    plt.close()

data_out = []
for i in range(len(azimuths)):
    plt.figure(figsize=(5,5))
    outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
    ax1 = plt.subplot(gs[0], projection='polar')
    p2, out_info = plot_layer(ax1, layers[1], zenith_i=zenith_i, azimuth_i=azimuths[i], clamp=True)
    data_out.append(out_info[2])

    ax2 = plt.subplot(gs[1])
    ax2.set_aspect(14.0)

    m = plt.cm.ScalarMappable(cmap=cm.jet)
    m.set_array(out_info[2])
    m.set_clim(0., np.max(out_info[2]))
    cb = plt.colorbar(m, cax=ax2, format='%.2f')

    plt.savefig("analytic_addition/added_" + str(azimuths[i]) + ".pdf", bbox_inches='tight')
    plt.close()

for i in range(len(azimuths)):
    plt.figure(figsize=(5,5))
    outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
    ax1 = plt.subplot(gs[0], projection='polar')
    p2 = plot_data(ax1, mc_data[i], zenith_i=zenith_i, azimuth_i=azimuths[i], vmax=np.max(data_out[i]))

    ax2 = plt.subplot(gs[1])
    ax2.set_aspect(14.0)

    m = plt.cm.ScalarMappable(cmap=cm.jet)
    m.set_array(data_out[i])
    m.set_clim(0., np.max(data_out[i]))
    cb = plt.colorbar(m, cax=ax2, format='%.2f')


    plt.savefig("analytic_addition/ref_" + str(azimuths[i]) + ".pdf", bbox_inches='tight')
    plt.close()

layers[0].close()
layers[1].close()
top.close()
print("done.")
