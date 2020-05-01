import os
import numpy as np
import scipy.sparse
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
    os.mkdir('measured_addition')
    os.mkdir('measured_addition/cardboard')
    os.mkdir('measured_addition/goldpaper')
except:
    pass

measurement_names = {
    4: "goldpaper0",
    20: "matte-film-01",
    22: "layer_020_004",
    28: "blue-cardboard",
    29: "layer_020_028",
}

n = 120
ms = 1
md = 301

mu, w = mitsuba.core.quad.gauss_lobatto(n)

def add_measurements(prefix, base_mid, top_mid, ref_mid, reverse_top_layer=False):
    top_name = measurement_names[top_mid]
    base_name = measurement_names[base_mid]
    ref_name = measurement_names[ref_mid]

    base_layer = mitsuba.layer.Layer(mu, w, ms, md)
    top_layer = mitsuba.layer.Layer(mu, w, ms, md)
    ref_layer = mitsuba.layer.Layer(mu, w, ms, md)

    Rt_1 = scipy.sparse.load_npz("measurements/" + top_name + "/" + str(top_mid) + "_Rt.npz")
    Ttb_1 = scipy.sparse.load_npz("measurements/" + top_name + "/" + str(top_mid) + "_Ttb.npz")
    Rb_1 = scipy.sparse.load_npz("measurements/" + top_name + "/" + str(top_mid) + "_Rb.npz")
    Tbt_1 = scipy.sparse.load_npz("measurements/" + top_name + "/" + str(top_mid) + "_Tbt.npz")

    Rt_2 = scipy.sparse.load_npz("measurements/" + base_name + "/" + str(base_mid) + "_Rt.npz")

    Rt_ref = scipy.sparse.load_npz("measurements/" + ref_name + "/" + str(ref_mid) + "_Rt.npz")

    assert(Rt_1.shape[0] == (ms+md-1)*n//2)
    assert(Ttb_1.shape[0] == (ms+md-1)*n//2)
    assert(Rb_1.shape[0] == (ms+md-1)*n//2)
    assert(Tbt_1.shape[0] == (ms+md-1)*n//2)
    assert(Rt_2.shape[0] == (ms+md-1)*n//2)
    assert(Rt_ref.shape[0] == (ms+md-1)*n//2)

    base_layer.set_reflection_top(Rt_2)
    ref_layer.set_reflection_top(Rt_ref)

    top_layer.set_reflection_top(Rt_1)
    top_layer.set_reflection_bottom(Rb_1)
    top_layer.set_transmission_top_bottom(Ttb_1)
    top_layer.set_transmission_bottom_top(Tbt_1)
    if reverse_top_layer:
        top_layer.reverse()

    added_layer = mitsuba.layer.Layer.add(top_layer, base_layer, epsilon=1e-6)
    added_name = ref_name + "_added"

    filename = "measured_addition/" + prefix + added_name + ".bsdf"
    added_storage = mitsuba.layer.BSDFStorage.from_layer(filename, added_layer, 1e-8)

    filename = "measured_addition/" + prefix + base_name + ".bsdf"
    base_storage = mitsuba.layer.BSDFStorage.from_layer(filename, base_layer, 1e-8)

    filename = "measured_addition/" + prefix + top_name + ".bsdf"
    top_storage = mitsuba.layer.BSDFStorage.from_layer(filename, top_layer, 1e-8)

    filename = "measured_addition/" + prefix + ref_name + ".bsdf"
    ref_storage = mitsuba.layer.BSDFStorage.from_layer(filename, ref_layer, 1e-8)

    layers = [base_storage, top_storage, added_storage, ref_storage]
    names = [base_name, top_name, added_name, ref_name]

    return layers, names

def generate_plots(prefix, layers, names, zenith_i=30):
    base = layers[0]; top = layers[1]; added = layers[2]; ref = layers[3];
    base_name = names[0]; top_name = names[1]; added_name = names[2]; ref_name = names[3];

    layers = [base, added, ref]
    names = ['base', 'added', 'ref']

    try:
        os.mkdir("figures/" + base_name + "_adding/")
    except:
        pass

    for i in range(len(layers)):
        plt.figure(figsize=(5,5))
        outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
        ax1 = plt.subplot(gs[0], projection='polar')
        p2, out_info = plot_layer(ax1, layers[i], zenith_i=zenith_i, clamp=True)

        if i == 1 or i == 2:
            ax2 = plt.subplot(gs[1])
            ax2.set_aspect(14.0)
            cb = plt.colorbar(p2, cax=ax2, format='%.2f')

        plt.savefig("measured_addition/" + prefix + names[i] + "_Rt.pdf", bbox_inches='tight')
        plt.close()

    transmissions = [False, False, True, True]
    zeniths = [zenith_i, 180-zenith_i, zenith_i, 180-zenith_i]
    components = ['Rt', 'Rb', 'Ttb', 'Tbt']

    for i in range(4):
        plt.figure(figsize=(5,5))
        outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
        ax1 = plt.subplot(gs[0], projection='polar')
        p2, out_info = plot_layer(ax1, top, zenith_i=zeniths[i], clamp=True, transmission=transmissions[i])

        plt.savefig("measured_addition/" + prefix + "top_" + components[i] + ".pdf", bbox_inches='tight')
        plt.close()

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
    view = ax1.contourf(phi_o-phi_i, theta_o_plot, data, 200, cmap='jet')
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


zenith_i = 30.0

# Blue cardboard + matte film

top_mid = 20
base_mid = 28
ref_mid = 29

layers, names = add_measurements("cardboard/", base_mid, top_mid, ref_mid)
generate_plots("cardboard/", layers, names, zenith_i)
for s in layers:
    s.close()


# Gold paper + matte film

top_mid = 20
base_mid = 4
ref_mid = 22

layers, names = add_measurements("goldpaper/", base_mid, top_mid, ref_mid, reverse_top_layer=True)
generate_plots("goldpaper/", layers, names, zenith_i)
for s in layers:
    s.close()
