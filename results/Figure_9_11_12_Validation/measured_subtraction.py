import os
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import identity
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

# Requirement: brew install suite-sparse, pip install sparseqr
import sparseqr

try:
    os.mkdir('measured_subtraction')
    os.mkdir('measured_subtraction/cardboard')
    os.mkdir('measured_subtraction/goldpaper')
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
mu = np.array(mu); w = np.array(w)
size = n // 2
blocks = ms+md-1

def subtract_measurements(added, top, eps=0):
    def mk_diagonal(x):
        return scipy.sparse.csc_matrix(scipy.sparse.spdiags(x, 0, x.shape[0], x.shape[0]))

    def modify_system(A, b, eps, R = None):
        if R is not None:
            A = scipy.sparse.vstack([A, R*eps])
        else:
            A = scipy.sparse.vstack([A, mk_diagonal(weights)*eps])
        b = scipy.sparse.vstack([b, scipy.sparse.csc_matrix(b.shape, dtype=np.float64)])
        return A, b

    def sparsesolve_qr(A, b, eps = 0, R = None):
        if eps > 0:
            A, b = modify_system(A, b, eps, R)
        return scipy.sparse.csc_matrix(sparseqr.solve(A, b))

    weights = np.ones(n//2*(md+ms-1))

    Ttb_1_i = sparsesolve_qr(Ttb_1, scipy.sparse.eye(Rt_1.shape[0]), eps)
    Tbt_1_i = sparsesolve_qr(Tbt_1, scipy.sparse.eye(Rt_1.shape[0]), eps)

    X = Tbt_1_i @ (Rt_add - Rt_1) @ Ttb_1_i

    I = scipy.sparse.eye(Rt_1.shape[0])
    Y = sparsesolve_qr(I + Rb_1 @ X, I)
    Z = X @ Y

    Z = scipy.sparse.csc_matrix(Z)

    # For \mu_i close to normal incidence, the problem gets even trickier,
    # as \mu_o loses some of its meaning (and vice-versa).
    # Thus, we propose to rely on polynomial extrapolation at these last few rows/columns of the scattering matrices.
    # Special care has to be taken to keep track of the correct integration/cosine weights for each column.
    weights_half = np.split(w, 2)[1]
    mus_half = np.abs(np.split(mu, 2)[1])

    inv_col_weights = scipy.sparse.csc_matrix(1.0/np.tile(weights_half * mus_half, ms+md-1))
    zero_col = scipy.sparse.csc_matrix((Z.shape[0], 1), dtype=np.float64)

    nn = 5  # last 'nn' columns will be extrapolated

    x = mus_half[0:-nn]
    weights = weights_half[0:-nn]
    inv_weights = 1.0/(x*weights)

    # Extrapolate with polynomial

    deg = 8
    for block in range(blocks):
        start_col = block * size

        y = Z[:,start_col:start_col+size-nn].todense().T
        for j in range(y.shape[0]):
            col = block * size + j
            y[j, :] /= w[size+j]
        fit = np.polyfit(x, y, deg)
        new_points = mus_half[-nn:]

        vals = np.vander(new_points, N=deg+1).T

        new_cols = vals.T @ fit

        for j in range(nn):
            col = col = block * size + size - nn + j
            tmp = scipy.sparse.csc_matrix(new_cols[j, :])
            Z[:, col] = tmp.T * w[-nn+j]

    # Mirror results to last 'nn' rows
    col_weights = scipy.sparse.csc_matrix(np.tile(weights_half * mus_half, ms+md-1))
    col_weights_dense = col_weights.todense()

    for block in range(blocks):
        for j in np.arange(size-nn, size):
            col = block * size + j

            tmp1 = (Z[:, col] / (w[size + j] * np.abs(mu[size + j]))).todense().T
            tmp = scipy.sparse.csc_matrix(np.multiply(tmp1, col_weights_dense))
            Z[col, :] = tmp

    # Another tricky part is mu_i ~= 0. Mirror results to first two columns
    inv_col_weights = scipy.sparse.csc_matrix(np.tile(1.0 / (weights_half * mus_half), ms+md-1))
    inv_col_weights_dense = inv_col_weights.todense()

    for block in range(blocks):
        for i in np.arange(2):
            row = block * size + i

            tmp1 = Z[row, :].todense()
            tmp2 = scipy.sparse.csc_matrix(np.multiply(tmp1, inv_col_weights_dense)).T
            Z[:, row] = tmp2 * weights_half[i] * mus_half[i]

    Z = scipy.sparse.csr_matrix(Z)

    def prune(A, eps=1e-8):
        mask = np.abs(A) > 1e-8
        return A.multiply(mask)

    Z = prune(Z)

    subtracted = mitsuba.layer.Layer(mu, w, ms, md)
    subtracted.set_reflection_top(Z.todense())

    return subtracted

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

def generate_plots(prefix, layers, names, zenith_i=30.0):
    added_name = names[0]; subtracted_name = names[1]; ref_name = names[2]; top_name = names[3];
    names = ['added', 'subtracted', 'ref']

    filename = "measured_subtraction/" + prefix + added_name + ".bsdf"
    added_storage = mitsuba.layer.BSDFStorage.from_layer(filename, layers[0], 1e-8)

    filename = "measured_subtraction/" + prefix + subtracted_name + ".bsdf"
    subtracted_storage = mitsuba.layer.BSDFStorage.from_layer(filename, layers[1], 1e-8)

    filename = "measured_subtraction/" + prefix + ref_name + ".bsdf"
    ref_storage = mitsuba.layer.BSDFStorage.from_layer(filename, layers[2], 1e-8)

    filename = "measured_subtraction/" + prefix + top_name + ".bsdf"
    top_storage = mitsuba.layer.BSDFStorage.from_layer(filename, layers[3], 1e-8)

    storages = [
        added_storage,
        subtracted_storage,
        ref_storage,
        top_storage
    ]

    for i in range(len(layers) - 1):
        plt.figure(figsize=(5,5))
        outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
        ax1 = plt.subplot(gs[0], projection='polar')
        p2, out_info = plot_layer(ax1, storages[i], zenith_i=zenith_i, clamp=True)

        if i == 1 or i == 2:
            ax2 = plt.subplot(gs[1])
            ax2.set_aspect(14.0)
            cb = plt.colorbar(p2, cax=ax2, format='%.2f')

        plt.savefig("measured_subtraction/" + prefix + names[i] + "_Rt.pdf", bbox_inches='tight')
        plt.close()

    transmissions = [False, False, True, True]
    zeniths = [zenith_i, 180-zenith_i, zenith_i, 180-zenith_i]
    components = ['Rt', 'Rb', 'Ttb', 'Tbt']

    for i in range(4):
        plt.figure(figsize=(5,5))
        outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.0)
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, width_ratios=[1.0, 0.06], subplot_spec = outer[0])
        ax1 = plt.subplot(gs[0], projection='polar')
        p2, out_info = plot_layer(ax1, storages[3], zenith_i=zeniths[i], clamp=True, transmission=transmissions[i])

        plt.savefig("measured_subtraction/" + prefix + "top_" + components[i] + ".pdf", bbox_inches='tight')
        plt.close()

    for s in storages:
        s.close()


zenith_i = 30.0

# Goldpaper

Rt_2 = scipy.sparse.load_npz("measurements/goldpaper0/4_Rt_mirror.npz")
Rt_add = scipy.sparse.load_npz("measurements/layer_020_004/22_Rt_mirror.npz")

Rt_1 = scipy.sparse.load_npz("measurements/matte-film-01/20_Rb_mirror.npz")
Ttb_1 = scipy.sparse.load_npz("measurements/matte-film-01/20_Tbt_mirror.npz")
Rb_1 = scipy.sparse.load_npz("measurements/matte-film-01/20_Rt_mirror.npz")
Tbt_1 = scipy.sparse.load_npz("measurements/matte-film-01/20_Ttb_mirror.npz")

top = mitsuba.layer.Layer(mu, w, ms, md)
top.set_reflection_top(Rt_1)
top.set_reflection_bottom(Rb_1)
top.set_transmission_top_bottom(Ttb_1)
top.set_transmission_bottom_top(Tbt_1)

ref = mitsuba.layer.Layer(mu, w, ms, md)
ref.set_reflection_top(Rt_2)

added = mitsuba.layer.Layer(mu, w, ms, md)
added.set_reflection_top(Rt_add)

subtracted = subtract_measurements(added, top, 0.06)
generate_plots("goldpaper/",
               [added, subtracted, ref, top],
               ["layer_020_004", "goldpaper0_subtracted", "goldpaper0", "matte-film-01"],
               zenith_i)


# Cardboard

Rt_2 = scipy.sparse.load_npz("measurements/blue-cardboard/28_Rt_mirror.npz")
Rt_add = scipy.sparse.load_npz("measurements/layer_020_028/29_Rt_mirror.npz")

Rt_1 = scipy.sparse.load_npz("measurements/matte-film-01/20_Rt_mirror.npz")
Ttb_1 = scipy.sparse.load_npz("measurements/matte-film-01/20_Ttb_mirror.npz")
Rb_1 = scipy.sparse.load_npz("measurements/matte-film-01/20_Rb_mirror.npz")
Tbt_1 = scipy.sparse.load_npz("measurements/matte-film-01/20_Tbt_mirror.npz")

top = mitsuba.layer.Layer(mu, w, ms, md)
top.set_reflection_top(Rt_1)
top.set_reflection_bottom(Rb_1)
top.set_transmission_top_bottom(Ttb_1)
top.set_transmission_bottom_top(Tbt_1)

ref = mitsuba.layer.Layer(mu, w, ms, md)
ref.set_reflection_top(Rt_2)

added = mitsuba.layer.Layer(mu, w, ms, md)
added.set_reflection_top(Rt_add)

subtracted = subtract_measurements(added, top, 0.09)
generate_plots("cardboard/",
               [added, subtracted, ref, top],
               ["layer_020_028", "blue-cardboard_subtracted", "blue-cardboard", "matte-film-01"],
               zenith_i)
