import os
import numpy as np
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
import mitsuba.layer


def get_slices(storage, storage_ref, in_zenith, n_samples=120):
    mu_i = np.cos(np.radians(in_zenith))
    mu_o_  = -np.cos(np.radians(np.linspace(0, 90.0, n_samples//2)))

    phi_s = 0.0
    phi_d_ = np.linspace(-np.pi, np.pi, n_samples)
    phi_d, mu_o = np.meshgrid(phi_d_, mu_o_)

    phi_i = (phi_s - phi_d) / 2
    phi_o = (phi_s + phi_d) / 2

    bsdf_slice = storage.eval(mu_i, mu_o, phi_i, phi_o)
    bsdf_slice_ref = storage_ref.eval(mu_i, mu_o, phi_i, phi_o)

    return bsdf_slice, bsdf_slice_ref

def rmse(bsdf_slice, bsdf_slice_ref):
    N = bsdf_slice.shape[0]*bsdf_slice.shape[1]
    diff = bsdf_slice - bsdf_slice_ref
    mse = np.sum(diff*diff)
    return np.sqrt(mse / N)

def plot(storage_list, storage_ref_list, n_zeniths=30, materials=None, title='', path='', zenith_max=90.0, digits=2):
    zeniths = np.linspace(0.0, zenith_max, n_zeniths)

    rmse_list = []

    for m in range(len(storage_list)):
        storage = storage_list[m]
        storage_ref = storage_ref_list[m]

        rmse_m = np.zeros(n_zeniths)
        for i, in_zenith in enumerate(zeniths):
            bsdf_slice, bsdf_slice_ref = get_slices(storage, storage_ref, in_zenith)

            rmse_m[i] = rmse(bsdf_slice, bsdf_slice_ref)

        rmse_list.append(rmse_m)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,3))

    k = len(rmse_list)

    colors = ['g', 'b']
    for m in range(k):
        ax.plot(zeniths, rmse_list[m], label=(materials[m] if materials else ''), lw=2, c=colors[m])
#     ax.set_title(title)
    ax.set_xlim([0, 90])

    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1]])

    xticks = np.linspace(0, 90, 6)
    xtick_labels = [("%d˚" % d) for d in xticks]

    yticks = ax.get_yticks()
    if digits == 1:
        ytick_labels = [("%.1f" % f) for f in yticks]
    elif digits == 2:
        ytick_labels = [("%.2f" % f) for f in yticks]
    else:
        ytick_labels = [("%.3f" % f) for f in yticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=xtick_labels, fontsize='14')
    ax.set_yticklabels(labels=ytick_labels, fontsize='14')
#     ax.set_xlabel(r"$\theta_i$", fontsize='14')
    ax.legend(loc='upper center', fontsize='14')

    plt.tight_layout()
    outname = '%s%s_error.pdf' % (path, title)
    plt.savefig(outname, bbox_inches='tight')
    print("Save \"%s\"" % outname)
    # plt.show()
    plt.close()

add_goldpaper_ref = mitsuba.layer.BSDFStorage("measured_addition/goldpaper/layer_020_004.bsdf")
add_goldpaper     = mitsuba.layer.BSDFStorage("measured_addition/goldpaper/layer_020_004_added.bsdf")

sub_goldpaper_ref = mitsuba.layer.BSDFStorage("measured_subtraction/goldpaper/goldpaper0.bsdf")
sub_goldpaper     = mitsuba.layer.BSDFStorage("measured_subtraction/goldpaper/goldpaper0_subtracted.bsdf")

add_cardboard_ref = mitsuba.layer.BSDFStorage("measured_addition/cardboard/layer_020_028.bsdf")
add_cardboard     = mitsuba.layer.BSDFStorage("measured_addition/cardboard/layer_020_028_added.bsdf")

sub_cardboard_ref = mitsuba.layer.BSDFStorage("measured_subtraction/cardboard/blue-cardboard.bsdf")
sub_cardboard     = mitsuba.layer.BSDFStorage("measured_subtraction/cardboard/blue-cardboard_subtracted.bsdf")

plot([add_cardboard, add_goldpaper],
     [add_cardboard_ref, add_goldpaper_ref],
     materials=['Matte Cardboard', 'Metallic Paper'],
     title='Addition',
     path='measured_addition/',
     n_zeniths=150, zenith_max=90.0, digits=3)

plot([sub_cardboard, sub_goldpaper],
     [sub_cardboard_ref, sub_goldpaper_ref],
     materials=['Matte Cardboard', 'Metallic Paper'],
     title='Subtraction',
     path='measured_subtraction/',
     n_zeniths=150, zenith_max=90.0, digits=1)

del add_goldpaper, add_goldpaper_ref, sub_goldpaper, sub_goldpaper_ref
del add_cardboard, add_cardboard_ref, sub_cardboard, sub_cardboard_ref
