import numpy as np
from numpy import linalg as la

import scipy
import scipy.sparse
from scipy import interpolate as interp

try:
    import mitsuba
except ImportError as error:
    print(error)
    print("Could not import the Mitsuba 2 python modules. Make sure to \"source setpath.sh\" before running this script.")

import mitsuba
mitsuba.set_variant('scalar_rgb')
import mitsuba.layer
import mitsuba.core.spline as sp

names = {
    4: "goldpaper0",
    20: "matte-film-01",
    22: "layer_020_004",
    28: "blue-cardboard",
    29: "layer_020_028",
}

has_transmission = {
    4: False,
    20: True,
    22: False,
    28: False,
    29: False,
}

def incident_to_slice(m):
    return max(0, m-1)

def sph(theta, phi):
    sinTheta = np.sin(theta); cosTheta = np.cos(theta);
    sinPhi = np.sin(phi); cosPhi = np.cos(phi);
    return np.array([
        sinTheta * cosPhi,
        sinTheta * sinPhi,
        cosTheta
    ])

def sphInv(w):
    theta = np.arccos(w[2])
    phi = np.arctan2(w[1], w[0])
    return theta, phi

def sph_scalar_theta(theta, phi):
    sinTheta = np.sin(theta); cosTheta = np.cos(theta);
    sinPhi = np.sin(phi); cosPhi = np.cos(phi);

    cosTheta = np.ones(sinPhi.shape) * cosTheta
    return np.array([
        sinTheta * cosPhi,
        sinTheta * sinPhi,
        cosTheta
    ])

def normalize(v):
    norm = np.linalg.norm(v, axis=(0))
    return v / norm

def halfvector_warp(theta_i, phi_i, theta_h, phi_h, transmission):
    f = np.pi / 180.0
    theta_i = theta_i*f; phi_i = phi_i*f
    theta_h = theta_h*f; phi_h = phi_h*f
    wi = np.array([
        np.sin(theta_i) * np.cos(phi_i),
        np.sin(theta_i) * np.sin(phi_i),
        np.cos(theta_i)
    ])[:, np.newaxis, np.newaxis]
    wh = np.array([
        np.sin(theta_h) * np.cos(phi_h),
        np.sin(theta_h) * np.sin(phi_h),
        np.cos(theta_h)
    ])
    dp = np.sum(wi*wh, axis=0)
    if transmission:
        eta = 1.5
        e = (1/eta)
        sqrt = np.sqrt(1 - e**2 *(1 - dp**2))
        sqrt[sqrt < 0] = 0

        wo = -e*(wi - dp*wh) - wh * sqrt
    else:
        dp = np.sum(wi*wh, axis=0)
        wo = 2 * dp * wh - wi
    f = 180.0 / np.pi
    theta_o = np.arccos(wo[2, ...]) * f
    phi_o = np.arctan2(wo[1, ...], wo[0, ...]) * f
    phi_o = np.where(phi_o < 0, phi_o + 360, phi_o)
    return theta_o, phi_o

def jacobian(theta_i, phi_i, theta_h, phi_h, transmission):
    if transmission:
        return 1

    f = np.pi / 180.0
    theta_i = theta_i*f; phi_i = phi_i*f
    theta_h = theta_h*f; phi_h = phi_h*f
    wi = np.array([
        np.sin(theta_i) * np.cos(phi_i),
        np.sin(theta_i) * np.sin(phi_i),
        np.cos(theta_i)
    ])[:, np.newaxis, np.newaxis]
    wh = np.array([
        np.sin(theta_h) * np.cos(phi_h),
        np.sin(theta_h) * np.sin(phi_h),
        np.cos(theta_h)
    ])
    dp = np.sum(wi*wh, axis=0)
    return 4*np.abs(dp)

def interpolate_slice(data, nodes, mu_i):
    y = 0

    if mu_i < nodes[1]:
        # We are in the special region where we don't even have data, fallback to simpler linear interpolation here!
        alpha = (mu_i - nodes[1]) / (nodes[2] - nodes[1])
        y = (1-alpha) * data[1] + alpha * data[2]
    elif mu_i < nodes[len(nodes)//2]:
        for i in range(len(nodes)-1):
            if nodes[i] < mu_i and nodes[i+1] > mu_i:
                idx0 = i
                idx1 = i + 1
                break

        alpha = (mu_i - nodes[idx0]) / (nodes[idx1] - nodes[idx0])
        y = (1-alpha) * data[idx0] + alpha * data[idx1]
    else:
        _, offset, weights = sp.eval_spline_weights(nodes, mu_i)
        for i in range(4):
            w = weights[i]
            if w == 0:
                continue
            y += data[offset + i] * w
    return y

def preprocess(mid, in_mu, orientation='tb', normal_incidence_fill_degree=8):
    measure_name = names[mid]

    m_max = len(in_mu)-1

    print("Converting measurement \"%s\" (mid: %d, orientation: %s) to scattering matrices.." % (measure_name, mid, orientation))

    n_theta_h = 500
    n_phi_h = 2000

    brdf = np.zeros((len(in_mu), n_theta_h, n_phi_h))
    btdf = np.zeros((len(in_mu), 2*n_theta_h, n_phi_h))

    phi_i = 0.0

    if has_transmission[mid]:
        total = 2*len(in_mu)
    else:
        total = len(in_mu)
    current = 0

    normal_incidence_fill_degree = 8

    for i, mu_i in enumerate(in_mu):
        ori = ('_%s' % orientation) if has_transmission[mid] else ''
        s = incident_to_slice(i)
        name = "measurements/%s/%d%s_%02d.npz" % (measure_name, mid, ori, s)

        data = np.load(name)["r"]
        interpolant = interp.LinearNDInterpolator(data[:, 0:2], data[:, 2])
        interpolant_nn = interp.NearestNDInterpolator(data[:, 0:2], data[:, 2])

        phi_i = 0.0
        mu_i = in_mu[i]
        theta_i = np.rad2deg(np.arccos(mu_i))

        wi = sph(np.deg2rad(theta_i), np.deg2rad(phi_i))

        phi_h_ = np.linspace(0, 360, n_phi_h)
        theta_h_ = np.linspace(0, 90, n_theta_h)
        phi_h, theta_h = np.meshgrid(phi_h_, theta_h_)

        theta_o, phi_o = halfvector_warp(theta_i, phi_i, theta_h, phi_h, False)
        res = interpolant(theta_o, phi_o)

        jac = jacobian(theta_i, phi_i, theta_h, phi_h, False)

        res_nn = interpolant_nn(theta_o, phi_o)
        mask_nn = np.isnan(res)
        res[mask_nn] = res_nn[mask_nn]

        if mu_i == 1.0 and normal_incidence_fill_degree > 0:
            i_off = i-1
            s_peak = incident_to_slice(i_off)

            name_peak = "measurements/%s/%d%s_%02d.npz" % (measure_name, mid, ori, s_peak)
            data_peak = np.load(name_peak)["r"]
            interpolant_peak = interp.LinearNDInterpolator(data_peak[:, 0:2], data_peak[:, 2])
            interpolant_peak_nn = interp.NearestNDInterpolator(data_peak[:, 0:2], data_peak[:, 2])

            mu_i_peak = in_mu[i_off]
            theta_i_peak = np.rad2deg(np.arccos(mu_i_peak))

            theta_o_peak, phi_o_peak = halfvector_warp(theta_i_peak, phi_i, theta_h, phi_h, False)
            res_peak = interpolant_peak(theta_o_peak, phi_o_peak)

            res_peak_nn = interpolant_peak_nn(theta_o_peak, phi_o_peak)
            mask_peak_nn = np.isnan(res_peak)
            res_peak[mask_peak_nn] = res_peak_nn[mask_peak_nn]

            mask_peak = (theta_o < normal_incidence_fill_degree)
            res[mask_peak] = res_peak[mask_peak]

        jac = jacobian(theta_i, phi_i, theta_h, phi_h, False)
        res *= jac

        brdf[i, :] = res

        current += 1

        print ("Pre-process: %.2f%%" % (100*current/total), end="\r")

    if has_transmission[mid]:
        for i, mu_i in enumerate(in_mu):
            ori = ('_%s' % orientation) if has_transmission[mid] else ''
            s = incident_to_slice(i)
            name = "measurements/%s/%d%s_%02d.npz" % (measure_name, mid, ori, s)

            data = np.load(name)["t"]
            interpolant = interp.LinearNDInterpolator(data[:, 0:2], data[:, 2])
            interpolant_nn = interp.NearestNDInterpolator(data[:, 0:2], data[:, 2])

            phi_i = 0.0
            mu_i = in_mu[i]
            theta_i = np.rad2deg(np.arccos(mu_i))
            wi = sph(np.deg2rad(theta_i), np.deg2rad(phi_i))

            phi_h_ = np.linspace(-180, 180, n_phi_h)
            theta_h_ = np.linspace(0, 180, 2*n_theta_h)
            phi_h, theta_h = np.meshgrid(phi_h_, theta_h_)

            theta_o, phi_o = halfvector_warp(theta_i, phi_i, theta_h, phi_h, True)
            mu_o = np.cos(np.deg2rad(theta_o))
            res = interpolant(theta_o, phi_o)

            jac = jacobian(theta_i, phi_i, theta_h, phi_h, True)

            res_nn = interpolant_nn(theta_o, phi_o)
            mask_nn = np.isnan(res)
            res[mask_nn] = res_nn[mask_nn]

            jac = jacobian(theta_i, phi_i, theta_h, phi_h, True)
            res *= jac

            res[mu_o*mu_i >= 0.0] = 0

            btdf[i, :] = res

            current += 1

            print ("Pre-process: %.2f%%" % (100*current/total), end="\r")

    return brdf, btdf

def eval_bsdf(brdf, btdf, mu_i, mu_o, phi_d, nodes):
    transmission = (mu_i*mu_o > 0)

    mu_o = -mu_o # Difference in parameterization
    inv_cos = 1.0 / np.abs(mu_o)

    theta_i = np.arccos(mu_i)
    theta_o = np.arccos(mu_o)

    phi_i = 0
    phi_o = phi_i + np.pi + phi_d

    # Find halfway angle to lookup brdf values
    wi = sph_scalar_theta(theta_i, phi_i)
    wo = sph_scalar_theta(theta_o, phi_o)

    if transmission:
        h = -normalize(wi[..., np.newaxis] + wo*1.5)
    else:
        h = normalize(wi[..., np.newaxis] + wo)

    theta_h_rad, phi_h_rad = sphInv(h)
    theta_h = np.rad2deg(theta_h_rad); phi_h = np.rad2deg(phi_h_rad)
    if not transmission:
        phi_h = np.where(phi_h < 0, phi_h + 360, phi_h)

    n = len(phi_d)
    res = np.zeros(n)
    for i in range(n):
        if transmission:
            n_theta_h = btdf.shape[1]
            n_phi_h = btdf.shape[2]
            phi_h_idx = int((phi_h[i] + 180) / 360 * (n_phi_h-1))
            theta_h_idx = int(theta_h[i] / 180 * (n_theta_h-1))
            bsdf_slice = btdf[:, theta_h_idx, phi_h_idx]
        else:
            n_theta_h = brdf.shape[1]
            n_phi_h = brdf.shape[2]
            phi_h_idx = int((phi_h[i]) / 360 * (n_phi_h-1))
            theta_h_idx = int(theta_h[i] / 90 * (n_theta_h-1))
            bsdf_slice = brdf[:, theta_h_idx, phi_h_idx]

        fr = interpolate_slice(bsdf_slice, nodes, mu_i)

        res[i] = fr

    if not transmission:
        jac = 4*np.abs(np.sum(wi[..., np.newaxis]*h, axis=0))
        res /= jac

    # res *= inv_cos

    return res

def convert_helper(mid, in_mu, n, ms, md, brdf, btdf, mirror=False):
    mu, w = mitsuba.core.quad.gauss_lobatto(n)

    fftw_size = 4*md
    if fftw_size % 2 == 0:
        fftw_size += 1
    phi_d = np.linspace(0, 2*np.pi, fftw_size, endpoint=False)

    layer = mitsuba.layer.Layer(mu, w, ms, md)

    coeffs = []
    total = n*n
    current = 0
    for i in range(n):
        for o in range(n):
            idx = i*n+o

            current += 1
            print ("Fourier transform: %.2f%%" % (100*current/total), end="\r")

            mu_i = -layer.nodes[i]
            mu_o = -layer.nodes[o]

            if has_transmission[mid]:
                if mu_i <= 0:
                    coeffs_io = np.zeros((1,1))
                    coeffs.append(np.copy(coeffs_io))
                    continue
            else:
                if mu_i <= 0 or mu_o >= 0:
                    coeffs_io = np.zeros((1,1))
                    coeffs.append(np.copy(coeffs_io))
                    continue

            if mirror and np.abs(mu_i) > np.abs(mu_o):
                if mu_o > 0:
                    mu_i, mu_o = mu_o, mu_i
                else:
                    mu_i, mu_o = -mu_o, -mu_i

            # Eval BRDF
            values = eval_bsdf(brdf, btdf, mu_i, mu_o, phi_d, in_mu)

            # FFTW
            coeffs_io = np.zeros((ms, md))
            coeffs_io[ms//2, :] = np.real(mitsuba.layer.fftw_transform_r2c(values))[fftw_size//2-md//2:fftw_size//2+md//2+1]

            # Save coefficients
            coeffs.append(np.copy(coeffs_io))

    layer.set_fourier_coeffs(coeffs)
    layer.reverse()

    R = layer.reflection_top
    if has_transmission[mid]:
        T = layer.transmission_top_bottom
    else:
        T = scipy.sparse.csc_matrix(R.shape, dtype=np.float64)

    return R, T



n = 120
ms = 1
md = 301

# Regular spacing in thetas, up to 85deg
in_theta = np.linspace(85, 0, 20)
in_mu = np.cos(np.deg2rad(in_theta))
in_mu = np.insert(in_mu, 0, 0.0)

def convert(mid):
    components = ['tb', 'bt'] if has_transmission[mid] else ['-']

    measure_name = names[mid]

    brdfs = []
    btdfs = []
    for c in components:
        brdf, btdf = preprocess(mid, in_mu, c)
        brdfs.append(brdf)
        btdfs.append(btdf)

    for m in ['_mirror', '']:
        for i, c in enumerate(components):
            R, T = convert_helper(mid, in_mu, n, ms, md, brdfs[i], btdfs[i], len(m)>0)

            if c == 'tb':
                scipy.sparse.save_npz("measurements/" + measure_name + "/" + str(mid) + "_Rt" + m, R)
                scipy.sparse.save_npz("measurements/" + measure_name + "/" + str(mid) + "_Ttb" + m, T)
            elif c == 'bt':
                scipy.sparse.save_npz("measurements/" + measure_name + "/" + str(mid) + "_Rb" + m, R)
                scipy.sparse.save_npz("measurements/" + measure_name + "/" + str(mid) + "_Tbt" + m, T)
            else:
                # No transmission, only export top reflection
                scipy.sparse.save_npz("measurements/" + measure_name + "/" + str(mid) + "_Rt" + m, R)

convert(4)
convert(20)
convert(22)
convert(28)
convert(29)