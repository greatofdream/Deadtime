'''
Reconstructed the mu use the whole dataset
'''
import argparse
import numpy as np, h5py
np.seterr(all='raise')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

from Likelihood import CorrParalyzable, NonCorr
from scipy import optimize

from scipy.interpolate import interp1d
from Generator import Generator

# T_lc_left, T_lc_right = (-bonsaiLikelihood.shape[0] + int(bonsaiLikelihoodReader.offsets[1])) * lc_TBIN, (bonsaiLikelihoodReader.offsets[1] - 1) * lc_TBIN

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input MC')
psr.add_argument('-o', dest='opt', help='output analysis')
psr.add_argument('--parser', dest='parser', default="TD900_MU0.5_DN0", help='Dead time[ns], expected number of photon, Dark noise rate[kHz] values string')
psr.add_argument('--model', dest='model', default='Rectangle', help='light curve shape')
psr.add_argument('--use_truth', dest='use_truth', default=False, action='store_true', help='use truth dark noise in the analysis')
args = psr.parse_args()
generator = Generator(args.model).generator
R_js = generator.R_js
T_lc_left, T_lc_right = generator.t_l, generator.t_r
lc_TBIN = generator.TBIN

values = args.parser.split('_')
v_m = {v[:2]: float(v[2:]) for v in values}
T_D, mu, DN = float(v_m['TD']), float(v_m['MU']), float(v_m['DN'])
sample_window = [-100.0, 500.0]

with h5py.File(args.ipt, 'r') as ipt:
    sim_b = ipt['darknoise'][:]
    sim_s = ipt['photons'][:]
    sim_meta = ipt['meta'][:]
Entries = sim_meta.shape[0]
res_recon = np.empty((1,), dtype=[('mu_truth', np.float64), ('T_D', np.float64), ('DN', np.float64), ('mu_unpara', np.float64), ('mu_para', np.float64), ('mu_unpara_first', np.float64), ('mu_para_first', np.float64)])
Ns_b_cum, Ns_s_cum = [0, *np.cumsum(sim_meta['num_b'])], [0, *np.cumsum(sim_meta['num_s'])]
# using event number in time window [-300:-100]
estimate_b_corr_unpara = np.sum(sim_b['unpara']&(sim_b['T']>-300)&(sim_b['T']<-100)) / Entries / 200
estimate_b_corr_para = np.sum(sim_b['para']&(sim_b['T']>-300)&(sim_b['T']<-100)) / Entries / 200
# 

def recon(paralabel='unpara', estimate_b_corr=estimate_b_corr_unpara, use_truth=True, likelihood='para'):
    b = estimate_b_corr / ( 1 - estimate_b_corr * T_D)
    if use_truth:
        b = DN / 1E6
        estimate_b_corr = b * np.exp(- b * T_D) # paralyzable approximation
    if likelihood == 'para':
        minimizeObj = CorrParalyzable(R_js, T_D, lc_TBIN, b, estimate_b_corr)
    else:
        minimizeObj = NonCorr(R_js, T_D, lc_TBIN, b)
    window_l, window_r = np.repeat(sample_window[0], Entries), np.repeat(sample_window[1], Entries)
    noise_pre_window = T_lc_left - window_l
    window_l[:] = T_lc_left

    hit_index = np.zeros((Entries,), dtype=bool)
    hit_ts = np.zeros((Entries,))
    sim_b_sel  = sim_b[(sim_b[paralabel])&(sim_b['T']>sample_window[0])&(sim_b['T']<sample_window[1])]
    sim_s_sel  = sim_s[(sim_s[paralabel])&(sim_s['T']>sample_window[0])&(sim_s['T']<sample_window[1])]
    assert(np.unique(sim_b_sel['EventID']).shape[0]==sim_b_sel.shape[0])
    assert(np.unique(sim_s_sel['EventID']).shape[0]==sim_s_sel.shape[0])
    hit_index[sim_b_sel['EventID']] = True
    hit_index[sim_s_sel['EventID']] = True
    hit_ts[sim_b_sel['EventID']] = sim_b_sel['T']
    hit_ts[sim_s_sel['EventID']] = sim_s_sel['T']
    window_l[hit_index] = hit_ts[hit_index] - T_D
    window_r[hit_index] = hit_ts[hit_index]
    useless_index = hit_ts[hit_index]<T_lc_left
    noise_pre_window[window_l<T_lc_left] = T_lc_left - hit_ts[hit_index] + T_D
    # noise_pre_window[(window_l<T_lc_left)&(useless_index)] = T_D
    window_l[window_l<T_lc_left] = T_lc_left
    window_r[window_r<T_lc_left] = T_lc_left # the hit before T_lc_left is useless, and included in the useless_index, just set them same, it must be considered in the NoCorr.
    # transfer to the index of light curve
    window_index_l = (window_l - T_lc_left) / lc_TBIN
    window_index_r = (window_r - T_lc_left) / lc_TBIN
    index_l_frac, index_l_int = np.modf(window_index_l)
    index_r_frac, index_r_int = np.modf(window_index_r)
    index_l_int, index_r_int = index_l_int.astype(int), index_r_int.astype(int)
    r_max = int(np.max(index_r_int)) + 2
    cols = np.arange(r_max)
    index_mask = (cols >= index_l_int[:, np.newaxis]) & ((cols-2) < index_r_int[:, np.newaxis])
    integrate_index_mask = (cols >= index_l_int[:, np.newaxis]) & (cols < (index_r_int[:, np.newaxis]-1))
    
    if likelihood == 'para':
        minimizeObj.Rt(useless_index, hit_index, index_l_int, index_l_frac, index_r_int, index_r_frac, r_max, noise_pre_window, integrate_index_mask)
    else:
        # useless_index is not used in this test benchmark
        minimizeObj.Rt(useless_index, hit_index, index_l_int, index_l_frac, index_r_int, index_r_frac, r_max, window_r - window_l + noise_pre_window)

    bounds = [0.01, 30]
    occupancy = np.sum(hit_index) / Entries
    Neff_estimate = occupancy * np.exp(occupancy)
    #print(occupancy, Neff_estimate)
    res_x = optimize.minimize_scalar(minimizeObj.NeffLikelihood_T, Neff_estimate/10, args=(hit_index), options={'maxiter': 500}, bounds=bounds)
    #print(res_x)
    #print(minimizeObj.NeffLikelihood_T(mu, (hit_index)))
    return res_x, minimizeObj
res_unpara, mObj_unpara = recon('unpara', estimate_b_corr_unpara, args.use_truth)
res_para, mObj_para = recon('para', estimate_b_corr_para, args.use_truth)
res_unpara_first, mObj_unpara_first = recon('unpara', estimate_b_corr_unpara, args.use_truth, likelihood='first')
res_para_first, mObj_para_first = recon('para', estimate_b_corr_para, args.use_truth, likelihood='first')

res_recon[0] = (v_m['MU'], T_D, DN, res_unpara.x, res_para.x, res_unpara_first.x, res_para_first.x)
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('res', data=res_recon, compression='gzip')
