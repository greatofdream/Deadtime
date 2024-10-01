import argparse
import numpy as np, h5py
from skpy.DetectorParaReader import BonsaiLikelihoodReader
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

def Unparalyzable(ts):
    ind = np.argsort(ts)
    select = np.zeros(ts.shape, dtype=bool)
    select[ind[0]] = True
    tmp_t = ts[ind[0]]
    for i in range(1, ts.shape[0]):
        if ts[ind[i]] > (tmp_t + T_Dead):
            select[ind[i]] = True
            tmp_t = ts[ind[i]]
    return select

def Paralyzable(ts):
    ind = np.argsort(ts)
    select = np.zeros(ts.shape, dtype=bool)
    select[ind[0]] = True
    tmp_t = ts[ind[0]]
    for i in range(1, ts.shape[0]):
        if ts[ind[i]] > (tmp_t + T_Dead):
            select[ind[i]] = True
            tmp_t = ts[ind[i]]
        else:
            # update the deadtime window if the pulse is in the dead time window
            tmp_t = ts[ind[i]]
    return select

bonsaiLikelihoodReader = BonsaiLikelihoodReader()
bonsaiLikelihoodReader.loadH5()
bonsaiLikelihood = bonsaiLikelihoodReader.pdfs[0, :]
# normalize
bonsaiLikelihood /= np.sum(bonsaiLikelihood) * bonsaiLikelihoodReader.TBIN
R_js = bonsaiLikelihood[::-1]
bonsaiIntegration = np.insert(np.cumsum((R_js[:-1] + R_js[1:])/2), 0, 0) * bonsaiLikelihoodReader.TBIN

bonsaiLikelihood_fun = interp1d((np.arange(-bonsaiLikelihood.shape[0], 0) + bonsaiLikelihoodReader.offsets[1]) * bonsaiLikelihoodReader.TBIN, bonsaiLikelihood[::-1], bounds_error=False, fill_value=0)
bonsaiIntegration_fun = interp1d((np.arange(-bonsaiLikelihood.shape[0], 0) + bonsaiLikelihoodReader.offsets[1]) * bonsaiLikelihoodReader.TBIN, bonsaiIntegration, bounds_error=False, fill_value=(0, 1))
bonsaiIntegration_reverse_fun = interp1d(bonsaiIntegration, (np.arange(-bonsaiLikelihood.shape[0], 0) + bonsaiLikelihoodReader.offsets[1]) * bonsaiLikelihoodReader.TBIN, bounds_error=False, fill_value=(100, 100))

T_max = 900
T_left, T_right = -2*T_max-100, T_max
times = np.arange(T_left, T_right) # due to the light curve has value in [-50,0], padding 100ns in the left
window = T_right - T_left
times_select = times[T_max:] # used for calculating the corrected rate

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output merge pdf')
psr.add_argument('--entries', dest='entries', type=int, default=1000000, help='entries')
psr.add_argument('--parser', dest='parser', default="TD900_MU0.5_DN0", help='Dead time[ns], expected number of photon, Dark noise rate[kHz] values string')
args = psr.parse_args()
values = args.parser.split('_')
v_m = {v[:2]: float(v[2:]) for v in values}

N_sample = args.entries
T_Dead, N_photon, R_b = v_m['TD'], v_m['MU'], v_m['DN']/1E6
# dark noise rate in theory 
R_b_corr = R_b / (1 + R_b*T_Dead) # precise for nonparlyzable, approximation for paralyzable
# signal light curve, t0=0
R = bonsaiLikelihood_fun(times)
# total rate in theory
R_t = R_b + N_photon * R
lambda_t_b = window * R_b
# Number of dark noise and photons
Ns_b, Ns_s = np.random.poisson(lambda_t_b, N_sample), np.random.poisson(N_photon, N_sample)
Ns_b_cum, Ns_s_cum = [0, *np.cumsum(Ns_b)], [0, *np.cumsum(Ns_s)]
# Time of dark noise and photons
Ts_b = np.random.rand(Ns_b_cum[-1]) * window + T_left
Ts_s = bonsaiIntegration_reverse_fun(np.random.rand(Ns_s_cum[-1]))
# MC of the events
sim_meta = np.empty((N_sample,), dtype=[('EventID', np.int64), ('num_b', np.int64), ('num_s', np.int64)])
sim_b = np.empty((Ns_b_cum[-1],), dtype=[('EventID', np.int64), ('T', np.float64), ('unpara', np.bool_), ('para', np.bool_)])
sim_s = np.empty((Ns_s_cum[-1],), dtype=[('EventID', np.int64), ('T', np.float64), ('unpara', np.bool_), ('para', np.bool_)])
sim_b['T'] = Ts_b
sim_s['T'] = Ts_s
sim_b['EventID'] = np.repeat(range(N_sample), Ns_b)
sim_s['EventID'] = np.repeat(range(N_sample), Ns_s)
sim_meta['EventID'] = range(N_sample)
sim_meta['num_b'] = Ns_b
sim_meta['num_s'] = Ns_s
for i in range(N_sample):
    if (Ns_b[i]+Ns_s[i])==0:
        continue
    select = Unparalyzable(np.concatenate([Ts_b[Ns_b_cum[i]:Ns_b_cum[i+1]], Ts_s[Ns_s_cum[i]:Ns_s_cum[i+1]]]))
    sim_b['unpara'][Ns_b_cum[i]:Ns_b_cum[i+1]] = select[:Ns_b[i]]
    sim_s['unpara'][Ns_s_cum[i]:Ns_s_cum[i+1]] = select[Ns_b[i]:]
    select = Paralyzable(np.concatenate([Ts_b[Ns_b_cum[i]:Ns_b_cum[i+1]], Ts_s[Ns_s_cum[i]:Ns_s_cum[i+1]]]))
    sim_b['para'][Ns_b_cum[i]:Ns_b_cum[i+1]] = select[:Ns_b[i]]
    sim_s['para'][Ns_s_cum[i]:Ns_s_cum[i+1]] = select[Ns_b[i]:]

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('darknoise', data=sim_b, compression='gzip')
    opt.create_dataset('photons', data=sim_s, compression='gzip')
    opt.create_dataset('meta', data=sim_meta, compression='gzip')

with PdfPages(args.opt+'.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_select, R_t[T_max:], c='k', label=r'$\mathcal{R}_j$')
    ax.hist([*Ts_b, *Ts_s], range=[-500, 900], bins=1400, weights=np.repeat(1/N_sample, Ns_b_cum[-1]+Ns_s_cum[-1]), histtype='step', color='k', label=r'MC $\mathcal{R}_j$')
    ax.hist([*Ts_b[sim_b['unpara']], *Ts_s[sim_s['unpara']]], range=[-500, 900], bins=1400, weights=np.repeat(1/N_sample, np.sum(sim_b['unpara']) + np.sum(sim_s['unpara'])), histtype='step', color='b', label='MC $\mathcal{R}^m_j$ (nonparalyzable)')
    ax.hist([*Ts_b[sim_b['para']], *Ts_s[sim_s['para']]], range=[-500, 900], bins=1400, weights=np.repeat(1/N_sample, np.sum(sim_b['para']) + np.sum(sim_s['para'])), histtype='step', color='g', label='MC $\mathcal{R}^m_j$ (paralyzable)')
    ax.set_xlabel('residual time[ns]')
    ax.set_ylabel('expect PE number[per ns]')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    fig.tight_layout()
    ax.set_xlim([-200, 500])
    pdf.savefig(fig)
    ax.set_xlim([-50, 100])
    ax.set_ylim([0, 1])
    pdf.savefig(fig)
    ax.set_xlim([-200, 500])
    ax.set_ylim([1E-9, 2])
    ax.set_yscale('log')
    pdf.savefig(fig)

