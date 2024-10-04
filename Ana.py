'''
Reconstructed the mu
'''
import argparse
import numpy as np, h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

from skpy.DetectorParaReader import BonsaiLikelihoodReader
from scipy.interpolate import interp1d
bonsaiLikelihoodReader = BonsaiLikelihoodReader()
bonsaiLikelihoodReader.loadH5()
bonsaiLikelihood = bonsaiLikelihoodReader.pdfs[0, :]
bonsaiIntegration = np.cumsum(bonsaiLikelihood[::-1])
# normalize
bonsaiLikelihood /= np.sum(bonsaiLikelihood) * bonsaiLikelihoodReader.TBIN
bonsaiIntegration /= bonsaiIntegration[-1]
bonsaiLikelihood_fun = interp1d((np.arange(-bonsaiLikelihood.shape[0], 0) + bonsaiLikelihoodReader.offsets[1]) * bonsaiLikelihoodReader.TBIN, bonsaiLikelihood[::-1], bounds_error=False, fill_value=0)
bonsaiIntegration_fun = interp1d((np.arange(-bonsaiLikelihood.shape[0], 0) + bonsaiLikelihoodReader.offsets[1]) * bonsaiLikelihoodReader.TBIN, bonsaiIntegration, bounds_error=False, fill_value=(0, 1))

def Recon(Ts, b_corr, T_D):
    if len(Ts) == 0:
        return 0
    else:
        b = b_corr / ( 1 - b_corr * T_D)
        R_t = bonsaiLikelihood_fun(Ts)
        if R_t[0] < 1e-5:
            return 0
        lambda_t = bonsaiIntegration_fun(Ts)
        return (1 / lambda_t[0] - b) / R_t[0]
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input MC')
psr.add_argument('-o', dest='opt', help='output analysis')
psr.add_argument('--parser', dest='parser', default="TD900_MU0.5_DN0", help='Dead time[ns], expected number of photon, Dark noise rate[kHz] values string')
args = psr.parse_args()
values = args.parser.split('_')
v_m = {v[:2]: float(v[2:]) for v in values}
T_D, mu = float(v_m['TD']), float(v_m['MU'])
sample_window = [-100, 300]

with h5py.File(args.ipt, 'r') as ipt:
    sim_b = ipt['darknoise'][:]
    sim_s = ipt['photons'][:]
    sim_meta = ipt['meta'][:]
Entries = sim_meta.shape[0]
recon = np.empty((Entries,), dtype=[('EventID', np.int64), ('mu_truth', np.float64), ('mu_unpara', np.float64), ('mu_para', np.float64), ('mu_unpara_direct', np.float64), ('mu_para_direct', np.float64), ('b_unpara_direct', np.float64), ('b_para_direct', np.float64)])
recon['EventID'] = sim_meta['EventID']
recon['mu_truth'] = v_m['MU']
Ns_b_cum, Ns_s_cum = [0, *np.cumsum(sim_meta['num_b'])], [0, *np.cumsum(sim_meta['num_s'])]
# using event number in time window [-300:-100]
estimate_b_corr_unpara = np.sum(sim_b['unpara']&(sim_b['T']>-300)&(sim_b['T']<-100)) / Entries / 200
estimate_b_corr_para = np.sum(sim_b['para']&(sim_b['T']>-300)&(sim_b['T']<-100)) / Entries / 200
# 
for i in range(Entries):
    # unparalyzable
    Ts_b = sim_b[Ns_b_cum[i]:Ns_b_cum[i+1]]['T'][sim_b[Ns_b_cum[i]:Ns_b_cum[i+1]]['unpara']]
    Ts_s = sim_s[Ns_s_cum[i]:Ns_s_cum[i+1]]['T'][sim_s[Ns_s_cum[i]:Ns_s_cum[i+1]]['unpara']]
    Ts_b  = Ts_b[Ts_b>sample_window[0]]
    Ts_s  = Ts_s[Ts_s>sample_window[0]]
    mu_unpara = Recon(np.concatenate([Ts_b, Ts_s]), estimate_b_corr_unpara, T_D)
    mu_unpara_direct = Ts_s.shape[0] + Ts_b.shape[0]
    b_unpara_direct = estimate_b_corr_unpara
    # paralyzable
    Ts_b = sim_b[Ns_b_cum[i]:Ns_b_cum[i+1]]['T'][sim_b[Ns_b_cum[i]:Ns_b_cum[i+1]]['para']]
    Ts_s = sim_s[Ns_s_cum[i]:Ns_s_cum[i+1]]['T'][sim_s[Ns_s_cum[i]:Ns_s_cum[i+1]]['para']]
    Ts_b  = Ts_b[Ts_b>sample_window[0]]
    Ts_s  = Ts_s[Ts_s>sample_window[0]]
    mu_para = Recon(np.concatenate([Ts_b, Ts_s]), estimate_b_corr_para, T_D)
    mu_para_direct = Ts_s.shape[0] + Ts_b.shape[0] 
    b_para_direct = estimate_b_corr_para

    recon[i] = (i, v_m['MU'], mu_unpara, mu_para, mu_unpara_direct, mu_para_direct, b_unpara_direct, b_para_direct)
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('recon', data=recon, compression='gzip')

with PdfPages(args.opt+'.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(recon['mu_unpara'], bins=100, histtype='step', label='recon')
    ax.set_xlabel('mu')
    ax.axvline(mu, ls='--', label='truth')
    ax.set_ylabel('entries')
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(recon['mu_para'], bins=100, histtype='step', label='recon')
    ax.set_xlabel('mu')
    ax.axvline(mu, ls='--', label='truth')
    ax.set_ylabel('entries')
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist([*sim_b['T'], *sim_s['T']], range=sample_window, bins=sample_window[1]-sample_window[0], histtype='step', color='k', label=r'MC $\mathcal{R}_j$')
    ax.hist([*sim_b['T'][sim_b['unpara']], *sim_s['T'][sim_s['unpara']]], range=sample_window, bins=sample_window[1]-sample_window[0], histtype='step', color='b', label='MC $\mathcal{R}^m_j$ (nonparalyzable)')
    ax.hist([*sim_b['T'][sim_b['para']], *sim_s['T'][sim_s['para']]], range=sample_window, bins=sample_window[1]-sample_window[0], histtype='step', color='g', label='MC $\mathcal{R}^m_j$ (paralyzable)')
    ax.set_xlabel('residual time[ns]')
    ax.set_ylabel('expect PE number[per ns]')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    fig.tight_layout()
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)

