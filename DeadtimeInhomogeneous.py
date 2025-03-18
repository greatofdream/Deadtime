'''
Theory calculation of the deadtime with Bondai lightcurve and 900ns deadtime
'''
import argparse
from skpy.DetectorParaReader import BonsaiLikelihoodReader
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
from Generator import Unparalyzable
T_Dead = 900
bonsaiLikelihoodReader = BonsaiLikelihoodReader()
bonsaiLikelihoodReader.loadH5()
bonsaiLikelihood = bonsaiLikelihoodReader.pdfs[0, :]
bonsaiIntegration = np.cumsum(bonsaiLikelihood[::-1])
# normalize
bonsaiLikelihood /= np.sum(bonsaiLikelihood) * bonsaiLikelihoodReader.TBIN
bonsaiIntegration /= bonsaiIntegration[-1]
bonsaiLikelihood_fun = interp1d((np.arange(-bonsaiLikelihood.shape[0], 0) + bonsaiLikelihoodReader.offsets[1]) * bonsaiLikelihoodReader.TBIN, bonsaiLikelihood[::-1], bounds_error=False, fill_value=0)
bonsaiIntegration_fun = interp1d((np.arange(-bonsaiLikelihood.shape[0], 0) + bonsaiLikelihoodReader.offsets[1]) * bonsaiLikelihoodReader.TBIN, bonsaiIntegration, bounds_error=False, fill_value=(0, 1))
bonsaiIntegration_reverse_fun = interp1d(bonsaiIntegration, (np.arange(-bonsaiLikelihood.shape[0], 0) + bonsaiLikelihoodReader.offsets[1]) * bonsaiLikelihoodReader.TBIN, bounds_error=False, fill_value=(100, 100))

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input MC hdf5')
psr.add_argument('-o', dest="opt", help="output file")
psr.add_argument('--n', dest="n", type=float, help="expected photons")
psr.add_argument('--dark', dest="dark", type=int, default= 10, help="expected dark rate kHz")
args = psr.parse_args()

N_photon = args.n
R_b = args.dark/1E6 # 1E-5#10kHz
times = np.arange(-1900, 800)
times_select = times[900:]
R_b_corr = R_b / (1 + R_b*T_Dead)
print(R_b, R_b_corr)
R = bonsaiLikelihood_fun(times)
R_t = R_b + N_photon * R
R_t_cumsum = np.insert(np.cumsum((R_t[1:] + R_t[:-1])/2), 0, 0)
R_cumsum = np.cumsum(R[900:])
# paralyzable
R_corr_paralyzable = np.exp(- (R_t_cumsum[900:] - R_t_cumsum[:-900]))
R_m_t_paralyzable = R_corr_paralyzable * R_t[900:]
## corrected paralyzable to unparalyzable using iteration: failed, not match the MC
'''
R_factor_cum = np.cumsum(R_t[-900:])
R_m_t_unparalyzable = R_t[-900:] * np.exp(-R_factor_cum)
R_m_t_unparalyzable[-700:] *= 10
for i in range(10):
    R_m_t_unparalyzable_cum = np.cumsum(R_m_t_unparalyzable)
    factor = 1 - R_b_corr * np.arange(900,0,-1) / (1 - R_m_t_unparalyzable_cum)
    R_factor = R_t[-900:] * factor
    R_factor_cum = np.cumsum(R_factor)
    R_m_t_unparalyzable =  R_factor * np.exp(-R_factor_cum)
'''
## corrected paralyzable to unparalyzable using iteration: failed, not match the MC; when iteration number is larger than 1, the probability is negative
'''
R_m_t_unparalyzable = np.zeros(1800)
R_m_t_unparalyzable[:] = R_m_t_paralyzable
for i in range(1):
    R_m_t_unparalyzable_cum = np.cumsum(R_m_t_unparalyzable)
    R_m_t_unparalyzable[-900:] = R_t[-900:] * (1-(R_m_t_unparalyzable_cum[900:] -R_m_t_unparalyzable_cum[:900]))
'''
## unparalyzable calculation one by one: very close, but not same
'''
R_m_t_unparalyzable = np.zeros(1900)
R_m_t_unparalyzable[:900] = R_b_corr
for i in range(1000):
    R_m_t_unparalyzable[900+i] = R_t[-1000+i] * (1 - np.sum(R_m_t_unparalyzable[i:(i+900)])) / (1 + R_t[-1000+i])
'''

## unparalyzable calculation using approximation: failed, not match MC
'''
R_factor = R_t[-900:] * (1-R_b_corr * np.arange(900,0,-1))
R_factor_cum = np.cumsum(R_factor)
R_m_t_unparalyzable =  R_factor * np.exp(-R_factor_cum)
'''
# unparalyzable poisson approximation
'''
R_cumsum_conv = np.zeros(R_cumsum.shape)
for i in range(1, R_cumsum_conv.shape[0]):
    R_cumsum_conv[i] = R_cumsum_conv[i-1] + R_cumsum[i]
Denominator = R_cumsum - R_cumsum_conv * R_b_corr
R_m_unparalyzable = R_t[900:]/(N_photon * Denominator + 1 + R_b_corr * T_Dead)
'''
# unparalyzable poission
# R_m_unparalyzable = R_t[900:] * (1 - R_b_corr * T_Dead) * np.exp() + R_b_corr
## unparalyzable calculation
b_factor = 1 - R_b_corr * np.arange(900,0,-1)
'''
R_t_cumsum = np.cumsum(R_t[-900:])
R_t_cumsum_matrix = R_t_cumsum[:,np.newaxis] - R_t_cumsum
integration_Rt = np.zeros(900)
for i in range(900):
    integration_Rt[i] = np.sum(R_t[-900:(-900+i)] * b_factor[:i] * np.exp(-R_t_cumsum_matrix[i,:i]))# + np.sum(R_t[-899:(-900+i+1)] * b_factor[1:(i+1)] * np.exp(-R_t_cumsum_matrix[i,1:(i+1)]))) / 2
## integration_Rt = np.cumsum(R_t[-900:] * b_factor[:i] * np.exp(R_t_cumsum)) * np.exp(-R_t_cumsum)
'''
# it seems that the error is large after the peak
R_t_cumsum = np.insert(np.cumsum((R_t[-900:-1] + R_t[-899:])/2), 0, 0)
'''
R_t_corr = R_t[-900:] * np.arange(900, 0, -1) * np.exp(R_t_cumsum)
R_t_corr_cumsum = np.insert(np.cumsum(R_t_corr[-900:-1] + R_t_corr[-899:])/2, 0, 0)
integration_Rt = np.exp(-R_t_cumsum) *  R_t_corr_cumsum * R_b_corr 
'''
R_t_corr = np.exp(R_t_cumsum)
R_t_corr_cumsum = np.insert(np.cumsum((R_t_corr[-900:-1] + R_t_corr[-899:]) / 2), 0, 0)

integration_Rt_test = 1 - np.exp(-np.cumsum((R_t[-900:-1] + R_t[-899:])/2))
R_m_t_unparalyzable = R_t[-900:] * (1 - R_b_corr * 900 + R_b_corr*R_t_corr_cumsum) * np.exp(-R_t_cumsum)
# load MC
'''
with h5py.File(args.ipt, 'r') as ipt:
    sim_b = ipt['darknoise'][:]
    sim_s = ipt['photons'][:]
    sim_meta = ipt['meta'][:]
'''
N_sample = 1000000
window = 2700
lambda_t_b = window * R_b
Ns_b, Ns_s = np.random.poisson(lambda_t_b, N_sample), np.random.poisson(N_photon, N_sample)
Ns_b_cum, Ns_s_cum = [0, *np.cumsum(Ns_b)], [0, *np.cumsum(Ns_s)]
Ts_b = np.random.rand(Ns_b_cum[-1]) * window - 1800
Ts_s = bonsaiIntegration_reverse_fun(np.random.rand(Ns_s_cum[-1]))
select_array_b = np.zeros(Ns_b_cum[-1], dtype=bool)
select_array_s = np.zeros(Ns_s_cum[-1], dtype=bool)
for i in range(N_sample):
    # assert(Ns_b[i]+Ns_s[i]>0, f"{Ns_b[i]}, {Ns_s[i]}")
    if (Ns_b[i]+Ns_s[i])==0:
        continue
    select = Unparalyzable(np.concatenate([Ts_b[Ns_b_cum[i]:Ns_b_cum[i+1]], Ts_s[Ns_s_cum[i]:Ns_s_cum[i+1]]]), T_Dead)
    select_array_b[Ns_b_cum[i]:Ns_b_cum[i+1]] = select[:Ns_b[i]]
    select_array_s[Ns_s_cum[i]:Ns_s_cum[i+1]] = select[Ns_b[i]:]
# Theory integration
R_cumsum = np.insert(np.cumsum((R[-900:-1] + R[-899:])/2), 0, 0)
exp_R_cumsum = np.exp(-N_photon * R_cumsum)
R_cumsum_cumsum = np.insert(np.cumsum((exp_R_cumsum[-900:-1] + exp_R_cumsum[-899:])/2), 0, 0)
F_R_m_t_paralyzable = np.exp(-T_Dead*R_b) * (1 - exp_R_cumsum + R_b * R_cumsum_cumsum)
with PdfPages(args.opt) as pdf:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_select[700:], R_t[900:][700:], c='k', label=r'$\mathcal{R}_j$')
    ax.plot(times_select[700:], R_m_t_paralyzable[700:], c='orange', ls='dotted', label=r'$\mathcal{R}^m_j$ (paralyzable)')
    #ax.plot(times_select[700:], R_m_unparalyzable[700:], c='b', label=r'$\mathcal{R}^m_j$ (unparalyzable approximation)')
    ax.plot(times_select[-900:], R_m_t_unparalyzable[-900:], c='b', ls='dotted', label=r'$\mathcal{R}^m_j$ (nonparalyzable)')
    # ax.plot(times_select[-900:], integration_Rt[-900:], c='g', label=r'$F\mathcal{R}^m_j$ (unparalyzable corrected)')
    # ax.plot(times_select[-900:-1], integration_Rt_test[:], c='g', ls='--')
    ax.hist([*Ts_b, *Ts_s], range=[-500, 900], bins=1400, weights=np.repeat(1/N_sample, Ns_b_cum[-1]+Ns_s_cum[-1]), histtype='step', color='k', label=r'MC $\mathcal{R}_j$')
    ax.hist([*Ts_b[select_array_b], *Ts_s[select_array_s]], range=[-500, 900], bins=1400, weights=np.repeat(1/N_sample, np.sum(select_array_b)+np.sum(select_array_s)), histtype='step', color='b', label='MC $\mathcal{R}^m_j$ (nonparalyzable)')
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
    ax.set_yscale('log')
    ax.set_xlim([-200, 500])
    ax.set_ylim([np.max([np.min(R_m_t_paralyzable[700:])/2, 1E-10]), N_photon/2])
    pdf.savefig(fig)
    ax2 = ax.twinx()
    ax2.plot(times_select[700:], R_m_t_paralyzable[700:]/R_t[900:][700:], ls='--', c='orange')
    # ax2.plot(times_select[700:], R_m_unparalyzable[700:]/R_t[900:][700:], ls='--', c='b')
    ax2.set_ylabel('Correction ratio')
    fig.tight_layout()
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_select[700:], R_m_t_paralyzable[700:], c='orange', ls='dotted', label=r'$\mathcal{R}^m_j$ (paralyzable)')
    ax.plot(times_select[-900:], R_m_t_unparalyzable[-900:], c='b', ls='dotted', label=r'$\mathcal{R}^m_j$ (nonparalyzable)')
    ax2 = ax.twinx()
    ax2.plot(times_select[-900:], np.insert(np.cumsum((R_m_t_paralyzable[-900:-1] + R_m_t_paralyzable[-899:])/2), 0, 0), c='orange', ls='dotted', label=r'$\mathcal{R}^m_j$ (paralyzable)')
    ax2.plot(times_select[-900:], np.insert(np.cumsum((R_m_t_unparalyzable[-900:-1] + R_m_t_unparalyzable[-899:])/2), 0, 0), c='b', ls='dotted', label=r'$\mathcal{R}^m_j$ (nonparalyzable)')
    ax2.plot(times_select[-900:], F_R_m_t_paralyzable)
    ax.set_xlabel('residual time[ns]')
    ax.set_ylabel('expect PE number[per ns]')
    ax.set_yscale('log')
    ax2.set_ylabel('Cummulative summation')
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    fig.tight_layout()
    pdf.savefig(fig)

