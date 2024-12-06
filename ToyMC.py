import argparse
import numpy as np, h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from Generator import Generator
from Spe import SpeSampler, Elec

def Unparalyzable(sim_data):
    # sim_data has same T
    ind = np.argsort(sim_data[:, 0]['T'])
    select = np.zeros(sim_data.shape, dtype=[('unpara', np.bool_), ('unparaQ', np.float64)])
    select[ind[0], :]['unpara'] = True
    select[ind[0], :]['unparaQ'] += sim_data[ind[0], :]['Q']
    gate_l = sim_data[ind[0], 0]['T']
    gate_i = 0
    for i in range(1, sim_data.shape[0]):
        if sim_data[ind[i], 0]['T'] < (gate_l + T_Integrate):
            efficiency = elec.GetEfficiency(sim_data[ind[i], 0]['T'] - gate_l)
            select[ind[gate_i], :]['unparaQ'] += efficiency * sim_data[ind[i], :]['Q']
        elif sim_data[ind[i], 0]['T'] > (gate_l + T_Dead):
            select[ind[i], :]['unpara'] = True
            select[ind[i], :]['unparaQ'] += sim_data[ind[i], :]['Q']
            gate_l = sim_data[ind[i], 0]['T']
            gate_i = i
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

T_max = 900
T_left, T_right = -2*T_max-100, T_max
times = np.arange(T_left, T_right) # due to the light curve has value in [-50,0], padding 100ns in the left
window = T_right - T_left
times_select = times[T_max:] # used for calculating the corrected rate

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output merge pdf')
psr.add_argument('--entries', dest='entries', type=int, default=100000, help='entries')
psr.add_argument('--model', dest='model', default='Rectangle', help='light curve shape')
psr.add_argument('--parser', dest='parser', default="TD900_MU0.5_DN0", help='Dead time[ns], expected number of photon, Dark noise rate[kHz] values string')
args = psr.parse_args()
generator = Generator(args.model).generator
speSampler = SpeSampler()
elec = Elec()
values = args.parser.split('_')
v_m = {v[:2]: float(v[2:]) for v in values}

N_sample = args.entries
T_Dead, N_photon, R_b = v_m['TD'], v_m['MU'], v_m['DN']/1E6
T_Integrate = 400
# dark noise rate in theory 
R_b_corr = R_b / (1 + R_b*T_Dead) # precise for nonparlyzable, approximation for paralyzable
# signal light curve, t0=0
R = generator.fun(times)
# total rate in theory
R_t = R_b + N_photon * R
lambda_t_b = window * R_b

# Number of dark noise and photons
Ns_b, Ns_s = np.random.poisson(lambda_t_b, N_sample), np.random.poisson(N_photon, N_sample)
Ns_b_cum, Ns_s_cum = [0, *np.cumsum(Ns_b)], [0, *np.cumsum(Ns_s)]

# storage of the res
sim_meta = np.empty((N_sample,), dtype=[('EventID', np.int64), ('num_b', np.int64), ('num_s', np.int64)])
sim_b = np.empty((Ns_b_cum[-1], 3), dtype=[('EventID', np.int64), ('T', np.float64), ('Q', np.float64), ('unpara', np.bool_), ('unparaQ', np.float64), ('threshold', np.bool_), ('unparaCorrQ', np.float64)])
sim_s = np.empty((Ns_s_cum[-1], 3), dtype=[('EventID', np.int64), ('T', np.float64), ('Q', np.float64), ('unpara', np.bool_), ('unparaQ', np.float64), ('threshold', np.bool_), ('unparaCorrQ', np.float64)])

# Time of dark noise and photons
Ts_b = np.random.rand(Ns_b_cum[-1]) * window + T_left
Ts_s = generator.integration_reverse_fun(np.random.rand(Ns_s_cum[-1]))
# MC of the events
c_rand = np.random.rand(3, Ns_b_cum[-1])
sim_b['Q'][:, 0] = speSampler.sample(c_rand[0], 2)
sim_b['Q'][:, 1] = speSampler.sample(c_rand[1], 3)
sim_b['Q'][:, 2] = speSampler.sample(c_rand[2], 5)
c_rand = np.random.rand(3, Ns_s_cum[-1])
sim_s['Q'][:, 0] = speSampler.sample(c_rand[0], 2)
sim_s['Q'][:, 1] = speSampler.sample(c_rand[0], 3)
sim_s['Q'][:, 2] = speSampler.sample(c_rand[0], 5)
sim_b['T'] = Ts_b[:, np.newaxis]
sim_s['T'] = Ts_s[:, np.newaxis]
sim_b['EventID'] = np.repeat(range(N_sample), Ns_b)[:, np.newaxis]
sim_s['EventID'] = np.repeat(range(N_sample), Ns_s)[:, np.newaxis]
sim_meta['EventID'] = range(N_sample)
sim_meta['num_b'] = Ns_b
sim_meta['num_s'] = Ns_s

for i in range(N_sample):
    if (Ns_b[i]+Ns_s[i])==0:
        continue
    select = Unparalyzable(np.concatenate([sim_b[Ns_b_cum[i]:Ns_b_cum[i+1]], sim_s[Ns_s_cum[i]:Ns_s_cum[i+1]]]))
    sim_b[['unpara', 'unparaQ']][Ns_b_cum[i]:Ns_b_cum[i+1]] = select[:Ns_b[i]]
    sim_s[['unpara', 'unparaQ']][Ns_s_cum[i]:Ns_s_cum[i+1]] = select[Ns_b[i]:]

for i, pmttype in enumerate([2, 3, 5]):
    if sim_b.shape[0] > 0:
        valid, corrQ = elec.IsElecHit(sim_b[sim_b['unpara'][:, i], i]['unparaQ'], pmttype)
        sim_b['threshold'][sim_b['unpara'][:, i], i] = valid
        sim_b['unparaCorrQ'][sim_b['unpara'][:, i], i] = corrQ

    valid, corrQ = elec.IsElecHit(sim_s[sim_s['unpara'][:, i], i]['unparaQ'], pmttype)
    print('ElecPE_q/ElecPE: {:.2f}, ElecPE/PE: {:.2f}'.format(np.sum(valid) / np.sum(sim_s['unpara'][:, i]), np.sum(sim_s['unpara'][:, i]) / sim_s.shape[0]))
    sim_s['threshold'][sim_s['unpara'][:, i], i] = valid
    sim_s['unparaCorrQ'][sim_s['unpara'][:, i], i] = corrQ

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('darknoise', data=sim_b, compression='gzip')
    opt.create_dataset('photons', data=sim_s, compression='gzip')
    opt.create_dataset('meta', data=sim_meta, compression='gzip')

with PdfPages(args.opt+'.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_select, R_t[T_max:], c='k', label=r'$\mathcal{R}_j$')
    ax.hist([*Ts_b, *Ts_s], range=[-500, 900], bins=1400, weights=np.repeat(1/N_sample, Ns_b_cum[-1]+Ns_s_cum[-1]), histtype='step', color='k', label=r'MC $\mathcal{R}_j$')
    ax.hist([*Ts_b[sim_b['unpara'][:, 0]], *Ts_s[sim_s['unpara'][:, 0]]], range=[-500, 900], bins=1400, weights=np.repeat(1/N_sample, np.sum(sim_b['unpara'][:, 0]) + np.sum(sim_s['unpara'][:, 0])), histtype='step', color='b', label='MC $\mathcal{R}^m_j$ (nonparalyzable)')
    ax.set_xlabel('time[ns]')
    ax.set_ylabel('expect number[per ns]')
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

    fig, ax = plt.subplots()
    for i, pmttype in enumerate([2, 3, 5]):
        ax.hist(sim_s['unparaCorrQ'][:, i], bins=100, range=[0, 10], histtype='step', label=f'Type{pmttype}')
    ax.set_xlabel('PE')
    ax.set_ylabel('Entries')
    ax.set_yscale('log')
    ax.legend()
    pdf.savefig(fig)
