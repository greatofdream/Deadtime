'''
Compare the dead time effect
'''
import argparse, numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from matplotlib.patches import Circle
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import statsmodels.api as sm
import ROOT

def loadH5(f):
    print('load {}'.format(f))
    with h5py.File(f, 'r') as ipt:
        sim_b, sim_s, sim_meta = ipt['darknoise'][:], ipt['photons'][:], ipt['meta'][:]
        return sim_s.shape[0], sim_b.shape[0], np.sum(sim_s['unpara'], axis=0), np.sum(sim_s['threshold'], axis=0), np.sum(sim_b['unpara'], axis=0), np.sum(sim_b['threshold'], axis=0), sim_meta.shape[0]

def fit(PE, ratios):
    mod = sm.OLS(np.log(1 - ratios), sm.add_constant(PE))
    res = mod.fit()
    print(res.summary())
    return res

def rootFit(PE, ratios, left=0, right=40):
    # It cannot fit well due the lack of data in the high PE region
    graph = ROOT.TGraph(PE.shape[0], np.array(PE), np.array(ratios))
    # ratios = 1 - b * exp(-PE/k)
    # log(1 - ratios) = log(b) - PE/k
    func = ROOT.TF1("", "1 - [0] * exp(-x / [1])", left, right)
    func.SetParameter(0, 0.3)
    func.SetParameter(1, 4)
    func.SetParLimits(0, 0, 1)
    func.SetParLimits(1, 0, 10)
    graph.Fit(func, 'RM')
    return func.GetParameters(), func.GetParErrors(), graph

psr = argparse.ArgumentParser()
psr.add_argument('--MU', dest="MU", nargs='+', help="MU list")
psr.add_argument('--DN', dest="DN", nargs='+', help="DN list")
psr.add_argument("--TD", dest='TD', nargs='+', help="TD list")
psr.add_argument("--format", dest='format', help="path format")
psr.add_argument("-o", dest='opt', help="output file")
args = psr.parse_args()

mu_l, dn_l, td_l = len(args.MU), len(args.DN), len(args.TD)
res_nonpara = np.empty((mu_l*dn_l*td_l, 3), dtype=[('mu', np.float64), ('dn', np.float64), ('td', np.float64), ('hit_in', np.int32), ('dn_in', np.int32), ('hit_out', np.int32), ('hit_valid', np.int32), ('dn_out', np.int32), ('dn_valid', np.int32), ('entries', np.int32)])
i = 0
for td in args.TD:
    for mu in args.MU:
        for dn in args.DN:
            hit_in, dn_in, hit_out_nonpara, hit_valid_nonpara, dn_out_nonpara, dn_valid_nonpara, entries = loadH5(args.format.format(td, mu, dn))
            res_nonpara[i] = [(float(mu), float(dn), float(td), hit_in, dn_in, hit_out_nonpara[t], hit_valid_nonpara[t], dn_out_nonpara[t], dn_valid_nonpara[t], entries) for t in range(3)]
            i += 1

nonpara_df = pd.DataFrame(res_nonpara[:, 0][['mu', 'dn', 'td']])
nonpara_df['index'] = np.arange(nonpara_df.shape[0])

res = np.empty((3), dtype=[('type', np.int32), ('b', np.float64), ('k', np.float64), ('b_err', np.float64), ('k_err', np.float64)])
for td, td_rows in nonpara_df.groupby('td'):
    for dn, rows in td_rows.groupby('dn'):
        for i, pmttype in enumerate([2, 3, 5]):
            if pmttype!=5:
                fitres = rootFit(rows['mu'], res_nonpara[rows['index'], i]['hit_valid'] / res_nonpara[rows['index'], i]['hit_out'])
                # res[i] = (pmttype, np.exp(fitres.params[0]), - 1/fitres.params[1], 0, 0)
                res[i] = (pmttype, fitres[0][0], fitres[0][1], fitres[1][0], fitres[1][1])
            else:
                res[i] = (pmttype, 0, 10, 0, 0)
print(res)
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('nonpara', data=res_nonpara, compression='gzip')
    opt.create_dataset('res', data=res, compression='gzip')

with PdfPages(args.opt + '.pdf') as pdf:
    for td, td_rows in nonpara_df.groupby('td'):
        for dn, rows in td_rows.groupby('dn'):
            fig, ax = plt.subplots()
            for i, pmttype in enumerate([2, 3, 5]):
                ax.scatter(rows['mu'], res_nonpara[rows['index'], i]['hit_valid'] / res_nonpara[rows['index'], i]['hit_out'], s=2, color=f'C{i}', label=f'Type{pmttype}')
                ax.plot(rows['mu'], 1 - res[i]['b'] * np.exp(-rows['mu'] / res[i]['k']))
            ax.legend()
            ax.set_xlabel(r'$\mu$')
            ax.set_ylabel('threshold ratio')
            ax.set_title('{:.1f}kHz'.format(dn))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            pdf.savefig(fig)
            ax.set_xscale('log')
            pdf.savefig(fig)

    for td, td_rows in nonpara_df.groupby('td'):
        for dn, rows in td_rows.groupby('dn'):
            fig, ax = plt.subplots()
            for i, pmttype in enumerate([2, 3, 5]):
                ax.scatter(rows['mu'], res_nonpara[rows['index'], i]['hit_out'] / res_nonpara[rows['index'], i]['hit_in'], s=10, color=f'C{i}', label=f'Type{pmttype}')
                # print(res_nonpara[rows['index'], i]['hit_out'] / res_nonpara[rows['index'], i]['hit_in'])
                ax.scatter(rows['mu'], res_nonpara[rows['index'], i]['hit_valid'] / res_nonpara[rows['index'], i]['hit_in'], s=10, color=f'C{i}', marker='x', label=f'Type{pmttype} w/ thre')
                # print(res_nonpara[rows['index'], i]['hit_valid'] / res_nonpara[rows['index'], i]['hit_in'])
            ax.legend()
            ax.set_xlabel(r'$\mu$')
            ax.set_ylabel('ratio')
            ax.set_title('{:.1f}kHz'.format(dn))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            pdf.savefig(fig)
            ax.set_xscale('log')
            pdf.savefig(fig)

    for td, td_rows in nonpara_df.groupby('td'):
        for dn, rows in td_rows.groupby('dn'):
            fig, ax = plt.subplots()
            for i, pmttype in enumerate([2, 3, 5]):
                ax.scatter(rows['mu'], res_nonpara[rows['index'], i]['hit_out'] / res_nonpara[rows['index'], i]['entries'], s=10, color=f'C{i}', label=f'Type{pmttype}')
                # print(res_nonpara[rows['index'], i]['hit_out'] / res_nonpara[rows['index'], i]['hit_in'])
                ax.scatter(rows['mu'], res_nonpara[rows['index'], i]['hit_valid'] / res_nonpara[rows['index'], i]['entries'], s=10, color=f'C{i}', marker='x', label=f'Type{pmttype} w/ thre')
                # print(res_nonpara[rows['index'], i]['hit_valid'] / res_nonpara[rows['index'], i]['hit_in'])
            ax.legend()
            ax.set_xlabel(r'$\mu$')
            ax.set_ylabel('ElecPE rate')
            ax.set_title('{:.1f}kHz'.format(dn))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            pdf.savefig(fig)
            ax.set_xscale('log')
            pdf.savefig(fig)


