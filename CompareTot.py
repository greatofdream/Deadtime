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
def loadH5(f):
    print('load {}'.format(f))
    with h5py.File(f, 'r') as ipt:
        return ipt['res'][:]
psr = argparse.ArgumentParser()
psr.add_argument('--MU', dest="MU", nargs='+', help="MU list")
psr.add_argument('--DN', dest="DN", nargs='+', help="DN list")
psr.add_argument("--TD", dest='TD', nargs='+', help="TD list")
psr.add_argument("--format", dest='format', help="path format")
psr.add_argument("-o", dest='opt', help="output file")
psr.add_argument('--plot', action='store_true', default=False, help='just plot')
args = psr.parse_args()
if not args.plot:
    mu_l, dn_l, td_l = len(args.MU), len(args.DN), len(args.TD)
    res = np.concatenate([loadH5(args.format.format(td, mu, dn)) for td in args.TD for mu in args.MU for dn in args.DN])
    
    with h5py.File(args.opt, 'w') as opt:
        opt.create_dataset('res', data=res, compression='gzip')
    res_df = pd.DataFrame(res)
else:
    with h5py.File(args.opt, 'r') as ipt:
        res_df = pd.DataFrame(ipt['res'][:])

colors = ['r', 'g', 'b', 'k']
with PdfPages(args.opt + '.pdf') as pdf:
    fig_unpara, ax_unpara = plt.subplots()
    fig_para, ax_para = plt.subplots()
    for (dn, rows), c in zip(res_df.groupby('DN'), colors):
        ax_unpara.scatter(rows['mu_truth'], rows['mu_unpara']/rows['mu_truth']-1, s=10, c=c, label='$L^{para}$'+' DN{}kHz'.format(dn))
        ax_unpara.scatter(rows['mu_truth'], rows['mu_unpara_first']/rows['mu_truth']-1, s=10, c=c, marker='x', label='$L^{first}$'+' DN{}kHz'.format(dn))
        ax_para.scatter(rows['mu_truth'], rows['mu_para']/rows['mu_truth']-1, s=10, c=c, label='$L^{para}$'+' DN{}kHz'.format(dn))
        ax_para.scatter(rows['mu_truth'], rows['mu_para_first']/rows['mu_truth']-1, s=10, c=c, marker='x', label='$L^{first}$'+' DN{}kHz'.format(dn))
    [(ax.set_ylabel(r'$\Delta \mu/\mu$'),
        ax.xaxis.set_minor_locator(MultipleLocator(1)),
        ax.yaxis.set_minor_locator(MultipleLocator(0.01)),
        ax.set_xlabel(r'$\mu$'),
        ax.set_xlim([0, 41]),
        ax.legend()
      ) for ax in [ax_unpara, ax_para]]
    pdf.savefig(fig_unpara)
    pdf.savefig(fig_para)

