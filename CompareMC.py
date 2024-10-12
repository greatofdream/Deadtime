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
        sim_b, sim_s, sim_meta = ipt['darknoise'][:], ipt['photons'][:], ipt['meta'][:]
        return sim_s.shape[0], sim_b.shape[0], np.sum(sim_s['para']), np.sum(sim_s['unpara']), np.sum(sim_b['para']), np.sum(sim_b['unpara'])
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
    res_para = np.empty((mu_l*dn_l*td_l), dtype=[('mu', np.float64), ('dn', np.float64), ('td', np.float64), ('hit_in', np.int32), ('dn_in', np.int32), ('hit_out', np.int32), ('dn_out', np.int32), ('ratio_hit', np.float64)])
    res_nonpara = np.empty((mu_l*dn_l*td_l), dtype=[('mu', np.float64), ('dn', np.float64), ('td', np.float64), ('hit_in', np.int32), ('dn_in', np.int32), ('hit_out', np.int32), ('dn_out', np.int32), ('ratio_hit', np.float64)])
    i = 0
    for td in args.TD:
        for mu in args.MU:
            for dn in args.DN:
                hit_in, dn_in, hit_out_para, hit_out_nonpara, dn_out_para, dn_out_nonpara = loadH5(args.format.format(td, mu, dn))
                res_para[i] = (float(mu), float(dn), float(td), hit_in, dn_in, hit_out_para, dn_out_para, hit_out_para/hit_in)
                res_nonpara[i] = (float(mu), float(dn), float(td), hit_in, dn_in, hit_out_nonpara, dn_out_nonpara, hit_out_nonpara/hit_in)
                i += 1
    
    with h5py.File(args.opt, 'w') as opt:
        opt.create_dataset('para', data=res_para, compression='gzip')
        opt.create_dataset('nonpara', data=res_nonpara, compression='gzip')
    para_df = pd.DataFrame(res_para)
else:
    with h5py.File(args.opt, 'r') as ipt:
        para_df = pd.DataFrame(ipt['para'][:])
    
with PdfPages(args.opt + '.pdf') as pdf:
    for td, td_rows in para_df.groupby('td'):
        fig, ax = plt.subplots()
        for dn, rows in td_rows.groupby('dn'):
            ax.scatter(rows['mu'], rows['ratio_hit'], s=2, label='{:.1f}kHz'.format(dn))
        ax.legend()
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel('ratio')
        ax.set_ylim([0, 1])
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        pdf.savefig(fig)

    
