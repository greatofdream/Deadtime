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
        return sim_s.shape[0], sim_b.shape[0], np.sum(sim_s['para']), np.sum(sim_s['unpara']), np.sum(sim_b['para']), np.sum(sim_b['unpara']), sim_meta.shape[0]
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
    res_para = np.empty((mu_l*dn_l*td_l), dtype=[('mu', np.float64), ('dn', np.float64), ('td', np.float64), ('hit_in', np.int32), ('dn_in', np.int32), ('hit_out', np.int32), ('dn_out', np.int32), ('ratio_hit', np.float64), ('entries', np.int32)])
    res_nonpara = np.empty((mu_l*dn_l*td_l), dtype=[('mu', np.float64), ('dn', np.float64), ('td', np.float64), ('hit_in', np.int32), ('dn_in', np.int32), ('hit_out', np.int32), ('dn_out', np.int32), ('ratio_hit', np.float64), ('entries', np.int32)])
    i = 0
    for td in args.TD:
        for mu in args.MU:
            for dn in args.DN:
                hit_in, dn_in, hit_out_para, hit_out_nonpara, dn_out_para, dn_out_nonpara, entries = loadH5(args.format.format(td, mu, dn))
                res_para[i] = (float(mu), float(dn), float(td), hit_in, dn_in, hit_out_para, dn_out_para, hit_out_para/hit_in, entries)
                res_nonpara[i] = (float(mu), float(dn), float(td), hit_in, dn_in, hit_out_nonpara, dn_out_nonpara, hit_out_nonpara/hit_in, entries)
                i += 1
    
    with h5py.File(args.opt, 'w') as opt:
        opt.create_dataset('para', data=res_para, compression='gzip')
        opt.create_dataset('nonpara', data=res_nonpara, compression='gzip')
    para_df = pd.DataFrame(res_para)
    nonpara_df = pd.DataFrame(res_nonpara)
else:
    with h5py.File(args.opt, 'r') as ipt:
        para_df = pd.DataFrame(ipt['para'][:])
        nonpara_df = pd.DataFrame(ipt['nonpara'][:])
colors = ['r', 'g', 'b', 'k']
with PdfPages(args.opt + '.pdf') as pdf:
    nonpara_df_gr = nonpara_df.groupby('td')
    for td, td_rows in para_df.groupby('td'):
        fig, ax = plt.subplots()
        fig_c, ax_c = plt.subplots()
        fig_non, ax_non = plt.subplots()
        fig_non_c, ax_non_c = plt.subplots()
        fig_diff, ax_diff = plt.subplots()
        fig_diff_c, ax_diff_c = plt.subplots()

        non_td_rows = nonpara_df_gr.get_group(td)
        non_td_rows_gr = non_td_rows.groupby('dn')
        for (dn, rows), c in zip(td_rows.groupby('dn'), colors):
            ax.scatter(rows['mu'], rows['ratio_hit'], s=4, color=c, marker='+', label='Paralyzable {:.1f}kHz'.format(dn))
            ax_c.scatter(rows['mu'], rows['hit_out'] / rows['entries'], s=4, color=c, marker='+', label='Paralyzable {:.1f}kHz'.format(dn))
            non_rows = non_td_rows_gr.get_group(dn)
            ax_non.scatter(non_rows['mu'], non_rows['ratio_hit'], s=4, color=c, marker='x', label='Nonparalyzable {:.1f}kHz'.format(dn))
            ax_non_c.scatter(non_rows['mu'], non_rows['hit_out'] / non_rows['entries'], s=4, color=c, marker='x', label='Nonparalyzable {:.1f}kHz'.format(dn))
            assert((rows['mu']==non_rows['mu']).all())
            ax_diff.scatter(rows['mu'], rows['ratio_hit'] - non_rows['ratio_hit'], s=4, color=c, label='{:.1f}kHz'.format(dn))
            ax_diff_c.scatter(rows['mu'], rows['hit_out'] / rows['entries'] - non_rows['hit_out'] / non_rows['entries'], s=4, color=c, label='{:.1f}kHz'.format(dn))
        [(ax_i.legend(),
        ax_i.set_xlabel(r'$\mu$'),
        ax_i.set_ylabel('ratio'),
        ax_i.set_ylim([0, 1]),
        ax_i.yaxis.set_major_locator(MultipleLocator(0.1)),
        ax_i.xaxis.set_major_locator(MultipleLocator(5)),
        ax_i.xaxis.set_minor_locator(MultipleLocator(0.2)),
        pdf.savefig(fig_i)) for fig_i, ax_i in zip([fig, fig_non], [ax, ax_non])]

        [(ax_i.legend(),
        ax_i.set_xlabel(r'$\mu$'),
        ax_i.set_ylabel(r'$\Delta$ratio'),
        ax_i.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)),
        ax_i.xaxis.set_major_locator(MultipleLocator(5)),
        ax_i.xaxis.set_minor_locator(MultipleLocator(0.2)),
        pdf.savefig(fig_i)) for fig_i, ax_i in zip([fig_diff], [ax_diff])]

        [(ax_i.legend(),
        ax_i.set_xlabel(r'$\mu$'),
        ax_i.set_ylabel(r'$N_{obs}$'),
        ax_i.yaxis.set_major_locator(MultipleLocator(0.1)),
        ax_i.xaxis.set_major_locator(MultipleLocator(5)),
        ax_i.xaxis.set_minor_locator(MultipleLocator(0.2)),
        pdf.savefig(fig_i)) for fig_i, ax_i in zip([fig_c, fig_non_c], [ax_c, ax_non_c])]

        [(ax_i.legend(),
        ax_i.set_xlabel(r'$\mu$'),
        ax_i.set_ylabel(r'$\Delta N_{obs}$'),
        ax_i.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)),
        ax_i.xaxis.set_major_locator(MultipleLocator(5)),
        ax_i.xaxis.set_minor_locator(MultipleLocator(0.2)),
        pdf.savefig(fig_i)) for fig_i, ax_i in zip([fig_diff_c], [ax_diff_c])]

