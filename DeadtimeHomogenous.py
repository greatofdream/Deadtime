import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

darkRates = np.arange(0, 100000) # kHz or no per ms
TDead = 900#ns
R_m_paralyzable = darkRates * np.exp(-darkRates * TDead * 1E-6)
upper_limit = 1/TDead*1E6
R_m_unparalyzable = np.arange(0, upper_limit)
R_unparalyzable = R_m_unparalyzable / (1 - R_m_unparalyzable * TDead * 1E-6)
R_m_unparalyzable_poisson = np.arange(0, 2*upper_limit)
# R_unparalyzable_poisson = R_m_unparalyzable_poisson * np.exp(R_m_unparalyzable_poisson * TDead * 1E-6)
with PdfPages('un_paralyzable.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.plot(darkRates, R_m_paralyzable, label='Paralyzable')
    ax.plot(R_unparalyzable, R_m_unparalyzable, label='Unparalyzable')
    # ax.plot(R_unparalyzable_poisson, R_m_unparalyzable_poisson, label='Unparalyzable (approximation)')
    ax.plot(darkRates, darkRates, ls='--', label='no correction')
    ax.axhline(upper_limit, ls='--')
    ax.set_xlabel('true rate[#/ms]')
    ax.set_ylabel('observed rate[#/ms]')
    ax.set_xlim([0, 100000])
    ax.set_ylim([0, 2*upper_limit])
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))
    ax.set_xlim([0, 10000])
    pdf.savefig(fig)
