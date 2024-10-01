import numpy as np
from scipy.special import logsumexp
from scipy import optimize

def interp(R_m_t_js, index_int, index_frac):
    row_index = np.arange(R_m_t_js.shape[0])
    return R_m_t_js[row_index, index_int] * (1 - index_frac) + R_m_t_js[row_index, index_int+1] * index_frac

def integrateBin(h1, h2, frac):
    return (frac**2 * h2 + (2 - frac) * frac * h1) / 2

def integrateCumSum(R_m_t_js, index_l_int, index_l_frac, index_r_int, index_r_frac):
    # cumsum is so slow
    F_R_m_t_js = np.cumsum(R_m_t_js, axis=1)
    lambda_js = interp(F_R_m_t_js, index_r_int, index_r_frac) - interp(F_R_m_t_js, index_l_int, index_l_frac)
    return lambda_js

def integrateSum(R_m_t_js, index_l_int, index_l_frac, index_r_int, index_r_frac, index_mask):
    row_index = np.arange(R_m_t_js.shape[0])
    R_l_1, R_l_2 = R_m_t_js[row_index, index_l_int], R_m_t_js[row_index, index_l_int+1]
    R_r_1, R_r_2 = R_m_t_js[row_index, index_r_int], R_m_t_js[row_index, index_r_int+1]
    lambda_js = np.sum(R_m_t_js, where=index_mask, axis=1) - R_l_1/2 + R_r_1/2
    lambda_js += integrateBin(R_r_1, R_r_2, index_r_frac) - integrateBin(R_l_1, R_l_2, index_l_frac)
    return lambda_js

class Corr(object):
    def __init__(self, R_js, T_Dead, lc_TBIN):
        # R_js: the likelihood of light curve
        # T_Dead: time window for light curve
        # lc_TBIN: bin width of the light curve
        self.T_Dead, self.lc_TBIN = T_Dead, lc_TBIN
        self.N_Rt = R_js.shape[0]
        self.N_Rt_padding = int((T_Dead + 100) // lc_TBIN) # padding 100ns for the t0 is outside of the sample window, just a temporary solution
        self.Rt_js = np.zeros((self.N_Rt_padding,)) # padding the following bit with zero
        self.Rt_js[:self.N_Rt] = R_js
        self.cumsum_Rt_js = np.insert(np.cumsum((self.Rt_js[:-1] + self.Rt_js[1:]) / 2), 0, 0)
        self.F_Rt_js = self.cumsum_Rt_js * lc_TBIN # The CDF of lc, should weighted with TBIN

class CorrExp(Corr):
    # it is a wrong deduction of the dead time correction with Poisson distribution assumption for the R_m
    def __init__(self, R_js, T_Dead, lc_TBIN, pmt_b_js, pmt_b_corr_js):
        super(CorrExp, self).__init__(R_js, T_Dead, lc_TBIN)
        self.F_Rt_conv_js = np.zeros(self.N_Rt_padding)
        for i in range(1, N_Rt_padding):
            self.F_Rt_conv_js[i] = self.F_Rt_conv_js[i-1] + self.cumsum_Rt_js[i-1]
        self.F_Rt_conv_js *= self.lc_TBIN
        ## denominator
        self.Denominator = self.F_Rt_js - self.F_Rt_conv_js * pmt_b_corr_js[:, np.newaxis] # [N_PMT, T_Dead/TBIN]
        self.pmt_b_js = pmt_b_js
        self.pmt_b_corr_js = pmt_b_corr_js

    #@profile
    def NeffLikelihood_T(self, xs, *args):
        # Rt_js, F_Rt_js: (1, T) or (N_PMT, T)
        # pmt_b_js: (N_PMT,)
        neff = xs
        pmt_k_js, useless_index, hit_index, r_max, index_l_int, index_l_frac, index_r_int, index_r_frac, T_noise_pre, index_mask, integrate_index, R_m_t_js = args
    
        pmt_K_js = neff * pmt_k_js
        # (pmt_K_js[:, np.newaxis] * Rt_js + pmt_b_js[:, np.newaxis]) / (pmt_K_js[:, np.newaxis] * Denominator + 1 + pmt_b_corr_js[:, np.newaxis])
        # R_m_t_js = np.divide(np.add(np.multiply(pmt_K_js[:, np.newaxis], Rt_js, where=index_mask), pmt_b_js[:, np.newaxis], where=index_mask), np.add(np.multiply(pmt_K_js[:, np.newaxis], Denominator, where=index_mask), 1 + (pmt_b_corr_js * T_Dead)[:, np.newaxis], where=index_mask), where=index_mask)
        # R_m_t_js[:] = np.divide(np.add(np.multiply(pmt_K_js[:, np.newaxis], Rt_js), pmt_b_js[:, np.newaxis]), np.add(np.multiply(pmt_K_js[:, np.newaxis], Denominator), 1 + (pmt_b_corr_js * T_Dead)[:, np.newaxis]))
        # R_m_t_js[:] = 1 #np.ones((pmt_K_js.shape[0], Rt_js.shape[0])) #np.add(np.multiply(pmt_K_js[:, np.newaxis], Rt_js), pmt_b_js[:, np.newaxis])
        # sleeptime, times = 0.5, 1
        #print(f'sleep k-th: {times}'); sleep(sleeptime); times +=1;
        np.multiply(pmt_K_js[:, np.newaxis], self.Rt_js[:r_max], out=R_m_t_js[0])
        #print(f'sleep k-th: {times}'); sleep(sleeptime); times +=1;
        R_m_t_js[0, :] += self.pmt_b_js[~useless_index, np.newaxis]
        #print(f'sleep k-th: {times}'); sleep(sleeptime); times +=1;
        np.multiply(pmt_K_js[:, np.newaxis], self.Denominator[~useless_index], out=R_m_t_js[1])
        #print(f'sleep k-th: {times}'); sleep(sleeptime); times +=1;
        R_m_t_js[1, :] += 1 + (self.pmt_b_corr_js[~useless_index] * self.T_Dead)[:, np.newaxis]
        #print(f'sleep k-th: {times}'); sleep(sleeptime); times +=1;
        R_m_t_js[0] /=R_m_t_js[1]
        #print(f'sleep k-th: {times}'); sleep(sleeptime); times +=1;
        Int_R_m_t_js = integrateSum(R_m_t_js[0], index_l_int, index_l_frac, index_r_int, index_r_frac, integrate_index_mask) * self.lc_TBIN
    
        # pmt_b_corr_js * T_noise_pre is useless when not minimize the t0
        # lambda_live_js = pmt_b_corr_js * T_noise_pre + Int_R_m_t_js
        lambda_live_js = Int_R_m_t_js
        hit_R_js = interp(R_m_t_js[0, hit_index], index_r_int[hit_index], index_r_frac[hit_index])
        likelihood = np.sum(lambda_live_js) - np.sum(np.log(hit_R_js))
        return likelihood

class NonCorr(Corr):
    # not consider the dead time effect correction on the Rt. Still consider the dead time in the likelihood
    def __init__(self, R_js, T_Dead, lc_TBIN, pmt_b_js):
        super(NonCorr, self).__init__(R_js, T_Dead, lc_TBIN)
        self.pmt_b_js = pmt_b_js

    def Rt(self, useless_index, hit_index, index_l_int, index_l_frac, index_r_int, index_r_frac, r_max, live_window):
        self.hat_hit_R_js = interp(self.Rt_js[np.newaxis, :r_max], index_r_int[hit_index], index_r_frac[hit_index])
        self.hat_lambda_live_js = interp(self.F_Rt_js[np.newaxis, :r_max], index_r_int, index_r_frac) - interp(self.F_Rt_js[np.newaxis, :r_max], index_l_int, index_l_frac)
        self.pmt_b_js_T = self.pmt_b_js[~useless_index] * live_window

    def NeffLikelihood_T(self, xs, *args):
        neff = xs
        self.Rt_m_js = neff * self.Rt_js[np.newaxis, :r_max] + pmt_b_js
        pmt_k_js, useless_index, hit_index = args
        pmt_K_js = neff * pmt_k_js
        # expected photon number * lc + b
        # likelihood_j = R_m(t)*(1-lambda_live)
        likelihood = np.sum(pmt_K_js * self.hat_lambda_live_js + self.pmt_b_js_T) - np.sum(np.log(pmt_K_js[hit_index] * self.hat_hit_R_js + self.pmt_b_js[~useless_index][hit_index]))
        return likelihood

    def NeffLikelihood_TQ(self, xs, *args):
        pass

class CorrParalyzable(Corr):
    # use paralyzable model to do the correction on the Rt.
    def __init__(self, R_js, T_Dead, lc_TBIN, pmt_b_js, pmt_b_corr_js):
        super(CorrParalyzable, self).__init__(R_js, T_Dead, lc_TBIN)
        self.pmt_b_js = pmt_b_js
        self.pmt_b_corr_js = pmt_b_corr_js

    def Rt(self, useless_index, hit_index, index_l_int, index_l_frac, index_r_int, index_r_frac, r_max, noise_pre_window, integrate_index_mask):
        # calculate the variable which not contain lambda
        # useless_index: hit pmt index but hit time is out of the start of the t0
        # hit_index, index_l_int, index_l_frac, index_r_int, index_r_frac: remove the useless_index
        self.useless_index = useless_index
        self.hit_index = hit_index

        self.pmt_b_corr_js_prewindow = self.pmt_b_corr_js * noise_pre_window
        self.pmt_b_js_T = self.pmt_b_js * self.T_Dead

        self.hat_hit_R_js = interp(self.Rt_js[np.newaxis, :r_max], index_r_int[hit_index], index_r_frac[hit_index]) * self.lc_TBIN
        assert((self.hat_hit_R_js[~useless_index]>=0).all())
        self.hat_hit_lambda_live_js = interp(self.F_Rt_js[np.newaxis, :r_max], index_r_int[hit_index], index_r_frac[hit_index])  # T_Dead window, the left window must outside the T_left, thus interp(self.F_Rt_js[np.newaxis, :r_max], index_l_int[hit_index], index_l_frac[hit_index])=0

        self.nohitN = np.sum(~hit_index)
        if self.nohitN!=0:
            self.F_R_js_tl = interp(self.F_Rt_js[np.newaxis, :r_max], index_l_int[~hit_index][0], index_l_frac[~hit_index][0])
            self.F_R_js_tr = interp(self.F_Rt_js[np.newaxis, :r_max], index_r_int[~hit_index][0], index_r_frac[~hit_index][0])
        else:
            self.F_R_js_tl = []
            self.F_R_js_tr = []

        self.r_max = r_max
        self.index_l_int, self.index_l_frac = index_l_int, index_l_frac
        self.index_r_int, self.index_r_frac = index_r_int, index_r_frac
        self.integrateIndexMask = integrate_index_mask


    def Rt_m(self, pmt_K_js, hit_index):
        # self.Rt_m_js = (pmt_K_js * self.Rt_js[np.newaxis, :r_max] + self.pmt_b_js[~useless_index]) * np.exp(- neff * self.F_Rt_js[np.newaxis, :r_max] - self.pmt_b_js[~useless_index])
        self.exp_R_js = np.exp(-pmt_K_js * self.F_Rt_js[:self.r_max])
        self.F_exp_R_js = integrateSum(self.exp_R_js[np.newaxis, :], self.index_l_int[~hit_index][0], self.index_l_frac[~hit_index][0], self.index_r_int[~hit_index][0], self.index_r_frac[~hit_index][0], self.integrateIndexMask[~hit_index][0]) * self.lc_TBIN

    def NeffLikelihood_T(self, xs, *args):
        neff = xs
        hit_index = args[0]
        pmt_K_js = neff
        likelihood = np.sum(pmt_K_js * self.hat_hit_lambda_live_js[~self.useless_index] + self.pmt_b_js_T) - np.sum(np.log(pmt_K_js * self.hat_hit_R_js[~self.useless_index] + self.pmt_b_js)) # R_m, noise part cdf is not related with pmt_K_js, which is omit
        if self.nohitN!=0:
            self.Rt_m(pmt_K_js, hit_index)
            likelihood += - self.nohitN * np.log(1 - (np.exp(-pmt_K_js * self.F_R_js_tl) - np.exp(-pmt_K_js * self.F_R_js_tr)) * np.exp(-self.pmt_b_js_T))
            #likelihood += - self.nohitN * np.log(1 - (np.exp(-pmt_K_js * self.F_R_js_tl) - np.exp(-pmt_K_js * self.F_R_js_tr) + self.pmt_b_js * self.F_exp_R_js) * np.exp(-self.pmt_b_js_T) - self.pmt_b_corr_js_prewindow[~hit_index][0])
        # expected photon number * lc + b
        # likelihood_j = nonhit probability
        # + R_m(t)*(1-lambda_live)
        return likelihood

    def NeffLikelihood_TQ(self, xs, *args):
        pass

class CorrNonParalyzable(Corr):
    # use non-paralyzable model to do the correction on the Rt.
    def __init__(self, R_js, T_Dead, lc_TBIN, pmt_b_js):
        super(CorrNonParalyzable, self).__init__(R_js, T_Dead, lc_TBIN)
        self.pmt_b_js = pmt_b_js

    def Rt(self, useless_index, hit_index, index_l_int, index_l_frac, index_r_int, index_r_frac, r_max, live_window, integrate_index_mask):
        # calculate the variable which not contain lambda
        # useless_index: hit pmt index but hit time is out of the start of the t0
        # hit_index, index_l_int, index_l_frac, index_r_int, index_r_frac: remove the useless_index
        self.useless_index = useless_index # for the pmt_b_js selection

        self.pmt_b_js_livewindow = self.pmt_b_js[~useless_index] * live_window
        self.pmt_b_js_T = self.pmt_b_js[~useless_index] * self.T_Dead

        self.hat_hit_R_js = interp(self.Rt_js[np.newaxis, :r_max], index_r_int[hit_index], index_r_frac[hit_index])
        self.hat_hit_lambda_live_js = interp(self.F_Rt_js[np.newaxis, :r_max], index_r_int[hit_index], index_r_frac[hit_index]) + self.pmt_b_js_T[hit_index]  # T_Dead window, the left window must outside the T_left, thus interp(self.F_Rt_js[np.newaxis, :r_max], index_l_int[hit_index], index_l_frac[hit_index])=0

        self.exp_R_js_Storage = np.empty((2, np.sum(~hit_index), r_max)) # pre allocate the storage
        self.F_R_js_tl = interp(self.F_Rt_js[np.newaxis, :r_max], index_l_int[~hit_index], index_l_frac[~hit_index])
        self.F_R_js_tr = interp(self.F_Rt_js[np.newaxis, :r_max], index_r_int[~hit_index], index_r_frac[~hit_index])
        self.r_max = r_max
        self.index_l_int, self.index_l_frac = index_l_int, index_l_frac
        self.index_r_int, self.index_r_frac = index_r_int, index_r_frac
        self.integrateIndexMask = integrate_index_mask


    def Rt_m(self, pmt_K_js, hit_index):
        np.exp(-np.multiply(pmt_K_js[~hit_index, np.newaxis], self.F_Rt_js[np.newaxis, :self.r_max], out=self.exp_R_js_Storage[0]), out=self.exp_R_js_Storage[1])
        self.F_exp_R_js = integrateSum(self.exp_R_js_Storage[1], self.index_l_int[~hit_index], self.index_l_frac[~hit_index], self.index_r_int[~hit_index], self.index_r_frac[~hit_index], self.integrateIndexMask[~hit_index]) * self.lc_TBIN

    def NeffLikelihood_T(self, xs, *args):
        neff = xs
        pmt_k_js, useless_index, hit_index = args
        pmt_K_js = neff * pmt_k_js
        self.Rt_m(pmt_K_js, hit_index)
        # expected photon number * lc + b
        # likelihood_j = nonhit probability
        # + R_m(t)*(1-lambda_live)
        likelihood = -np.sum(np.log(1 - (np.exp(-pmt_K_js[~hit_index] * self.F_R_js_tl) - np.exp(-pmt_K_js[~hit_index] * self.F_R_js_tr) + self.pmt_b_js[~useless_index][~hit_index] * self.F_exp_R_js) * np.exp(-self.pmt_b_js_T[~hit_index])))\
            + np.sum(pmt_K_js[hit_index] * self.hat_hit_lambda_live_js) - np.sum(np.log(pmt_K_js[hit_index] * self.hat_hit_R_js + self.pmt_b_js[~useless_index][hit_index])) # R_m
        return likelihood
