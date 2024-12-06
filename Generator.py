import numpy as np
from scipy.interpolate import interp1d
class Rectangle():
    def __init__(self):
        self.t_l, self.t_r = 0, 200
        self.N = 200
        self.TBIN = 1
        self.R_j = 1/self.N
        self.R_js = np.repeat([self.R_j], self.N+1) #[0, N], N+1 points
        self.fun = interp1d(np.arange(self.t_l, self.t_r+1), self.R_js, bounds_error=False, fill_value=(0, 0))
        self.integration = np.insert(np.cumsum(self.R_js[1:]), 0, 0) # [0, N], N+1 points
        self.integration_fun = interp1d(np.arange(self.t_l, self.t_r+1), self.integration, bounds_error=False, fill_value=(0, 1))
        self.integration_reverse_fun = interp1d(self.integration, np.arange(self.t_l, self.t_r+1), bounds_error=False, fill_value=(100, 100))
class Bonsai():
    def __init__(self):
        from skpy.DetectorParaReader import BonsaiLikelihoodReader
        self.bonsaiLikelihoodReader = BonsaiLikelihoodReader()
        self.bonsaiLikelihoodReader.loadH5()
        bonsaiLikelihood = self.bonsaiLikelihoodReader.pdfs[0, :]
        # normalize
        bonsaiLikelihood /= np.sum(bonsaiLikelihood) * self.bonsaiLikelihoodReader.TBIN # already to be the probability
        self.N = bonsaiLikelihood.shape[0]
        self.TBIN = self.bonsaiLikelihoodReader.TBIN
        self.R_js = bonsaiLikelihood[::-1]
        self.bonsaiIntegration = np.insert(np.cumsum((self.R_js[:-1] + self.R_js[1:])/2), 0, 0) * self.bonsaiLikelihoodReader.TBIN
        self.t_l, self.t_r = (-bonsaiLikelihood.shape[0] + int(self.bonsaiLikelihoodReader.offsets[1])) * self.bonsaiLikelihoodReader.TBIN, int(self.bonsaiLikelihoodReader.offsets[1]-1) * self.bonsaiLikelihoodReader.TBIN

        self.bonsaiLikelihood_fun = interp1d((np.arange(-bonsaiLikelihood.shape[0], 0) + self.bonsaiLikelihoodReader.offsets[1]) * self.bonsaiLikelihoodReader.TBIN, bonsaiLikelihood[::-1], bounds_error=False, fill_value=0)
        self.bonsaiIntegration_fun = interp1d((np.arange(-bonsaiLikelihood.shape[0], 0) + self.bonsaiLikelihoodReader.offsets[1]) * self.bonsaiLikelihoodReader.TBIN, self.bonsaiIntegration, bounds_error=False, fill_value=(0, 1))
        self.bonsaiIntegration_reverse_fun = interp1d(self.bonsaiIntegration, (np.arange(-bonsaiLikelihood.shape[0], 0) + self.bonsaiLikelihoodReader.offsets[1]) * self.bonsaiLikelihoodReader.TBIN, bounds_error=False, fill_value=(100, 100))
        self.fun = self.bonsaiLikelihood_fun
        self.integration_fun = self.bonsaiIntegration_fun
        self.integration_reverse_fun = self.bonsaiIntegration_reverse_fun

class Generator():
    def __init__(self, model='rectangle'):
        self.model = model
        if model=='Rectangle':
            self.generator = Rectangle()
        elif model=='Bonsai':
            self.generator = Bonsai()
        else:
            assert(1, "no suitable generator")
