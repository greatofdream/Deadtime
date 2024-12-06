from skpy.DetectorParaReader import PMTReader, spefit, QBEE
from scipy.interpolate import interp1d
import numpy as np

class SpeSampler():
    def __init__(self):
        self.pmtr = PMTReader()
        print(np.unique(self.pmtr.pmtinfo['type'], return_counts=True))
        cdf_bins = np.arange(1000)
        self.Q2 = interp1d(self.pmtr.speinfo[2][:1000], cdf_bins, bounds_error=False, fill_value=(0, 1000))
        self.Q3 = interp1d(self.pmtr.speinfo[3][:1000], cdf_bins, bounds_error=False, fill_value=(0, 1000))
        self.Q5 = interp1d(self.pmtr.speinfo[5][:1000], cdf_bins, bounds_error=False, fill_value=(0, 1000))
    def sample(self, cdf_rand, pmttype=2):
        if pmttype==2:
            return self.Q2(cdf_rand) / self.pmtr.speChargePE.iloc[0]['scale']
        elif pmttype==3:
            return self.Q3(cdf_rand) / self.pmtr.speChargePE.iloc[1]['scale']
        elif pmttype==5:
            return self.Q5(cdf_rand) / self.pmtr.speChargePE.iloc[2]['scale']

class Elec():
    def __init__(self):
        self.elec = QBEE()
    def GetEfficiency(self, dDeltaT):
        return self.elec.GetQBeeEfficiency(np.array([dDeltaT]))[0]
    def IsElecHit(self, dHitCharge, PMTType, Geometry=7):
        if Geometry>=4:
            return self.elec.IsQBeeHit(dHitCharge, PMTType, Geometry)
