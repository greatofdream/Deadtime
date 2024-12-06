from skpy.DetectorParaReader import PMTReader, spefit, QBEE
from scipy.interpolate import interp1d

class SpeSampler():
    def __init__(self):
        self.pmtr = PMTReader()
        print(np.unique(pmtr.pmtinfo['type'], return_counts=True))
        cdf_bins = np.arange(1000)
        self.Q2 = interp1d(self.pmtr.speinfo[2][:1000], cdf_bins, bounds_error=False, fill_value=(0, 1000))
        self.Q3 = interp1d(self.pmtr.speinfo[3][:1000], cdf_bins, bounds_error=False, fill_value=(0, 1000))
        self.Q5 = interp1d(self.pmtr.speinfo[5][:1000], cdf_bins, bounds_error=False, fill_value=(0, 1000))
    def sample(self, cdf_rand, pmttype=2):
        if pmttype==2:
            return self.Q2(cdf_rand)
        elif pmttype==3:
            return self.Q3(cdf_rand)
        elif pmttype==5:
            return self.Q5(cdf_rand)

class Elec():
    def __init__(self):
        self.elec = QBEE()
    def GetEfficiency(self, dDeltaT):
        return self.elec.GetQBeeEfficiency([dDeletaT])[0]
    def IsElecHit(self, dHitCharge, PMTType, Geometry=7):
        if Geometry>=4:
            return self.elec.IsQBeeHit(dHitCharge, PMTType, Geometry)
