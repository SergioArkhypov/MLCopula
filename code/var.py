import numpy as np
import pandas as pd


class Var:
    def __init__(self, calibration, spot, copula, portfolio):
        self.calibration = calibration
        self.spot = spot
        self.copula = copula
        self.portfolio = portfolio
        self.pnls = None
        self.name = 'Not implemented'

    def calculate_pnls(self):
        raise NotImplementedError

    def get_quantile(self, quantile):
        return np.quantile(self.pnls, quantile)
    
    def get_port_value(self):
        return self.spot.dot(self.portfolio.T)


class MCVar(Var):
    def __init__(self, calibration, spot, copula, portfolio, method='linear'):
        Var.__init__(self, calibration, spot, copula, portfolio)
        self.name =f'MC VaR - Empirical marginals ({method}) - {self.copula.name}'
        self.method = method

    def calculate_pnls(self, numb_scen):
        joint_sim = self.copula.get_sims(numb_scen)
        #joint_sim.to_csv(f'C:\\Temp\\joint_sim.csv')
        #self.spot.to_csv(f'C:\\Temp\\spot.csv')
        scen = pd.DataFrame(
            data=np.array([(np.exp(np.quantile(self.calibration[x], joint_sim[x], method=self.method))-1.0)*self.spot[x] for x in self.calibration.columns ]).T, 
            #data=np.array([np.quantile(self.calibration[x], joint_sim[x]) for x in self.calibration.columns ]).T,
            columns=self.calibration.columns)
        #scen.to_csv(f'C:\\Temp\\scen.csv')
        print(f'Simulation stats, non nones count: {sum(list(scen.count()))}, needs {scen.size}')
        self.pnls = scen.dot(self.portfolio.T)
         



