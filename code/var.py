import numpy as np
from scipy import stats
import pandas as pd
import copy
import matplotlib.pyplot as plt


class Var:
    def __init__(self, market, joint_sim, portfolio):
        self.market = market
        self.joint_sim = joint_sim
        self.portfolio = portfolio

    def caclulate_pnls():
        raise NotImplementedError

    def get_quantile(quantile):
        return np.quantile(self.pnls, quantile)


class MCVar(Var):
    def __init__(self, market, joint_sim, portfolio):
        Var.__init__(self, market, joint_sim, portfolio)
        self.name ='MC VaR'

    def caclulate_pnls():
        scen = pd.DataFrame(data=np.array([np.quantile(self.market[x], self.joint_sim[x]) for x in self.market.columns ]).T, columns=self.market.columns)
        self.pnls = scen.dot(portfolio.T).sum(axis=1) 
         



