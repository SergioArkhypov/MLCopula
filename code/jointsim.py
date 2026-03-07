import numpy as np
from scipy import stats
import pandas as pd
import copy
import matplotlib.pyplot as plt


class JointSim:
    def __init__(self, calib_data, rf_dir, seed=0):
        self.seed = seed
        np.random.seed(seed)
        self.figpath = 'C:\\dev\\MLCopula\\document\\figures'
        self.name = 'Not implemented'
        self.calib_data = copy.deepcopy(calib_data)
        
        i = 0
        for rf in calib_data.columns:
            if np.isnan(rf_dir[i]): self.calib_data = self.calib_data.drop(columns=[rf])
            if rf_dir[i]==-1.0: self.calib_data[rf] = 1.0 - self.calib_data[rf] 
            i+=1

        self.rf_dir = rf_dir
        self.size = self.calib_data.shape[0]
        self.sizerf = self.calib_data.shape[1]
    
    def get_sims(self, scen_number):
        raise NotImplementedError
    
    def plot_sims(self, rf1 : int, rf2 : int):
        plt.title(f'Uniform scenarios: {self.name}')
        plt.scatter(self.sims.iloc[:,rf1], self.sims.iloc[:,rf2], s =0.3)
        title=self.name.replace(' ', '_')
        plt.savefig(f'{self.figpath}\\Uniform_scenarios_{title}.png')
        plt.show()

    def plot_contour(self, rf1 : int, rf2 : int, step = 0.025):
        plt.title(f'Contour plot scenarios: {self.name}')
        nscen = self.sims.shape[0]
        x_vals = np.linspace(0.0, 1.0, int(1.0/step))
        y_vals = np.linspace(0.0, 1.0, int(1.0/step))
        x, y = np.meshgrid(x_vals, y_vals)

        data = np.vstack([self.sims.iloc[:,rf1], self.sims.iloc[:,rf2]])
        kernel = stats.gaussian_kde(data)
        z = kernel.evaluate(np.vstack([x.ravel(), y.ravel()]))/nscen
        plt.contourf(x, y, z.reshape(x.shape), cmap='Greens')
        plt.colorbar()
        title=self.name.replace(' ', '_')
        plt.savefig(f'{self.figpath}\\Contour_plot_{title}.png')
        plt.show()

    def plot_pdf(self, rf1 : int, rf2 : int, step = 0.025):
        nscen = self.sims.shape[0]
        fig = plt.figure()
        #plt.title(f'PDF: {self.name}')
        ax = fig.add_subplot(111, projection='3d')
        
        x_vals = np.linspace(0.0, 1.0, int(1.0/step))
        y_vals = np.linspace(0.0, 1.0, int(1.0/step))
        x, y = np.meshgrid(x_vals, y_vals)

        data = np.vstack([self.sims.iloc[:,rf1], self.sims.iloc[:,rf2]])
        kernel = stats.gaussian_kde(data)
        z = kernel.evaluate(np.vstack([x.ravel(), y.ravel()]))/nscen
        ax.plot_surface(x, y, z.reshape(x.shape), cmap='Greens')
        #ax.plot_wireframe(x, y, z.reshape(x.shape))
        title=self.name.replace(' ', '_')
        plt.savefig(f'{self.figpath}\\PDF_{title}.png')
        plt.show()


class GaussCopulaCorr(JointSim):
    def __init__(self, calib_data, rf_dir, seed=0):
        JointSim.__init__(self, calib_data, rf_dir, seed)
        self.name ='Gaussian copula (correlation based)'
    
    def get_sims(self, scen_number, override_corr=None):
        mean = np.zeros((self.sizerf,))
        corr = self.calib_data.corr()
        if override_corr is not None:
            corr.loc[:] = override_corr
            np.fill_diagonal(corr.values, 1.0)
        data = np.random.multivariate_normal(mean, corr, size=scen_number)
        df = pd.DataFrame(data=data, columns=self.calib_data.columns)
        self.sims = df.rank(axis=0, pct=True) 
        return self.sims


class HistSimulation(JointSim):
    def __init__(self, calib_data, rf_dir, seed=0):
        JointSim.__init__(self, calib_data, rf_dir, seed)
        self.name ='Historical simulation'
    
    def get_sims(self, scen_number):
        scen = self.calib_data.rank(axis=0,pct=True)
        repeat = round(scen_number/self.calib_data.shape[0], 0)
        self.sims = pd.concat([scen] * int(repeat), ignore_index=True) 
        return self.sims


class GaussCopulaNum(JointSim):
    def __init__(self, calib_data, rf_dir, seed=0):
        JointSim.__init__(self, calib_data, rf_dir, seed)
        self.name = 'Gaussian copula (numerical simulation)'
    
    def get_sims(self, scen_number):
        z = np.array([np.random.standard_normal(scen_number) for s in range(self.size)]) #np.random.standard_normal(scen_number, size)
        sim = np.matmul(np.transpose(z), self.calib_data)
        sim_stddev = sim.std(axis=0, ddof=1)

        self.sims = copy.deepcopy(sim)
        for sn in self.calib_data.columns:
            self.sims[sn] = stats.norm.cdf(sim[sn], loc=0.0, scale=sim_stddev[sn])

        return self.sims


class CauchyCopulaNum(JointSim):
    def __init__(self, calib_data, rf_dir, seed=0):
        JointSim.__init__(self, calib_data, rf_dir, seed)
        self.name = 'Cauchy copula (numerical simulation)'
    
    def get_sims(self, scen_number):
        z = np.array([np.random.standard_cauchy(scen_number) for s in range(self.size)])
        sim = np.matmul(np.transpose(z), self.calib_data)

        #Wiki version: as we perform linear combination sum(ksi_cauchy(0, 1)* weight) location should be zero
        # as Cauchy is a stable distribution. In this case we can estimate second parameter (scale) as median of abs values
        # https://en.wikipedia.org/wiki/Cauchy_distribution
        
        scales  = np.median(np.abs(sim), axis=0)
        sim_scale = pd.DataFrame(data=[scales], columns=sim.columns)

        self.sims = copy.deepcopy(sim)
        for sn in self.calib_data:
            sc = sim_scale[sn]
            self.sims[sn] = stats.cauchy.cdf(sim[sn], loc=0.0, scale=sc)

        return self.sims


class MixCopulaNum(JointSim):
    def __init__(self, calib_data, rf_dir, seed=0, weights=None):
        JointSim.__init__(self, calib_data, rf_dir, seed)
        self.name =f'Mixture copula g{weights[0]},c{weights[1]},u{weights[2]},i{weights[3]}'
        self.weights=weights
    
    def get_sims(self, scen_number):
        c1 = GaussCopulaNum(self.calib_data, self.rf_dir, self.seed)
        c2 = CauchyCopulaNum(self.calib_data, self.rf_dir, self.seed)
        c3 = GaussCopulaCorr(self.calib_data, self.rf_dir, self.seed)
        self.sims = pd.concat(
            [c1.get_sims(int(scen_number*self.weights[0])), 
             c2.get_sims(int(scen_number*self.weights[1])),
             c3.get_sims(int(scen_number*self.weights[2]), override_corr = 1.0),
             c3.get_sims(int(scen_number*self.weights[3]), override_corr = 0.0)
             ], ignore_index=True)
        return self.sims