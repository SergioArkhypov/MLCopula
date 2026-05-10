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
        pearson = round(self.sims.iloc[:,rf1].corr(self.sims.iloc[:,rf2], method='pearson'), 2)
        spearmn = round(self.sims.iloc[:,rf1].corr(self.sims.iloc[:,rf2], method='spearman'),2)
        kendall = round(self.sims.iloc[:,rf1].corr(self.sims.iloc[:,rf2], method='kendall'), 2)
        title=self.name.replace(' ', '_')
        #plt.title(f'Uniform scen: {self.name} (p:{pearson}, s:{spearmn}, k:{kendall})')
        plt.title(f'Uniform scen.: {self.name}')
        plt.scatter(self.sims.iloc[:,rf1], self.sims.iloc[:,rf2], s =0.3)
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


    def plot_tailcoef(self, rf1 : int, rf2 : int, step = 0.025):
        plt.title(f'Tail coefficient: {self.name}')
        max_list = np.maximum(self.sims.iloc[:,rf1], self.sims.iloc[:,rf2])
        min_list = np.minimum(self.sims.iloc[:,rf1], self.sims.iloc[:,rf2])
        count_sim = len(self.sims.iloc[:,rf1])
        indexlist = ['Q01', 'Q1', 'Q5', 'Q10', 'Q25', 'Q50-', 'Q50+', 'Q75', 'Q90', 'Q95', 'Q99', 'Q999']
        #indexlist1 = [0.01, 0.05, 0.1, 0.25, 0.499, 0.501, 0.75, 0.90, 0.95, 0.99]

        vals = [
            np.count_nonzero(max_list <= 0.001)/count_sim*1000.0,
            np.count_nonzero(max_list <= 0.01)/count_sim*100.0,
            np.count_nonzero(max_list <= 0.05)/count_sim*20.0,
            np.count_nonzero(max_list <= 0.1)/count_sim*10.0,
            np.count_nonzero(max_list <= 0.25)/count_sim*4.0,
            np.count_nonzero(max_list <= 0.5)/count_sim*2.0,
            np.count_nonzero(min_list >= 0.5)/count_sim*2.0,
            np.count_nonzero(min_list >= 0.75)/count_sim*4.0,
            np.count_nonzero(min_list >= 0.9)/count_sim*10.0,
            np.count_nonzero(min_list >= 0.95)/count_sim*20.0,
            np.count_nonzero(min_list >= 0.99)/count_sim*100.0,
            np.count_nonzero(min_list >= 0.999)/count_sim*1000.0,
        ]
        plt.ylim(0.0, 1.0)
        plt.plot(indexlist, vals, marker='o')
        title=self.name.replace(' ', '_')
        plt.savefig(f'{self.figpath}\\Tail_coefficient_{title}.png')
        plt.show()



class GaussCopulaCorr(JointSim):
    def __init__(self, calib_data, rf_dir, seed=0):
        JointSim.__init__(self, calib_data, rf_dir, seed)
        self.name ='Gaussian copula (correlation based)'
    
    def get_sims(self, scen_number, override_corr=None):
        mean = np.zeros((self.sizerf,))
        corr = self.calib_data.corr()
        if override_corr is not None:
            temp = corr.values
            temp.setflags(write=1)
            temp[:,:] = override_corr
            np.fill_diagonal(temp, 1.0)
            temp_corr = pd.DataFrame(data=temp, index=corr.index, columns=corr.columns)
            corr = temp_corr
        data = np.random.multivariate_normal(mean, corr, size=scen_number)
        df = pd.DataFrame(data=data, columns=self.calib_data.columns)
        sims_np = stats.norm.cdf(df, loc=0.0, scale=1.0)
        df = pd.DataFrame(data=sims_np, columns=self.calib_data.columns)
        #self.sims = df.rank(axis=0, pct=True) 
        self.sims = df
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


class SkewCopulaNum(JointSim):
    def __init__(self, calib_data, rf_dir, seed=0, weights=None):
        JointSim.__init__(self, calib_data, rf_dir, seed)
        self.name =f'Skewed Cauchy copula'
        self.weights=weights
    
    def get_sims(self, scen_number):
        c1 = CauchyCopulaNum(self.calib_data, self.rf_dir, self.seed)
        
        sim = c1.get_sims(int(scen_number))
        theta_w=pd.DataFrame(data=[self.weights], columns=self.calib_data.columns)

        self.sims = copy.deepcopy(sim)
        for sn in self.calib_data.columns:
            theta = theta_w[sn][0]
            # self.sims[sn] = [u/(2*theta) if u<=theta
            #                  else ((u-theta)/(1 - 2*theta) if u <= 1-theta else (u-1+2*theta)/(2*theta)) 
            #                  for u in sim[sn]]
            self.sims[sn] = [u/theta if u<=theta else ((1-u)/(1-theta)) for u in sim[sn]] if theta!=1.0 or theta!=0.0 else sim[sn]

        return self.sims
    

class MixCopulaNumTest(JointSim):
    def __init__(self, calib_data, rf_dir, seed=0, weights=None):
        JointSim.__init__(self, calib_data, rf_dir, seed)
        self.name =f'Mixture copula test g{1-weights[0]},sc{weights[0]}-{weights[1]}'
        self.weights=weights
    
    def get_sims(self, scen_number):
        c1 = GaussCopulaNum(self.calib_data, self.rf_dir, self.seed)
        c2 = SkewCopulaNum(self.calib_data, self.rf_dir, self.seed, [self.weights[1]]*len(self.rf_dir) )
        self.sims = pd.concat(
            [c1.get_sims(int(scen_number*(1-self.weights[0]))), 
             c2.get_sims(int(scen_number*self.weights[0])),
             ], ignore_index=True)
        return self.sims