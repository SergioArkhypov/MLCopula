---
title: Machine Learning for Modeling High Dimensional Joint Dependency
author: Sergii Arkhypov
date: 30 September 2026
---

# Abstract


# Kernel
Despite the popularity of ML/AI algorithms in finance, they have mainly been used for either one or a small number of random factors. Nevertheless, joint analytics of high dimensionality (e.g., >1000 random factors) present serious challenges. This task is hard to address with classical statistical methods, and the available toolset is quite limited, i.e., mainly based on Gaussian copulas. There is hope that modern advances in ML/AI algorithms could help expand this toolset and provide alternative approaches for modeling high-dimensional joint dependency.

The main aim of this work would be to develop a machine learning algorithm for calibrating a high-dimensional copula with tail dependence different from Gaussian, while still reflecting the observed history with regards to selected measures (i.e., matching correlation matrix, matching pairwise tail dependence at the specific percentile, etc.). The latter is expected to require the creation of a custom loss function to achieve minimization with regards to those measures. On the simulation part, a range of different approaches can be considered, starting from semi-analytical methods to Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).


# ... theory
## Copulas
$C: [0,1]^d -> [0,1]$ is a d-dimentional copula if $C$ is a joint cumulative distribution function of a d-dimentional random vector on the unit cube $[0,1]^d with uniform marginals$

## Sklar's theorem
Every multivariate cumulative distribution function $H(x_1, ... x_d) = Pr[X_1 \leq x_1, ... X_d \leq x_d]$ of a random $(X_1, X_2, ... , X_d )$ can be expressed in terms of its marginals $F_i(x_i)=Pr[X_i \leq x_i]$ and a copula $C$, as:
[$$H(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d)) \;\;\; \ (Eq.1)$$]{id="eq1"}

## Gaussian copula
![Figure 1: Uniform scenarios Gaussian_copula, numerical simulation](figures\Uniform_scenarios_Gaussian_copula_(numerical_simulation).png)

![Figure 2: Contour plot Gaussian copula, numerical simulation](figures\Contour_plot_Gaussian_copula_(numerical_simulation).png)


## Cauchy copula 
![Figure 3: Uniform scenarios Cauchy copula, numerical simulation](figures\Uniform_scenarios_Cauchy_copula_(numerical_simulation).png)

![Figure 4: Contour plot Cauchy copula, numerical simulation](figures\Contour_plot_Cauchy_copula_(numerical_simulation).png)

# Literature
1. D.Oh, A.Patton (2015). Modelling dependance in High dimensions with factor copulas. [(link)](https://www.federalreserve.gov/econresdata/feds/2015/files/2015051pap.pdf)
2. R.Cont et al. (2022). Tail-GAN: Nonparametric scenario generation for tail risk estimation. [(link)](https://arxiv.org/abs/2203.01664)
3. H.Buhler, B.Horvath (2021). A data-driven market simulator for small data environments. [(link)](https://arxiv.org/abs/2006.14498)
4. M.Vuletic, R.Cont (2025). VolGAN: a generative model for arbitrage-free implied volatility surfaces.[(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4617536)
5. B.Horvath et al. (2025). Generative Models in Finance: Market Generators, a Paradigm Shift in Financial Modeling.[(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5284313)
6. M.Vuletic, M.Cucuringu (2025). GraFiN-Gen: graph-based ensemble generative modelling for multi-asset forecasting. [(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5317725)
7. A.Sancetta, S.Satchell (2004). The Bernstein Copula and Its Applications to Modeling and Approximations of Multivariate Distributions. [(link)](https://www.jstor.org/stable/pdf/3533531.pdf)

# Appendix
See corresponding equation [(Eq.1)](#eq1)

