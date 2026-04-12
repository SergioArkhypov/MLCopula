---
title: "Machine Learning for Modeling High Dimensional Joint Dependency"
subtitle: "MSc Data Analytics and AI. School of Computing and Mathematical Sciences. Birkbeck, University of London"
author: "Author: Sergii Arkhypov, Supervisor: Dr Alessandro Provetti"
abstract: xxx to populate xxx
date: "30 September 2026. Word count: XXX"
toc: true
numbersections: true
---

# Introduction
Despite the popularity of ML/AI algorithms in finance, they have mainly been used for either one or a small number of random factors. Nevertheless, joint analytics of high dimensionality (e.g., >1000 random factors) present serious challenges. This task is hard to address with classical statistical methods, and the available toolset is quite limited, i.e., mainly based on Gaussian copulas. There is hope that modern advances in ML/AI algorithms could help expand this toolset and provide alternative approaches for modeling high-dimensional joint dependency.

This proposal is orgganised as following:


# Aims and Objectives
The main aim of this work would be to develop a machine learning algorithm for calibrating a high-dimensional copula with tail dependence different from Gaussian, while still reflecting the observed history with regards to selected measures (i.e., matching correlation matrix, matching pairwise tail dependence at the specific percentile, etc.). The latter is expected to require the creation of a custom loss function to achieve minimization with regards to those measures. On the simulation part, a range of different approaches can be considered, starting from semi-analytical methods to Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).

Create a software aaplication able to extract market data

*Provide a list of objectives. These are the steps that lead to achieving your aim. No more that 6 objectives usually in a student project.*


# Description of the problem and relevant work

## Key concepts
 
### Copulas. 
$C: [0,1]^d -> [0,1]$ is a d-dimentional copula if $C$ is a joint cumulative distribution function of a d-dimentional random vector on the unit cube $[0,1]^d$ with uniform marginals.

Sklar's theorem. Every multivariate cumulative distribution function $H(x_1, ... x_d) = Pr[X_1 \leq x_1, ... X_d \leq x_d]$ of a random $(X_1, X_2, ... , X_d )$ can be expressed in terms of its marginals $F_i(x_i)=Pr[X_i \leq x_i]$ and a copula $C$, as:
[$$H(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d)) \;\;\; \ (Eq.1)$$]{id="eq1"}


Gaussian copula.

![Figure 1: Uniform scenarios Gaussian_copula, numerical simulation](figures\Uniform_scenarios_Gaussian_copula_(numerical_simulation).png)

![Figure 2: Contour plot Gaussian copula, numerical simulation](figures\Contour_plot_Gaussian_copula_(numerical_simulation).png)


Cauchy copula.

![Figure 3: Uniform scenarios Cauchy copula, numerical simulation](figures\Uniform_scenarios_Cauchy_copula_(numerical_simulation).png)

![Figure 4: Contour plot Cauchy copula, numerical simulation](figures\Contour_plot_Cauchy_copula_(numerical_simulation).png)

### Value-at-risk and expected shortfall



## Literature review





# Methodology and methods
*provide a diagram that illustrate your approach/data pipeline*
*for each objective you can provide methodology, methods and tools that you will use, and organise this part accordingly*
*difference between an evaluation with respect to how your software performs, versus a critical evaluation where you reflect on your aims and objectives*
*how do you plan to implement it? E.g. if you are using machine learning, you will require a model (explain the structure of the neural network)*

# Organise and plan the project
*you have to convince the marker and the supervisor that you are able to manage your time and juggle two separate concerns: software artefacts (software app, ML models) and final report. Develop a realistic workplan for the project.*


# Literature
1. D.Oh, A.Patton (2015). Modelling dependance in High dimensions with factor copulas. [(link)](https://www.federalreserve.gov/econresdata/feds/2015/files/2015051pap.pdf)
2. R.Cont et al. (2022). Tail-GAN: Nonparametric scenario generation for tail risk estimation. [(link)](https://arxiv.org/abs/2203.01664)
3. H.Buhler, B.Horvath (2021). A data-driven market simulator for small data environments. [(link)](https://arxiv.org/abs/2006.14498)
4. M.Vuletic, R.Cont (2025). VolGAN: a generative model for arbitrage-free implied volatility surfaces.[(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4617536)
5. B.Horvath et al. (2025). Generative Models in Finance: Market Generators, a Paradigm Shift in Financial Modeling.[(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5284313)
6. M.Vuletic, M.Cucuringu (2025). GraFiN-Gen: graph-based ensemble generative modelling for multi-asset forecasting. [(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5317725)
7. A.Sancetta, S.Satchell (2004). The Bernstein Copula and Its Applications to Modeling and Approximations of Multivariate Distributions. [(link)](https://www.jstor.org/stable/pdf/3533531.pdf)
8. M.Wiese et al. Multi-Asset Spot and Option Market Simulation. [(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3980817)

# Appendixes
See corresponding equation [(Eq.1)](#eq1)

## Appendix A: List of used AI tools

 * Gemini via Google Search;
 * Mistral-7B-Instruct-v0.3 for polishing the text. Prompt: *"You're an academic assistant with an expertise in software engineering to proofread and rewrite as a single connected text. Keep academic tone."*

