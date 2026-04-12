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

This academic proposal proceeds by outlining its primary goals and objectives. Subsequently, it explains the problem under investigation, along with pertinent work that provides essential concepts and a literature review. The subsequent sections detail the methodology and techniques employed to achieve the previously outlined objectives. The proposal concludes in a development plan for the project.

# Aims and Objectives
The main aim of this work would be to develop a machine learning algorithm for calibrating a high-dimensional copula with tail dependence different from Gaussian, while still reflecting the observed history with regards to selected measures (i.e., matching correlation matrix, matching pairwise tail dependence at the specific percentile, etc.). The latter is expected to require the creation of a custom loss function to achieve minimization with regards to those measures. On the simulation part, a range of different approaches can be considered, starting from semi-analytical methods to Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).

Develop a software application designed for extracting and caching market data, with subsequent incorporation of calculations of portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) measures. To achieve this, implement Monte Carlo simulations using various copulas within the application, in addition to providing support for pure historical calculation methods.

Upon completion of the initial software development, conduct an experimental analysis to evaluate the impact of copula assumptions on the selected VaR and ES measures across distinct portfolios. Following this, establish a calibration procedure aimed at fitting appropriate copulas to real-world market data. Lastly, devise a testing exercise to verify whether the calibrated copulas align with the prescribed conditions.

In summary:

1. Create software for market data extraction and caching.
2. Implement VaR/ES calculations using Monte Carlo simulations (with various copulas) and historical methods.
3. Perform an initial experiment to gauge copula impact on selected measures across different portfolios.
4. Define a machine calibration procedure tailored towards fitting suitable copulas to real-world market data.
5. Design a testing exercise to ensure the calibrated copulas adhere to the specified conditions.


# Description of the problem and relevant work

## Key concepts

### Value-at-risk (VaR) and expected shortfall (ES)
Based on [[1]](#literature), portfolio Value-at-risk (VaR) can be defined as the quantile (usually high 99%, 99.9% etc.) of the projected distribution of gains and loses over the target horizon. In simple terms it summarizes the worst loss that will not be exceeded with a given level of confidence. 
$$\text{VaR}_{\alpha}(L) = F_L^{-1}(\alpha)$$
where $\alpha$ is a confidence level and $F_L^{-1}$ is an inverted loss distribution. 

On the other hand, Expected Shortfall (ES) measures the average loss in the tail of a loss distribution. 
$$ES_{\alpha}(L) = 1 /(1-\alpha) \int_{\alpha}^{1} VaR_{p}(L) dp$$
Additional details and relevant industry discussion about ES are available in [[4]](#literature).



 
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




## Literature review




# Methodology and methods
*provide a diagram that illustrate your approach/data pipeline*
*for each objective you can provide methodology, methods and tools that you will use, and organise this part accordingly*
*difference between an evaluation with respect to how your software performs, versus a critical evaluation where you reflect on your aims and objectives*
*how do you plan to implement it? E.g. if you are using machine learning, you will require a model (explain the structure of the neural network)*


# Project development plan
*you have to convince the marker and the supervisor that you are able to manage your time and juggle two separate concerns: software artefacts (software app, ML models) and final report. Develop a realistic workplan for the project.*


# Literature
1. Jorion P., 2006. Value as Risk: The New Benchmark for Managing Financial Risk. New York: The McGraw-Hill.
2. Cherubini U. et al, 2004. Copula methods in finance. England: Wiley finance.
3. Chollet F., 2021. Deep learning with Python. New York: Manning Publications Co. 

4. Acerbi C., Szekely B., 2014. Back-testing expected shortfall. Risk, 27(11):76–81.
5. Cont R. et al, 2022. Tail-GAN: Nonparametric scenario generation for tail risk estimation. [(link)](https://arxiv.org/abs/2203.01664)
6. Oh D., Patton A., 2015. Modelling dependance in High dimensions with factor copulas. [(link)](https://www.federalreserve.gov/econresdata/feds/2015/files/2015051pap.pdf)
7. Buhler H., Horvath B., 2021. A data-driven market simulator for small data environments. [(link)](https://arxiv.org/abs/2006.14498)
8. Horvath B. et al, 2025. Generative Models in Finance: Market Generators, a Paradigm Shift in Financial Modeling.[(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5284313)


# Appendixes
See corresponding equation [(Eq.1)](#eq1)

## Appendix A: List of used AI tools

 * Gemini via Google Search;
 * Mistral-7B-Instruct-v0.3 for polishing the text. Prompt: *"You're an academic assistant with an expertise in software engineering to proofread and rewrite as a single connected text. Keep academic tone."*

