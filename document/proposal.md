---
title: "Machine Learning for Modeling High Dimensional Joint Dependency"
subtitle: "MSc Data Analytics and AI. School of Computing and Mathematical Sciences. Birkbeck, University of London"
author: "Author: Sergii Arkhypov, Supervisor: Dr Alessandro Provetti"
abstract: "The modelling of high‑dimensional joint dependence among financial risk factors remains a significant major challenge for the financial industry and Data Science. Classical statistical approaches, notably Gaussian copulas, fail to capture tail dependence adequately, leading to systematic under‑estimation of extreme portfolio losses. This MSc project aims to develop a scalable algorithm for calibrating high‑dimensional copulas that exhibit non‑Gaussian tail dependence while preserving observed marginal and correlation structures. The methodology combines semi‑analytical techniques with Machine Learning models to generate realistic joint‑distribution scenarios for up to several hundred risk factors. A proof‑of‑concept Python software implementation will be built to (i) retrieve and cache market data via the Yahoo Finance API, (ii) compute a portfolio Value‑at‑Risk using both historical and Monte‑Carlo simulations, and (iii) integrate the calibrated copulas as modular components within our new Monte‑Carlo engine. Empirical experiments will assess the impact of copula selection on Value‑at‑Risk estimates across a range of selected portfolios, followed by calibration routine to fit copula parameters to observed data. Finally, a deep‑neural‑network will be trained to learn the calibrated parameters directly from the data, and a testing suite will verify adherence to prescribed dependence and tail criteria. The expected contributions are threefold: (1) a novel high‑dimensional copula calibration framework capable of reproducing empirical tail dependence, (2) a software prototype for market‑data acquisition, risk‑metric computation, and scenario generation, and (3) an empirical evaluation of how alternative copula assumptions affect risk‑measure outcomes, thereby offering a more robust toolset for stress‑testing and capital‑planning in high‑dimensional settings."
date: "30 September 2026. Word count: 2481"
toc: true
numbersections: true
---

# Introduction
Despite the popularity of ML/AI algorithms in finance, they have mainly been used for either one or a small number of random factors. Nevertheless, joint analytics of high dimensionality (e.g., >100 random factors) still present serious challenges. This task is hard to address with classical statistical methods, and the available toolset is quite limited, i.e., mainly based on Gaussian copulas. There is hope that advances in ML/AI algorithms and corresponding software libraries will help expand this toolset and provide alternative approaches for modeling high-dimensional joint dependency.

This MSc proposal proceeds by outlining its primary goals and objectives. Subsequently, it explains the problem under investigation, along with pertinent work that provides essential concepts and a literature review. The subsequent sections detail the methodology and techniques employed to achieve the previously outlined objectives. The proposal concludes in a development plan for the project and a risk mitigation analysis.

# Aims and Objectives
The main aim of this project is to develop a Machine Learning algorithm for calibrating a high-dimensional copula with tail dependence different from Gaussian, while still reflecting the observed history with regards to selected measures (i.e., matching correlation matrix, matching Value-at-Risk measures or pairwise tail dependence at the specific percentile, etc.). On the simulation part, a range of different approaches can be considered, starting from semi-analytical methods to Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).

To create this new algorithm empirically requires to develop a software for extracting and caching market data, with subsequent incorporation of calculations of portfolio Value-at-Risk (VaR) and/or Expected Shortfall (ES) measures, which are computationally intensive on their own. To create my 'workbench' I will implement Monte Carlo simulations using various copulas within the application, in addition to providing support for the pure historical calculation methods (which will serve as an assessment baseline).

Upon completion of the initial software development, I plan to conduct an experimental analysis to evaluate the impact of copula assumptions on the selected VaR / ES measures (defined below) across distinct portfolios. Following this, establish a calibration procedure aimed at fitting appropriate copulas to real-world market data. Lastly, devise a testing exercise to verify whether the calibrated copulas align with the prescribed conditions and evaluate results against the literature baselines.

# Description of the problem and relevant work

## Key concepts

### Value-at-Risk (VaR) and expected shortfall (ES)
According to classic Jorion's work [[1]](#references), portfolio Value-at-risk (VaR) can be defined as the quantile (usually high 99%, 99.9% etc.) of the projected distribution of gains and losses over the target horizon. In simple terms it summarizes the worst loss that will not be exceeded with a given level of confidence. 
$$\text{VaR}_{\alpha}(L) = F_L^{-1}(\alpha)$$
where $\alpha$ is a confidence level and $F_L^{-1}$ is an inverted loss distribution. 

On the other hand, Expected Shortfall (ES) measures the average loss in the tail of a loss distribution. 
$$ES_{\alpha}(L) = 1 /(1-\alpha) \int_{\alpha}^{1} VaR_{p}(L) dp$$
Additional details and relevant industry discussions about ES are available in Cont et. al. [[4]](#references).

Today there are several main contesting industry spread approaches to VaR (see [[1]](#references) for additional details):

 * Historical Value-at-Risk (HVaR);
 * Monte Carlo Value-at-Risk (MC VaR) and
 * their hybrids.

When assessing Historical Value-at-Risk (HVaR), several inherent limitations are apparent, particularly when applied to high-percentile risk measures i.e. its inability to estimate extreme percentiles beyond the 99% threshold, as the method relies solely on observed historical data, which often lacks sufficient extreme observations to produce stable tail estimates (e.g. 1 year of data means approx. 250 observations, so the worst one would correspond to 99.6 % and hence being quite unstable). While HVaR does allow for filtration techniques to refine marginal distributions, this flexibility does not extend to the dependence structure, meaning it cannot effectively separate the modelling of marginals from the copula that links them. As a result, the approach struggles to capture complex joint behaviours in a structured way. Another related critical drawback is the high numerical error associated with HVaR, stemming from the finite and often sparse nature of historical data, especially in the tails. On the other hand, the relative strength of the method is its low model error (very few assumptions used), since it does not impose strong parametric assumptions but instead directly uses empirical distributions. This makes it robust in scenarios where the true data-generating process is unknown and there are no other available benchmarks, though at the cost of precision in extreme quantiles.

In contrast, Monte Carlo Value-at-Risk (MC VaR), whether based on parametric assumptions or Extreme Value Theory (EVT) see [[1]](#references), offers distinct advantages for high-percentile risk estimation. Unlike HVaR, these methods are well-suited for quantiles far beyond 99%, as they generate synthetic data to populate the tails more densely. A key structural benefit is their ability to decouple marginal distributions from the copula, allowing for independent calibration of each component. This separation enhances flexibility in modelling dependence structure while maintaining control over individual risk factors.

From a computational perspective, MC VaR methods exhibit low numerical error, as the simulated datasets can be made arbitrarily large to smooth out estimation noise. However, this precision comes at the expense of higher model error, particularly when parametric assumptions are misspecified or when EVT extrapolations deviate from true tail behavior. While these methods provide greater control and extensibility in extreme risk modelling, their reliability hinges on the accuracy of the underlying assumptions, whether in the choice of distributions copula functions, or tail decay parameters.

 
### Copulas. 
In this section we briefly introduce the concept of copulas and provide additional references, followed by the impacts it could have on VaR/ES measures.

$C: [0,1]^d -> [0,1]$ is a d-dimentional copula if $C$ is a joint cumulative distribution function of a d-dimentional random vector on the unit cube $[0,1]^d$ with uniform marginals.

Sklar's theorem. Every multivariate cumulative distribution function $H(x_1, ... x_d) = Pr[X_1 \leq x_1, ... X_d \leq x_d]$ of a random $(X_1, X_2, ... , X_d )$ can be expressed in terms of its marginals $F_i(x_i)=Pr[X_i \leq x_i]$ and a copula $C$, as:
[$$H(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d))$$]

For more details on copula and their applications in finance please see [[2]](#references). However it is important to introduce several properties of the copulas which will be significant for the current project:

 * A convex combination of copulas is also a copula. It is often called a mixture copula and can be expressed as a weighted sum of copulas $C_i$, defined as $C_{mix} = \Sigma_{i=1}^{n} w_i C_i(u,v)$, where all $w_i>0$ and sum to one $\Sigma_{i=1}^{n} w_i = 1$.

 * Exists a set of functions (W-transforms) applied to copula and also returning copula but with modified properties. For additional details on these transformations and their properties please see [[9]](#references).


### Copulas in Value-at-Risk
The selection of a copula has a significant influence on the assessment of tail‑risk measures such as Value‑at‑Risk (VaR) / Expected Shortfall (ES). This influence becomes increasingly pronounced as we move to more extreme quantiles (e.g. 99 % and 99.9 % VaR). The principal limitation of the Gaussian copula lies in its asymptotic independence: as one approach the far ends of the distribution, the probability that two (or more) variables exceed a high threshold simultaneously tends to zero at the same rate as if the variables were independent. Consequently, the Gaussian copula systematically under‑estimates the likelihood of joint extreme events and, therefore, under‑states tail‑risk metrics.

Tail dependence provides a concise way to expose this shortcoming, some introduction to the use of tail dependence can be found in [[6]](#references). Formulas below define lower and upper tail dependence respectively: 

$$\tau^L_{ij} = \lim_{q \to 0^+} P[X_i \leq G_i^{-1}(q) \mid X_j \leq G_j^{-1}(q)]/(1-q)$$

$$\tau^U_{ij} = \lim_{q \to 1^-} P[X_i > G_i^{-1}(q) \mid X_j > G_j^{-1}(q)]/q$$

By computing these tail‑dependence coefficients, one can directly compare how different copulas treat extreme co‑movements. The contrast is stark: while a Gaussian copula predicts virtually no joint tail events, for example Cauchy (Student-T with zero degrees of freedom) assigns a non‑negligible probability to simultaneous extreme losses. See [Appendix A](#copula-examples) for the visual comparison between two copulas both reflecting equivalent correlation values. This disparity translates into markedly different VaR and ES estimates, especially at the highest percentiles. Consequently, when modelling portfolio risk that is sensitive to rare but severe events, choosing a copula with appropriate tail dependence is essential. Otherwise, VaR and ES risk measures may be substantially understated.

Figures below compare historically observed tail dependence (the orange line) with the one of Gaussian and Cauchy copulas i.e. Student-T with zero degrees of freedom (the blue and green lines respectively). The vertical axis measures tail dependence, while the horizontal axis displays the chosen percentile. As the graphs reveal, the discrepancy between the observed and Gaussian tail dependence expands significantly when more risk factors considered together.

![Figure 1: Tail dependence for different types of copulas and different number of random variables](figures\Tail_dependance.png)

The Cauchy copula is a statistical technique for capturing dependence structures, especially when modelling tail‑risk or extreme events. Its principal advantage lies in preserving the overall correlation matrix of the underlying variables while simultaneously enabling the generation of scenarios that exhibit much heavier tails (joint‑movement) and de-correlation. In practice, this means that the copula retains the familiar linear dependence captured by Pearson‑type correlations, yet it can produce joint realizations that reflect the heightened co‑movement observed during rare, high‑impact shocks.

De‑correlation scenarios describe circumstances in which variables that normally exhibit a strong positive (or negative) relationship suddenly lose that linkage when stress or extreme events materializes. In other words, the usual together‑ness of the data breaks down. A classic illustration comes from equity markets: two stocks that historically track each other e.g. because they belong to the same industry, may abruptly move in opposite directions during a stress event. For example, Apple vs. Microsoft  - Apple taking over Microsoft business after Microsoft collapses.

On the other hand, joint‑movement scenarios involve a set of variables that, despite exhibiting only modest or even negligible correlation under normal conditions, all swing in the same direction because they are all exposed to a shared extreme driver. Think of a severe natural disaster that simultaneously spikes commodity prices or a sovereign default that forces both sovereign bond yields and the domestic currency to deteriorate together. In these cases the common shock creates a temporary, high‑intensity dependence that standard correlation estimates fail to capture.

Both types of scenarios are crucial for robust risk assessment. De‑correlation highlights the risk of assets that are supposed to hedge each other stop doing so, while joint‑movement emphasizes the danger of hidden tail‑dependence that can generate simultaneous losses across seemingly unrelated positions. Modelling frameworks that can generate both behaviours, provide a more realistic picture of potential extreme outcomes than approaches that rely solely on historic linear correlations.


## Literature review
The rapid advancement of generative machine learning techniques has generated significant interest within the financial industry, as evidenced by numerous studies (e.g., see [[8]](#references) for a review). However, much of the existing research has primarily focused on generating paths for either a single asset or a limited number of risk factors (as discussed in [[7]](#references)). Moreover, the majority of these efforts have cantered on simulating so-called "central scenarios," with limited attention given to the generation of realistic extreme scenarios. An exception to this trend can be found in [[5]](#references), which explicitly target the generation of tail events using Generative Adversarial Network (GAN) (see [[5]](#references) for examples and implementation) architectures, though even these approaches remain constrained to relatively low-dimensional environments (e.g., up to 20–50 factors).

Inspired by these works, this research seeks to extend the dimensionality of scenario generation while relaxing the requirement for path dependence. This adjustment allows us to leverage more classical tools for modelling joint dependencies, such as copulas. However, even within this framework, only Gaussian copulas (and selected Student-T) have demonstrated the capacity to scale effectively to very high dimensions. This project aims to bridge the gap between the data-driven GAN-like approaches and the scalability of copulas by developing a high-dimensional copula model capable of capturing historical tail dependence properties. The ultimate goal is to generate realistic extreme scenarios that reflect the complex dependencies observed in financial markets.


# Methodology and methods
As mentioned above unlike studies where the intricacies of underlying asset path dynamics (such as time-dependent volatility or complex stochastic processes) are critical, this work deliberately abstracts from such details. As such there is no need to include dynamically changing portfolios (i.e. dynamic trading strategies), limiting portfolio dimension optimization to static. Portfolios of approx. 500 names with all long positions and 500 long/short hedged positions will be used in this study.

Figure below presents overall workflow of the project aiming at calibrating copula to historical timeseries related to the list of selected portfolios.

![Figure 2: Copula calibration workflow](figures\Project-workflow-high.png)

It is further detailed in the next figure, which illustrates the cumulative process of constructing different time series based on the observed history and representing required data transformations.

![Figure 3: Detailed process of data transformations for copula calibration](figures\Project-workflow-detailed.png)

Beyond scalability, another key advantage lies in the interpretability of results. While alternative approaches, such as Tail-GAN-based simulations, often produce outputs that are difficult to dissect or explain, the methodology employed here offers an inherent structure for understanding extreme scenarios. The framework decomposes market dynamics into a weighted combination of copulas (see [Copulas](#copulas) properties in the previous section), each with a clear and intuitive economic or statistical meaning. This not only enhances transparency but also provides with a more actionable foundation for stress testing and capital planning. The ability to attribute risk contributions to specific copula components is a significant improvement in practical applicability and model governance.

Please also note that aim of this software (Python) is a proof of concept rather than an end-user application, while it still follows the main principals of software design with the development plan outlined below.


# Project development plan and risk mitigation analysis
The work is expected to start in the beginning of May 2026. Development and research phase is expected to be finalized within 3 months leaving additional time for the report preparation. 

* **Step 1 [week 1-2]:** Develop functionality (Python) able to discover required market data and cache it from Yahoo Finance API according to the required population, length and granularity.
* **Step 2 [week 3-4]:** Develop functionality able to calculate different Value-at-Risk metrics based on specified methodology i.e. Historical VaR, Monte-Carlo VaR. The latter should be structured in a way that copula class is provided as an input, allowing calculations with different copula hypothesis.
* **Step 3 [week 5-6]:** Perform initial experiments to gauge copula impact on selected measures across different portfolios. This should help to develop better understanding required for the future interpretation of the results and testing. Assessing Gaussian, Cauchy, their transformations and weighted combinations.
* **Step 4 [week 7-8]:** Build optimization functionality select parameters for the copula matching current measures for the list of selected portfolios.    
* **Step 5 [week 9-10]:** Design a testing exercise to ensure the calibrated copulas adhere to the specified conditions.
* **Step 6 [week 11 +]:** Assess obtained results, summarize observations and developed software within the report.

There is a distinct possibility that certain project phases may overrun, particularly code development, testing, and the selection of the copula calibration algorithm. To mitigate this, it is aimed to complete the core work by August 2026, providing a buffer to accommodate potential delays.

# References
1. Jorion P., 2006. Value as Risk: The New Benchmark for Managing Financial Risk. New York: The McGraw-Hill.
2. Cherubini U. et al, 2004. Copula methods in finance. England: Wiley finance.
3. Chollet F., 2021. Deep learning with Python. New York: Manning Publications Co. 

4. Acerbi C., Szekely B., 2014. Back-testing expected shortfall. Risk, 27(11):76–81.
5. Cont R. et al, 2022. Tail-GAN: Nonparametric scenario generation for tail risk estimation. [(link)](https://arxiv.org/abs/2203.01664)
6. Oh D., Patton A., 2015. Modelling dependance in High dimensions with factor copulas. [(link)](https://www.federalreserve.gov/econresdata/feds/2015/files/2015051pap.pdf)
7. Buhler H., Horvath B., 2021. A data-driven market simulator for small data environments. [(link)](https://arxiv.org/abs/2006.14498)
8. Horvath B. et al, 2025. Generative Models in Finance: Market Generators, a Paradigm Shift in Financial Modeling.[(link)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5284313)

9. Hofert M. and Pang Z., 2025. W-transforms: Uniformity-preserving transformations and induced dependence structures.[(link)](https://arxiv.org/pdf/2509.26280)

# Appendixes

## Appendix A: List of used AI tools

 * Gemini via Google Search;
 * Mistral-7B-Instruct-v0.3 for polishing the text. Prompt: *"You're an academic assistant with an expertise in software engineering to proofread and rewrite as a single connected text. Keep academic tone."*

## Appendix B: Copula examples
Gaussian copula.

![Figure 1: Uniform scenarios Gaussian_copula, numerical simulation](figures\Uniform_scenarios_Gaussian_copula_(numerical_simulation).png)

![Figure 2: Contour plot Gaussian copula, numerical simulation](figures\Contour_plot_Gaussian_copula_(numerical_simulation).png)

Cauchy copula.

![Figure 3: Uniform scenarios Cauchy copula, numerical simulation](figures\Uniform_scenarios_Cauchy_copula_(numerical_simulation).png)

![Figure 4: Contour plot Cauchy copula, numerical simulation](figures\Contour_plot_Cauchy_copula_(numerical_simulation).png)




