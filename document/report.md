---
title: "Machine Learning for Modelling High Dimensional Joint Dependency"
subtitle: "MSc Data Analytics and AI. School of Computing and Mathematical Sciences. Birkbeck, University of London"
author: "Author: Sergii Arkhypov, Supervisor: Dr Alessandro Provetti"
abstract: "The modelling of high‑dimensional joint dependence among financial risk factors remains a significant major challenge for the financial industry and Data Science. Classical statistical approaches, notably Gaussian copulas, fail to capture tail dependence adequately, leading to systematic under‑estimation of extreme portfolio losses. This MSc project aims to develop a scalable algorithm for calibrating high‑dimensional copulas that exhibit non‑Gaussian tail dependence while preserving observed marginal and correlation structures. The methodology combines semi‑analytical techniques with Machine Learning models to generate realistic joint‑distribution scenarios for up to several hundred risk factors. A proof‑of‑concept Python software implementation will be built to (i) retrieve and cache market data via the Yahoo Finance API, (ii) compute a portfolio Value‑at‑Risk using both historical and Monte‑Carlo simulations, and (iii) integrate the calibrated copulas as modular components within our new Monte‑Carlo engine. Empirical experiments will assess the impact of copula selection on Value‑at‑Risk estimates across a range of selected portfolios, followed by calibration routine to fit copula parameters to observed data. Finally, a deep‑neural‑network will be trained to learn the calibrated parameters directly from the data, and a testing suite will verify adherence to prescribed dependence and tail criteria. The expected contributions are threefold: (1) a novel high‑dimensional copula calibration framework capable of reproducing empirical tail dependence, (2) a software prototype for market‑data acquisition, risk‑metric computation, and scenario generation, and (3) an empirical evaluation of how alternative copula assumptions affect risk‑measure outcomes, thereby offering a more robust toolset for stress‑testing and capital‑planning in high‑dimensional settings."
date: "10 September 2026. Word count: XXX"
toc: true
numbersections: true
---

# Introduction
Despite the popularity of ML/AI algorithms in finance, they have mainly been used for either one or a small number of random factors. Nevertheless, joint analytics of high dimensionality (e.g., >100 random factors) still present serious challenges. This task is hard to address with classical statistical methods, and the available toolset is quite limited, i.e., mainly based on Gaussian copulas. There is hope that advances in ML/AI algorithms and corresponding software libraries will help expand this toolset and provide alternative approaches for modelling high-dimensional joint dependency.

This MSc project proceeds by outlining its primary goals and objectives. Subsequently, it explains the problem under investigation, along with pertinent work that provides essential concepts and a literature review. The subsequent sections detail the methodology and techniques employed to achieve the previously outlined objectives.The subsequent part of the document will explain the project design and implementation, providing rationales for the chosen code architecture. Following this, a section dedicated to testing and evaluation will be presented. It is followed by an analysis of the results, focusing on their implications, and concluding with suggestions for future research direction.


# Aims and Objectives

The main aim of this project is to develop a Machine Learning algorithm for calibrating a high-dimensional copula with tail dependence different from Gaussian, while still reflecting the observed history with regards to selected measures (i.e., matching correlation matrix, matching Value-at-Risk measures or pairwise tail dependence at the specific percentile, etc.). On the simulation part, a range of different approaches can be considered, starting from semi-analytical methods to Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).

To create this new algorithm empirically requires to develop a software for extracting and caching market data, with subsequent incorporation of calculations of portfolio Value-at-Risk (VaR) and/or Expected Shortfall (ES) measures, which are computationally intensive on their own. To create my 'workbench' I will implement Monte Carlo simulations using various copulas within the application, in addition to providing support for the pure historical calculation methods (which will serve as an assessment baseline).

Upon completion of the initial software development, I plan to conduct an experimental analysis to evaluate the impact of copula assumptions on the selected VaR / ES measures (defined below) across distinct portfolios. Following this, establish a calibration procedure aimed at fitting appropriate copulas to real-world market data. Lastly, devise a testing exercise to verify whether the calibrated copulas align with the prescribed conditions and evaluate results against the literature baselines.

# Description of the problem and relevant work

## Key concepts

### Value-at-Risk (VaR) and expected shortfall (ES)
According to classic Jorion's work [[1]](#references), portfolio Value-at-risk (VaR) can be defined as the quantile (usually high 99%, 99.9% etc.) of the projected distribution of gains and losses over the target horizon. In simple terms it summarizes the worst loss that will not be exceeded with a given level of confidence. 
$$\mathrm{VaR}_{\alpha}(L) = F_L^{-1}(\alpha)$$
where $\alpha$ is a confidence level and $F_L^{-1}$ is an inverted loss distribution. 

On the other hand, Expected Shortfall (ES) measures the average loss in the tail of a loss distribution:

$$ES_{\alpha}(L) = \frac{1}{1-\alpha} \int_{\alpha}^{1} VaR_{p}(L) dp.$$

Additional details and relevant industry discussions about ES are available in Cont et. al. [[4]](#references).

Today there are several main contesting industry spread approaches to VaR (see [[1]](#references) for additional details):

 * Historical Value-at-Risk (HVaR);
 * Monte Carlo Value-at-Risk (MC VaR) and
 * their hybrids.

When assessing Historical Value-at-Risk (HVaR), several inherent limitations are apparent, particularly when applied to high-percentile risk measures i.e. its inability to estimate extreme percentiles beyond the 99% threshold, as the method relies solely on observed historical data, which often lacks sufficient extreme observations to produce stable tail estimates (e.g. 1 year of data means approx. 250 observations, so the worst one would correspond to 99.6 % and hence being quite unstable). 
While HVaR does allow for filtration techniques to refine marginal distributions, this flexibility does not extend to the dependence structure, meaning it cannot effectively separate the modelling of marginals from the copula that links them. 
As a result, the approach struggles to capture complex joint behaviours in a structured way. 
Another related critical drawback is the high numerical error associated with HVaR, stemming from the finite and often sparse nature of historical data, especially in the tails. 
On the other hand, the relative strength of the method is its low model error (very few assumptions used), since it does not impose strong parametric assumptions but instead directly uses empirical distributions. 
This makes it robust in scenarios where the true data-generating process is unknown and there are no other available benchmarks, though at the cost of precision in extreme quantiles.

In contrast, Monte Carlo Value-at-Risk (MC VaR), whether based on parametric assumptions or Extreme Value Theory (EVT) see [[1]](#references), offers distinct advantages for high-percentile risk estimation. 
Unlike HVaR, these methods are well-suited for quantiles far beyond 99%, as they generate synthetic data to populate the tails more densely. A key structural benefit is their ability to decouple marginal distributions from the copula, allowing for independent calibration of each component. 
This separation enhances flexibility in modelling dependence structure while maintaining control over individual risk factors.

From a computational perspective, MC VaR methods exhibit low numerical error, as the simulated datasets can be made arbitrarily large to smooth out estimation noise. 
However, this precision comes at the expense of higher model error, particularly when parametric assumptions are mis specified or when EVT extrapolations deviate from true tail behaviour. 
While these methods provide greater control and extensibility in extreme-risk modelling, their reliability hinges on the accuracy of the underlying assumptions, whether in the choice of distributions, copula functions, or tail decay parameters.

 
### Copulas
In this section we briefly introduce the concept of copulas and provide additional references, followed by a discussion of the impact it has on VaR/ES measures.

$C: [0,1]^d -> [0,1]$ is a d-dimentional copula if $C$ is a joint cumulative distribution function of a d-dimentional random vector on the unit cube $[0,1]^d$ with marginals (i.e. each dimension) following a uniform distribution.

Sklar's theorem (see also in [[2]](#references)). Every multivariate cumulative distribution function $H(x_1, \dots, x_d) = Pr[X_1 \leq x_1, \dots, X_d \leq x_d]$ of a random variable $(X_1, X_2, \dots, X_d )$ can be expressed in terms of its marginals $F_i(x_i)=Pr[X_i \leq x_i]$ and a copula $C$, as:

$$H(x_1, \dots, x_d) = C(F_1(x_1), \dots, F_d(x_d)),$$

For more details on copula and their applications in finance please see [[2]](#references). However it is important to introduce several properties of copulas which will be significant for the current project:

 * A convex combination (i.e. all weights are non-negative and sum to exactly one) of copulas is also a copula. It is often called a mixture copula and can be expressed as a weighted sum of copulas $C_i$, defined as $C_{mix} = \Sigma_{i=1}^{n} w_i C_i(u,v)$, where all $w_i>0$ and sum to one $\Sigma_{i=1}^{n} w_i = 1.$

 * There exists a class of functions (W-transforms) applied to copula and also returning copula but with modified properties.

 * For additional details on these transformations and their properties please see [[9]](#references).

One of the most widely used copula is the one constructed from a multivariate normal distribution by using the probability integral transform, Gaussian copula, see [[2]](#references) for additional details.

### Copulas in Value-at-Risk
The selection of a copula has a significant influence on the assessment of tail‑risk measures such as Value‑at‑Risk (VaR) / Expected Shortfall (ES). 
This influence becomes increasingly pronounced as we move to more extreme quantiles (e.g. 99 % and 99.9 % VaR). 
The principal limitation of the Gaussian copula lies in its asymptotic independence: as one approach the far ends of the distribution, the probability that two (or more) variables exceed a high threshold simultaneously tends to zero at the same rate as if the variables were independent. 
Consequently, the Gaussian Copula systematically under‑estimates the likelihood of joint extreme events and, therefore, under‑states tail‑risk metrics.

Tail dependence provides a concise way to expose this shortcoming, some introduction to the use of tail dependence can be found in [[6]](#references). Formulas below define lower and upper tail dependence respectively: 

$$\tau^L_{ij} = \lim_{q \to 0^+} P[X_i \leq G_i^{-1}(q) \mid X_j \leq G_j^{-1}(q)]/(1-q),$$

$$\tau^U_{ij} = \lim_{q \to 1^-} P[X_i > G_i^{-1}(q) \mid X_j > G_j^{-1}(q)]/q,$$

where $G_i, G_j$ are marginal CDFs, while $G_i^{-1}, G_j^{-1}$ are inverse CDFs respectively.

High levels of tail dependence essentially means that the extremes of $X_i$ and $X_j$ occur more in unison. By computing these tail‑dependence coefficients, one can directly compare how different copulas treat extreme co‑movements. 
The contrast is stark: while a Gaussian copula predicts virtually no joint tail events, for example Cauchy (Student-T with zero degrees of freedom) assigns a non‑negligible probability to simultaneous extreme losses. 
See [Appendix A](#copula-examples) for the visual comparison between two copulas both reflecting equivalent correlation values. 
This disparity translates into markedly different VaR and ES estimates, especially at the highest percentiles. 
Consequently, when modelling portfolio risk that is sensitive to rare but severe events, choosing a copula with appropriate tail dependence is essential. 
Otherwise, VaR and ES risk measures may be substantially understated.

Figure 1 below compares historically observed tail dependence (the orange line) with the one of Gaussian and Cauchy copulas i.e. Student-T with zero degrees of freedom (the blue and green lines respectively). 
The vertical axis measures tail dependence, while the horizontal axis displays the chosen percentile. 
As the graphs reveal, the discrepancy between the observed and Gaussian tail dependence expands significantly when more risk factors considered together. It is based on 20Y of market price historical observations data (2002-2022) for 2 names Microsoft and Morgan Stanley and 5 names with additionally Google, Amazon and JP Morgan. 

![*Figure 1: Tail dependence for different types of copulas and different number of random variables.*](figures\Tail_dependance.png)

### Economics of the Cauchy copula 

The Cauchy copula is a statistical technique for capturing dependence structures, especially when modelling tail‑risk or extreme events. 
Its principal advantage lies in preserving the overall correlation matrix of the underlying variables while simultaneously enabling the generation of scenarios that exhibit much heavier tails (joint‑movement) and de-correlation. 
In practice, this means that the copula retains the familiar linear dependence captured by Pearson‑type correlations, yet it can produce joint realizations that reflect the heightened co‑movement observed during rare, high‑impact shocks.

De‑correlation scenarios describe circumstances in which variables that normally exhibit a strong positive (or negative) relationship suddenly lose that linkage when stress or extreme events materializes. 
In other words, the usual together‑ness of the data breaks down. A classic illustration comes from equity markets: two stocks that historically track each other, e.g., because they belong to the same industry, may abruptly move in opposite directions during a stress event. For example, Google vs. Microsoft  - Google taking over Microsoft business after Microsoft collapses.

On the other hand, joint‑movement scenarios involve a set of variables that, despite exhibiting only modest or even negligible correlation under normal conditions, all swing in the same direction because they are all exposed to a common extreme driver. 
Examples are severe natural disasters that simultaneously spike commodity prices, or a sovereign default that forces both sovereign bond yields and the domestic currency to deteriorate together. 
In these cases, the common shock creates a temporary, high‑intensity dependence that standard correlation estimates fail to capture.

Both types of scenarios are crucial for robust risk assessment.
De‑correlation highlights the risk of assets that are supposed to hedge each other stop doing so, while joint‑movement emphasizes the danger of hidden tail‑dependence that can generate simultaneous losses across seemingly unrelated positions. 
Modelling frameworks that can generate both behaviours would provide a more realistic picture of potential extreme outcomes than approaches that rely solely on historic linear correlations.


## Literature review

The rapid advancement of generative Machine Learning techniques has generated significant interest within the financial industry, as evidenced by numerous studies (e.g., see [[8]](#references) for a review). However, much of the existing research has primarily focused on generating paths for either a single asset or a limited number of risk factors (as discussed in [[7]](#references)). Moreover, the majority of these efforts have cantered on simulating so-called "central scenarios," with limited attention given to the generation of realistic extreme scenarios. An exception to this trend can be found in [[5]](#references), which explicitly target the generation of tail events using Generative Adversarial Network (GAN) (see Cont R. et al for examples and implementation) architectures, though even these approaches remain constrained to relatively low-dimensional environments, i.e., up to 20–50 factors.

Inspired by these works, this research seeks to extend the dimensionality of scenario generation while relaxing the requirement for path dependence. This adjustment allows us to leverage more classical tools for modelling joint dependencies, such as copulas. However, even within this framework, only Gaussian copulas (and selected Student-T) have demonstrated the capacity to scale effectively to very high dimensions. This project aims to bridge the gap between the data-driven GAN-like approaches from the literature and the scalability of copulas by developing a high-dimensional copula model capable of capturing historical tail dependence properties. The ultimate goal is to generate realistic 'extreme' scenarios that reflect the complex dependencies observed in financial markets.


# Methodology and methods
As mentioned above, unlike studies where the intricacies of underlying asset path dynamics (such as time-dependent volatility or complex stochastic processes) are critical, this work deliberately abstracts from such details. As such there is no need to include dynamically-changing portfolios (i.e. dynamic trading strategies), we limit portfolio dimension optimisation to the static core. Portfolios of approx. 500 names with all long positions and 500 long/short hedged positions will be used in this study.

Figure 2 presents the overall workflow of the project. The aim is to calibrate copula to historical timeseries related to a list of selected portfolios.

![*Figure 2: The proposed copula calibration workflow.*](figures\Project-workflow-high.png)

Figure 3 further details the cumulative process of constructing different time series based on the observed history and representing the required data transformations.

![*Figure 3: The proposed process of data transformations for copula calibration.*](figures\Project-workflow-detailed.png)

Beyond scalability, another key advantage of the proposed architecture lies in the interpretability of results. While alternative approaches, such as Tail-GAN-based simulations, often produce outputs that are difficult to dissect or explain (see [[5]](#references) for a discussion), the methodology employed here offers a structure for understanding extreme scenarios. The framework decomposes market dynamics into a weighted combination of copulas (see [Copulas](#copulas) properties in the previous section), each with a clear and intuitive economic or statistical meaning. Not only will it enhance transparency but also provide a more actionable foundation for stress testing and capital planning. Indeed, the ability to attribute risk contributions to specific copula components is a significant improvement in practical applicability and model governance.

The workflow described in Figure 3 will be implemented in Python with recourse to well-known Python modules that are frequently used for such type of data processing. In particular, the Scikit-learn module described in Geron's textbook (see [[10]](#references)) supply us with Python classes required to implement the workflow above and testing phase.

It is important to note that the final phase of this process requires an optimisation procedure designed to identify the optimal coefficients for the various copulas under consideration. The core objective during this stage is to minimize the distance between the Value-at-Risk (VaR) measures generated by our target copula model (used in Monte-Carlo VaR) and those empirically observed in the historical data (HVaR). Achieving this precise calibration, however, presents a significant mathematical challenge. Because the mapping from the underlying copula parameters to the final VaR metrics does not offer straightforward analytical gradients, the system cannot rely on traditional calculus-based optimisation methods. Instead, this calibration process effectively transforms into the complex task of optimizing a noisy (i.e. seed changes), black-box function. In this environment, evaluations can be computationally expensive, and the outputs are often subject to statistical noise inherent in both the empirical data and the simulation processes. This scenario is conceptually identical to the challenge of hyperparameter optimisation in advanced machine learning models, where practitioners must efficiently search a vast and complex parameter space without knowing the underlying structure of the objective function. To overcome this challenge and find the best-fitting coefficients, the application  utilizes the scikit-optimize package (see [[11]](#references)). In this study we tried all tree different methods available in the package:

* Bayesian optimisation using Gaussian Processes;
* Sequential optimisation using gradient boosted trees;
* Sequential optimisation using decision trees;

# Requirements specification and Design

In terms of implementation this project requires two parts a) efficient code which is able to calculate Value-at-Risk for the prescribed portfolio b) research part which is able to calibrate joint high-dimensional copula copula, exploring different settings, algorithms and parametrisation.
Due to its different nature it was decided to follow different design approach for the each part. Part a) was implemented by following object oriented approach, where each class abstraction performs its own role of the calculations. It is composed of the following parts:

* __Market data gathering and preparation__, where the solution fetches and pre-processes all required data (i.e. 'market.py' module);

* __Joint simulation__ part is the central component of the current project, it serves to create scenarios expressing joint behavior, based on provided market data and different statistical hypotheses (i.e. 'jointsim.py' module);

* __Value-at-Risk calculation__, where generated scenarios are combined with the portfolio information to calculate profit and loss distribution. Once distribution is ready it can be queried for the specific percentile (i.e. 'var.py' module);

The figure below represents the UML class diagram for the Value-at-Risk calculation; each component is discussed in detail further in the section.

![*Figure 4: Value-at-Risk calculation UML class diagram.*](figures\class-diagram.png)

Part b) is done in a list of jupyter notebooks designed for the quick interactions with the results and exploitive study of different settings and parameters.

[TODO: explain jupytre notebooks]

## Market data gathering and preparation

To execute the complete workflow as detailed in the previous section, we start with the portfolio information provided as an input. Subsequently, we access historical stock price data from Yahoo Finance API, according to the specified timeframe duration. The existing implementation, presented within the 'Market' class in the 'market.py' file (see listing in [Appendix](#appendix-c-code-listing)), performs storing of all retrieved data in a locally saved CSV file for the reference in case multiple runs be necessary. After the gathering stage of all required timeseries, they undergo pre-processing to calculate log-returns, segmented into the chunks of dedicated length. 

## Joint simulation

The __JointSim__ class serves as an abstract base class specifically designed to provide a standardized framework for all copula implementations within the project. By defining a list of virtual functions, it enforces a consistent interface that all descendant classes must implement to ensure architectural integrity. Beyond that, the class centralizes the core analytical logic required for investigations, including the automated plotting of all simulations, resulting contour plots and tail coefficients. This centralized approach ensures that common statistical visualizations and diagnostic tools are applied uniformly across all models. The current project has integrated several specific copula models into its framework:

* The __HistSimulation__ class implements a historical simulation approach, which generates data by returning a sample of normalized observations derived directly from actual historical time series. Rather than assuming an underlying parametric distribution, this method relies on the empirical distribution of the observed data.
* The __GaussCopulaCorr__ class implements the Gaussian copula model by utilizing a specified correlation matrix as its primary input. To perform simulations, the class employs classical sampling techniques derived from a multivariate normal distribution. This process leverages the highly optimized multivariate_normal implementation provided by the NumPy library.
* The __GaussCopulaNum__ class provides an alternative implementation of the Gaussian copula designed specifically for applications involving actual historical time series data. Unlike traditional approaches, this implementation bypasses the potentially problematic process of calculating a formal correlation matrix from raw data. This design choice is intended to avoid the numerical instability and computational intensity often associated with such matrices, particularly the risk of generating a matrix that is not positive semi-definite. By avoiding this step, the class provides a more robust framework for working with empirical datasets where some irregularities might otherwise lead to mathematical errors.
* The __CauchyCopulaNum__ class implements a Cauchy copula, which can be viewed as a special case of the Student-T distribution with one degree of freedom. Much like the “GaussCopulaNum” implementation, this class is designed to accept actual historical time series as an input. This approach intentionally bypasses the computationally intensive and numerically sensitive task of calculating an explicit correlation matrix, thereby avoiding the common risk of encountering a matrix that is not positive semi-definite. Instead, the implementation is grounded in the properties of a standard multivariate Cauchy distribution with zero mean and unit scale, providing a robust method for modelling heavy-tailed dependencies without the instability of traditional matrix estimation. An addition benefit of the Cauchy copula is the ability to produce an alternative diagonal and generate de-correlation scenarios (see [section 3](#economics-of-the-cauchy-copula) for discussion). This capability is particularly valuable for stress-testing and risk management, as it enables the simulation of scenarios where assets or variables may de-correlate unexpectedly during periods of high volatility or extreme market stress.
* The __SkewCopulaNum__ class implements a skewed Cauchy copula, which serves as an asymmetric extension. Similarly to the CauchyCopulaNum implementation, this class utilizes actual historical time series as its primary input. To achieve asymmetric tail behaviour, the class applies a W-transform (see [[9]](#references) for details) to the underlying Cauchy copula. This transformation allows the model to capture directional dependencies and tail asymmetries that a standard symmetric Cauchy and Gaussian copulas would otherwise fail to represent, making it particularly effective for modelling empirical data with non-symmetric extreme events (i.e. market crisis behaviour is different to the periods of growth, see [section 3](#economics-of-the-cauchy-copula) for discussion).
* The __MixCopulaNumTest__ class is designed to perform mixture simulations by combining skewed Cauchy and Gaussian copulas. This approach is mathematically grounded in the principle that a convex combination of various copulas also constitutes a valid copula (see [section 3](#copulas) for details), thereby allowing for the modelling of complex, hybrid dependency structures. The user can define the specific composition of the mixture by providing the relative proportions of each constituent copula as an input parameter, enabling the simulation of data that exhibits both symmetric and asymmetric tail behaviours simultaneously, in addition to the alternative diagonal (i.e. de-correlation scenarios).
 
## Value-at-Risk calculation
As before the __Var__ class serves as an abstract base class, designed to establish a standardized, cohesive framework for Value-at-Risk (VaR) calculations. This foundational class integrates the core computational components of the risk model by combining the portfolio definition, the underlying copula specification, and the empirical market data required to construct marginal distributions. Although __VarSim__ is currently the sole concrete child class implemented within this project, the object-oriented architecture is inherently extensible, deliberately leaving room to seamlessly incorporate alternative models and methodologies in future research. In its execution, processes output scenarios generated by the copula, which manifest as uniformly distributed coordinate vectors situated within a $d$-dimensional hypercube $[0, 1]^d$. To translate these theoretical, uniform probabilities into realistic market conditions, the system maps them against the market data serving as the marginal distribution for each individual stock. This transformation is executed utilizing the "quantile" function (effectively the inverse cumulative distribution function) provided by the NumPy package. During this step, the model explicitly relies on the standard assumption of lognormality, which is widely accepted convention for modeling the behavior of equity returns. Once a simulated vector of individual asset returns is generated, it is multiplied with the portfolio weights to calculate the aggregated return of the entire portfolio. To ensure statistical robustness and accurately capture a wide spectrum of potential market movements, this exact procedure is iterated 10,000 times (i.e. number of Monte-Carlo simulations). Such sampling effectively constructs a probability distribution of the overall portfolio returns, from which specified VaR metrics can be extracted (e.g. 75th, 90th, 99th and 99.5th percentiles).


# Implementation 

# Testing and Evaluation 

# Results/Findings and Discussion 

# Conclusion/Recommendations/Future Work





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
10. Geron A., 2019. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Sebastopol: O’Reilly Media. 2nd Edition.
11. Scikit-optimize contributors, 2017 - 2020. User Guide.[(link)](https://scikit-optimize.github.io/stable/)


# Appendixes

## Appendix A: List of used AI tools

 * Gemini via Google Search and as a chatbot;
 * Mistral-7B-Instruct-v0.3 for polishing the text. Prompt: *"You're an academic assistant with an expertise in software engineering to proofread and rewrite as a single connected text. Keep academic tone."*

## Appendix B: Copula examples
**Gaussian copula.**

![Figure A.1: Uniform scenarios for Gaussian_copula, numerical simulation.](figures\Uniform_scenarios_Gaussian_copula_(numerical_simulation).png)

![Figure A.2: Contour plot for Gaussian copula, numerical simulation.](figures\Contour_plot_Gaussian_copula_(numerical_simulation).png)

----------------

**Cauchy copula.**

![Figure A.3: Uniform scenarios for Cauchy copula, numerical simulation.](figures\Uniform_scenarios_Cauchy_copula_(numerical_simulation).png)

![Figure A.4: Contour plot for Cauchy copula, numerical simulation.](figures\Contour_plot_Cauchy_copula_(numerical_simulation).png)

## Appendix C: Code listing




