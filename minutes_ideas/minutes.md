# Meeting minutes and ideas

## 2026 Jan 26 - Fabrizio, Alessandro, Sergii 

* It is nice to formulate problem in ML terms defining: Solution, Measure
* To summarize, question could be posted as:
     what is the "best" copula for a given portfolio (long/short/hedged, sector/industry clustering) for a given quantile ?

PS. why do we need copula ? (or what is the problem) in this case we are actually facing a classical problem for Generative ML models. We need to estimate very high quantile (lets say 99.99%) but we have 250-750 observations (1 year - 3 year of data).
If copula is known you could sample as many data points as you want and estimate required quantile, or otherway if you can sample many datapoints you could effectively extract the copula from them. 


![Figure 1: Initial ideas](PXL_20260128.png)