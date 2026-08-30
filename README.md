# MCCopula

py -3.12 -m venv .myvenv

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.myvenv\Scripts\activate.ps1

pip install -r requirements.txt

## Abstract

The modelling of high‑dimensional joint dependence among financial risk factors remains a significant major challenge for the financial industry and Data Science. Classical statistical approaches, notably Gaussian copulas, fail to capture tail dependence adequately, leading to systematic under‑estimation of extreme portfolio losses. This MSc project aims to develop a scalable algorithm for calibrating high‑dimensional copulas that exhibit non‑Gaussian tail dependence while preserving observed marginal and correlation structures. The methodology combines semi‑analytical techniques with Machine Learning models to generate realistic joint‑distribution scenarios for up to several hundred risk factors. A proof‑of‑concept Python software implementation will be built to (i) retrieve and cache market data via the Yahoo Finance API, (ii) compute a portfolio Value‑at‑Risk using both historical and Monte‑Carlo simulations, and (iii) integrate the calibrated copulas as modular components within our new Monte‑Carlo engine. Empirical experiments will assess the impact of copula selection on Value‑at‑Risk estimates across a range of selected portfolios, followed by calibration routine to fit copula parameters to observed data. Finally a testing suite will verify adherence to prescribed dependence and tail criteria. The expected contributions are threefold: (1) a novel high‑dimensional copula calibration framework capable of reproducing empirical tail dependence, (2) a software prototype for market‑data acquisition, risk‑metric computation, and scenario generation, and (3) an empirical evaluation of how alternative copula assumptions affect risk‑measure outcomes, thereby offering a more robust toolset for stress‑testing and capital‑planning in high‑dimensional settings.
