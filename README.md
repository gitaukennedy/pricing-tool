# IAA AI-Assisted Actuarial Pricing Engine 2026: Technical Report

A production-grade general insurance pricing and analytics pipeline developed for the **IAA AI-Assisted Hackathon 2026**. This report outlines the end-to-end methodology, mathematical formulations, machine learning architecture, and software engineering framework used to construct the final pricing engine and submission.

---

## Live Demo & Repository Assets
* **Interactive Web Dashboard:** [Streamlit Cloud Live App](https://your-pricing-engine.streamlit.app) *(Update with your deployed URL)*
* **Leaderboard Submission File:** `submission/final_submission.csv`

---

## 1. Mathematical Framework & Actuarial Methodology

### A. Exposure and Risk Normalization
In general insurance, policies are written at various times throughout the year, meaning exposure time differs across policyholders. Let $E_i$ represent the exposure period (in years) for policy $i$, defined as:

$$E_i = \frac{\text{End Date}_i - \text{Start Date}_i}{365.25}$$

All claim frequencies and severities are normalized against this exposure term to calculate annualised risk rates.

### B. Kaplan-Meier Survival Analysis for Retention & Lapse Modeling
To analyze policyholder duration, retention, and lapse behavior from the historical tables (`CS2S26Q1HA.csv` and `HB`), we implement the non-parametric **Kaplan-Meier survival estimator**. The survival probability $S(t)$ at time $t$ is calculated as:

$$S(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

Where:
* $d_i$ is the number of policy lapses or claims occurring at time $t_i$.
* $n_i$ is the number of active policyholders at risk just prior to time $t_i$.

This survival curve establishes duration-based risk adjustments across policy tenures.

### C. Tweedie Compound Poisson-Gamma Machine Learning Model
Insurance claim loss distributions are characterized by a spike at zero (policies with no claims) and a continuous right-skewed distribution for positive claim amounts. We model this via a **Tweedie Generalized Linear Model (GLM)** wrapped inside a gradient boosting framework (**LightGBM**). 

The Tweedie variance function is defined as:

$$V(\mu) = \mu^p$$

Where:
* $\mu = \mathbb{E}[Y]$ represents the expected claim cost.
* $p = 1.5$ is the Tweedie power parameter, capturing the compound Poisson-Gamma distribution ideal for insurance severities and frequencies.

### D. Technical Premium & Loading Formulas
Once the pure risk premium ($\text{PRP}$) is predicted by the machine learning model, the final technical premium is derived by applying voluntary excess relativities and statutory expense and profit loadings:

$$\text{Technical Premium} = \frac{\text{Pure Risk Premium} \times \text{Excess Relativity}}{1.0 - \text{Expense Loading} - \text{Profit Margin}}$$

Where:
* **Expense Loading:** Set to $20\%$ ($0.20$) to cover operational and administrative overhead.
* **Profit Margin:** Set to $5\%$ ($0.05$) to ensure targeted return on capital.
* **Excess Relativity:** Multiplier based on the policyholder's chosen deductible/voluntary excess level (e.g., higher excess yields a discount factor $< 1.0$).

---

## 2. Step-by-Step Pipeline Implementation

1. **Data Ingestion & Cleaning (`data_prep.py`)**:
   * Ingests portfolio risk tables (`CS2S26Q3.csv`) and retention tables (`CS2S26Q1HA.csv`/`HB`).
   * Handles missing values, performs feature encoding, and computes standardized exposure variables.

2. **Survival Curve Generation (`data_prep.py`)**:
   * Computes empirical survival probabilities across tenures to feed actuarial risk weighting.

3. **Model Training (`train.py`)**:
   * Fits a LightGBM Regressor using the Tweedie objective function ($p=1.5$).
   * Implements 5-fold cross-validation with early stopping to prevent overfitting.
   * Enforces strict non-negativity constraints ($\hat{y} \geq 0$) to align with insurance fundamentals.
   * Serializes the trained model binary to `models/pricing_tweedie_model.pkl`.

4. **Pricing & Loading Engine (`engine.py`)**:
   * Loads `config.yaml` containing business rules, loading percentages, and excess relativity tables.
   * Combines model predictions with actuarial adjustment factors to produce final quote breakdowns.

5. **Quality Assurance (`tests/test_pricing.py`)**:
   * Automated `pytest` suite validating dataframe schemas, survival probability bounds ($0 \leq S(t) \leq 1$), and non-negative premium outputs.

---

## Project Structure

```text
pricing_tool/
│
├── data/
│   └── raw/                   # Raw datasets (CS2S26Q3.csv, CS2S26Q1HA.csv, HB)
├── models/
│   └── pricing_tweedie_model.pkl # Serialized LightGBM Tweedie model
├── submission/
│   └── final_submission.csv   # Final competition leaderboard predictions
├── tests/
│   └── test_pricing.py        # Automated pytest suite
├── data_prep.py               # Data ingestion & Kaplan-Meier survival curves
├── train.py                   # LightGBM Tweedie model training script
├── engine.py                  # Core actuarial pricing engine & loading logic
├── app.py                     # FastAPI backend REST service
├── app_ui.py                  # Streamlit interactive dashboard for judges
├── config.yaml                # Actuarial assumptions & loading parameters
└── requirements.txt           # Python dependencies