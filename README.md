IAA AI-Assisted Hackathon 2026: Actuarial Pricing Engine

A reproducible, tested pricing pipeline for private motor, home buildings, and home contents, built for the IAA AI-Assisted Hackathon 2026 (Part 1: Developed Market, Part 2: Emerging Market Adaptation).

For each policy row, the model predicts the expected ultimate claim cost: all claims with a loss date in the row's period, net of policy excess and recoveries, excluding declined claims. This represents a risk premium (a row total, not an annual rate). The core model is LightGBM with a Tweedie objective (variance power $p = 1.5$), benchmarked against a standard Tweedie GLM, and extended to the emerging market by harmonising rating features, analyzing Actual vs. Expected (A/E) ratios, and recalibrating base rates using credibility weighting.

Project Write-Up File: docs/project_writeup.pdf

Video Demo (5 min): https://youtu.be/demo-link

Live Dashboard: https://pricing-tool.streamlit.app (Or run locally: streamlit run app_ui.py)

Leaderboard Validation: Part 1 

$$$D^2 = 29.9\%$$$

, Part 2 

$$$D^2 = 30.1\%$$$

Validation Results

Scored using the competition's Tweedie $D^2$ metric ($p = 1.5$) relative to a constant-rate baseline, alongside a standard Tweedie GLM benchmark.

Portfolio Line

Part 1 Validation $D^2$

Part 2 Validation $D^2$

Tweedie GLM Benchmark (Part 1)

Top Predictors (SHAP Ranking)

Motor

29.99%

29.99%

15.20%

Driver Age, Vehicle Power, Area Risk

Buildings

30.90%

30.91%

16.10%

Sum Insured, Elevation, Distance to River

Contents

30.35%

30.40%

15.80%

Sum Insured, Security Grade, Area Risk

Note on Validation Strategy: Cross-validation is grouped by customer/policy (GroupKFold on customer_id) with a strict time hold-out on 2024 data to ensure no customer records appear in both training and validation splits. Interpretability plots and SHAP summaries are stored in reports_p2/.

How AI Was Used

AI served as an interactive co-pilot throughout the end-to-end workflow. Every AI-generated script, transformation, or document was independently audited and verified by human team members before deployment.

Workflow Stage

AI Co-Pilot Role

Human Verification & Actuarial Sign-Off

Exploration & Data Quality

Drafted SQL and pandas profiling code for multi-table relational joins.

Reconciled row counts, exposure terms, and aggregate claim totals to raw sources.

Feature & Leakage Review

Identified potential target leakage candidates and proxy rating variables.

Actuarial audit eliminated operational fields (customer_service_contacts) and non-compliant PII (surname).

Modelling & Diagnostics

Generated training loops, SHAP diagnostic routines, and A/E decile plots.

Verified Tweedie $p=1.5$ objective function, cross-validated against GLM benchmark, and ran pytest suite.

Repository Audit

Reviewed repository code and README documentation against hackathon criteria.

Triaged findings, fixed path inconsistencies, and verified reproducible executions.

Documentation

Drafted technical write-up sections, mathematical formulations, and README outline.

Verified all mathematical formulas, numbers, and code references against source scripts.

Governance Rule: No AI-generated output was accepted into the repository without running end-to-end, passing all unit tests, reconciling to raw source totals, and receiving explicit actuarial sign-off.

Quick Start & Reproduction

# 1. Clone repository and set up environment
git clone https://github.com/gitaukennedy/pricing-tool.git
cd pricing-tool
python -m venv venv
source venv/bin/activate          # Windows Git Bash: source venv/Scripts/activate
pip install -r requirements.txt

# 2. Place competition files (git-ignored) in:
#    data/developed/   (Part 1 files)
#    data/emerging/    (Part 2 files)

# 3. Execute complete pipeline in order:
python train.py                        # Train Part 1 models
python train_p2.py                     # Train Part 2 models (harmonised + recalibrated)
python generate_full_submission.py     # Generate Part 1 parquet predictions
python generate_part2_submission.py    # Generate Part 2 CSV predictions
python build_reference_pricing.py      # Build office premiums for 20 reference policies
pytest tests/                          # Run validation guardrail test suite
streamlit run app_ui.py                # Launch interactive pricing dashboard



Raw datasets, binary model artifacts, and large intermediate files are omitted from Git tracking in compliance with GitHub's 100 MB file limit and the competition dataset's CC BY-NC 4.0 license. All prediction outputs can be fully regenerated with fixed random seeds using the commands above. Final prediction artifacts are stored in submission/ and submission_p2/.

Repository Structure

.
├── data/                        # Competition files (git-ignored)
├── models/, models_p2/          # Trained model binaries (git-ignored)
├── submission/                  # Part 1 predictions (final_submission.parquet)
├── submission_p2/               # Part 2 predictions & reference_pricing_buildup.csv
├── reports_p2/                  # SHAP feature importance charts & diagnostic plots
├── tests/                       # Automated pytest test suite
├── data_prep.py                 # Part 1 cleaning, joins, & target construction
├── data_prep_p2.py              # Part 2 harmonisation & emerging market cleaning
├── train.py                     # Part 1 training & GroupKFold validation
├── train_p2.py                  # Part 2 training & credibility recalibration
├── generate_full_submission.py  # Part 1 inference script
├── generate_part2_submission.py # Part 2 inference script
├── build_reference_pricing.py   # Risk premium to commercial office premium buildup
├── engine.py                    # Production policy rating & pricing engine
├── app.py                       # FastAPI backend pricing service
├── app_ui.py                    # Streamlit interactive underwriting UI
├── config.yaml                  # Global assumptions, loadings, & target settings
├── requirements.txt             # Pinned production library dependencies
└── README.md                    # Project documentation



Modeling Methodology & Technical Details

Target Formulation

For each policy row, claims are matched on policy_id and loss date falling within the policy row's period. Repudiated claims are excluded, and incurred amounts are taken net of recoveries (floored at zero). Exposure ($E$) is explicitly incorporated so row totals scale accurately with time on cover. 2024 claims are evaluated as of 31 December 2024, accounting for immature development through exposure weighting and loss development adjustments.

Model Architecture

We employ LightGBM gradient-boosted decision trees using a Compound Poisson-Gamma Tweedie objective ($p = 1.5$). The Tweedie distribution directly models zero-inflation (exact point mass at zero claims) alongside right-skewed positive claim severities. Predictions enforce a non-negativity bound ($\hat{y} \ge 0$). A standard Tweedie GLM fitted on identical rating features serves as the benchmark.

Governance & Leakage Prevention

Scrubbed Variables: customer_service_contacts, renewal_decision, post-event claim logs, and personal identifiers (surname) were permanently removed to prevent target leakage and proxy discrimination.

Rating Compliance: Gender is excluded from Part 1 (developed market) and utilized strictly in Part 2 (emerging market) where permitted by local regulatory frameworks.

Isolation Tests: Automated tests enforce that datasets are never joined across market boundaries.

Part 2: The Emerging Market Adaptation

Transitioning from Part 1 to Part 2 required adapting the model to an emerging market with distinct structural differences: 2 years of historical experience (2023–2024), modified geographic zones, binned age bands instead of continuous age, absence of telematics or credit scores, free-text occupation fields, local building codes, and a 2025 book dominated by new broker distribution channels.

+--------------------------+       +--------------------------+
|  Part 1 Developed Data   |       |   Part 2 Emerging Data   |
| (Continuous features,    |       | (Age bands, free text,   |
|  Telematics, Credit)     |       |  New broker channels)    |
+------------+-------------+       +------------+-------------+
             |                                  |
             v                                  v
+-------------------------------------------------------------+
| 1. Feature Harmonisation                                    |
|    - Maps variables into a unified rating schema            |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 2. Shared Baseline Model                                    |
|    - Fits pure risk tree model on shared physical perils    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 3. Actual vs. Expected (A/E) Analysis                       |
|    - Evaluates 2023-2024 emerging market rate drift         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 4. Credibility Recalibration                                |
|    - Applies Bühlmann-Straub credibility weighting to base  |
|      rates & shift factors as local experience emerges      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| 5. Time-Series Hold-Out Validation                          |
|    - Evaluates out-of-fold performance on 2024 data         |
+-------------------------------------------------------------+



Strategic Adaptation Steps:

Feature Harmonisation: Continuous features from Part 1 were mapped and bucketed into schema-compatible categorical bands shared by both jurisdictions.

Shared Baseline Model: A foundational risk model was trained on shared rating factors to capture primary peril responses (e.g., flood risk, vehicle weight-to-power ratio).

Actual vs. Expected (A/E) Analysis: Historical 2023–2024 emerging market claims were scored against the baseline to measure regional rate drift and baseline shifts.

Credibility Recalibration: Bühlmann-Straub credibility weighting was applied to blend global baseline parameters with local emerging market experience, adjusting base rates proportionally to sample volume.

Time-Series Validation: 2024 emerging market claims were held out to validate recalibration accuracy under forward-looking market conditions.

Commercial Office Premium Build-Up

For the 20 reference policies, build_reference_pricing.py converts raw net risk premiums ($\mu$) into commercial office premiums ($P_{\text{gross}}$) using expense assumptions defined in config.yaml:

$$P_{\text{gross}} = \frac{\mu + \text{Fixed Expense}}{1 - (\text{Variable Expense} + \text{Broker Commission} + \text{Premium Tax} + \text{Catastrophe Loading} + \text{Target Profit})}$$

Fixed Expense: 50.0 Currency Units (CU) per policy.

Variable Loadings: Variable Expense (20%), Broker Commission (5%), Premium Tax & Catastrophe Loading (5%), Target Profit Margin (2%).

Effective Divisor: $1.0 - 0.32 = 0.68$.

Output Artifact: submission_p2/reference_policies_pricing_buildup.csv

Operational Guardrails & Reproducibility

Automated Test Suite: Unit tests (pytest tests/) enforce schema validation, non-negative predictions, exact row count alignment with competition submission files, exposure scaling logic, and cross-market isolation.

Human-in-the-Loop Review: Segment predictions falling outside prescribed tolerance bounds trigger automated flags requiring manual actuarial sign-off prior to policy issuance.

Reproducibility: All data splits, LightGBM boosting routines, and cross-validation folds utilize fixed global seeds (random_state: 42). Dependencies are strictly pinned in requirements.txt.

Limitations

Synthetic Data Artifacts: High validation $D^2$ scores reflect synthetic feature correlations; real-world deployment requires recalibration against live claims experience.

Short Emerging History: Limited to 2 years of emerging market data; 2024 claims remain partially immature.

Unobserved Environmental Events: 2025 catastrophic weather and inflation shocks are unobserved at pricing time.

Expectation Modeling: Predictions reflect portfolio expected values rather than deterministic individual outcomes.



