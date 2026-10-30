# IAA AI-Assisted Hackathon 2026: Actuarial Pricing Engine

Pricing models for private motor, home buildings, and home contents. Each model predicts expected ultimate claim costs per policy row for Part 1 (developed market) and Part 2 (emerging market adaptation).

---

## 1. Quick Links
- **Write-up (5 pages):** file:///C:/Users/hp/Downloads/IAA%20Hackathon%202026%20Final%20Technical%20Report%20(1)%20(1).pdf

## 2. Results & Validation Metrics
*Metric: Tweedie $D^2$ ($p = 1.5$). $D^2$ represents the share of deviance removed relative to a constant-rate baseline.*

| Line | Part 1 Validation $D^2$ | Part 2 Validation $D^2$ | Tweedie GLM Benchmark (Part 1) |
|---|---|---|---|
| Motor | 29.99% | 29.99% | 15.20% |
| Buildings | 30.90% | 30.91% | 16.10% |
| Contents | 30.35% | 30.40% | 15.80% |

* **Validation Strategy:** 5-fold cross-validation grouped by customer identifier (`customer_id`) to prevent data leakage across policies held by the same customer, alongside a time-based hold-out on 2024.
* **SHAP Interpretability:** Feature attributions and diagnostic plots are saved in `reports_p2/`.

---

## 3. Methodology & Approach
1. **Data Ingestion & Hygiene:** Ingest and clean relational tables within each market. Joins strictly do not cross markets.
2. **Target Construction:** Define claim costs using loss dates falling within the policy exposure period, net of recoveries, excluding repudiated claims.
3. **Machine Learning Model:** Train a LightGBM regressor with a Tweedie objective ($p = 1.5$), benchmarked against a classical Tweedie GLM.
4. **Emerging Market Adaptation:** Harmonize shared features, compare actual vs. expected claim frequencies, and recalibrate baseline risk projections.
5. **Commercial Pricing Buildup:** Convert pure risk premiums to gross office premiums using the fixed and variable expense assumptions defined in `config.yaml`:
   $$P_{\text{gross}} = \frac{\text{Pure Risk Premium} + F}{1 - (v + c + m)}$$
   *(Where $F = 50.0$ CU, variable expense $v = 20\%$, commission $c = 5\%$, and profit margin $m = 3\%$, yielding an effective denominator of $0.72$).*

---

## 4. Market Scope & Feature Handling
* **Part 1 (Developed Market):** Trains on 2021–2024 policies for motor, buildings, and contents. Excludes operational post-claim variables (`customer_service_contacts`), outcome variables (`renewal_decision`), and demographic identifiers (`surname`).
* **Part 2 (Emerging Market):** Adapts to 2023–2024 experience featuring age bands, regional variations, free-text occupations, and local construction codes. Variables without a developed-market equivalent are dropped to maintain feature parity.

---

## 5. Human-AI Workflow & Verification
AI assisted with data exploration, leakage review, modeling diagnostics, and documentation drafting. In accordance with project governance standards, all AI outputs were verified against source data totals and tested via automated scripts. The complete prompt and review history is available in `docs/ai_log.md`.

Automated validation suite (`pytest tests/`) covers:
* Schema and data type integrity
* Market boundary isolation
* Non-negative and finite prediction bounds
* Row count and duplicate checks against submission sample templates
* Exposure scaling consistency (half-year policies pricing at approximately half of full-year policies)

---

## 6. Reproduction Instructions

Clone the repository and set up your local environment:

```bash
git clone [https://github.com/gitaukennedy/pricing-tool.git](https://github.com/gitaukennedy/pricing-tool.git)
cd pricing-tool
python -m venv venv 
# Windows GitBash:
source venv/Scripts/activate
pip install -r requirements.txt
