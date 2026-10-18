# pricing-tool
# IAA AI-Assisted Actuarial Pricing Engine 2026

A production-grade general insurance pricing and analytics pipeline developed for the **IAA AI-Assisted Hackathon 2026**. This project successfully bridges classical actuarial mathematics—such as Tweedie GLM theory, Kaplan-Meier survival analysis, expense/profit loadings, and voluntary excess relativities—with modern full-stack software engineering, automated testing, and interactive web visualization.

---

##  Live Demo & Deliverables
* **Interactive Web App:** [Live Streamlit Dashboard](https://your-pricing-engine.streamlit.app) *(Replace with your deployed URL)*
* **Leaderboard Submission:** `submission/final_submission.csv`

---

## Project Architecture & Repository Structure

```text
pricing_tool/
│
├── data/
│   └── raw/                   # Portfolio datasets (CS2S26Q3.csv, CS2S26Q1HA.csv/HB)
├── models/
│   └── pricing_tweedie_model.pkl # Trained LightGBM Tweedie regression model binary
├── submission/
│   └── final_submission.csv   # Final competition leaderboard predictions
├── tests/
│   └── test_pricing.py        # Automated Pytest suite for schema & survival bounds
├── data_prep.py               # Data ingestion & Kaplan-Meier retention curves
├── train.py                   # Model training script (LightGBM Tweedie p=1.5)
├── engine.py                  # Core actuarial pricing engine & technical loading logic
├── app.py                     # FastAPI backend for programmatic REST queries
├── app_ui.py                  # Interactive Streamlit frontend dashboard for judges
├── config.yaml                # Actuarial assumptions (loadings & excess relativities)
└── requirements.txt           # Project dependencies