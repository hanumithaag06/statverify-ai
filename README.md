# 📊 StatVerify AI

> Automated Statistical Analysis, Verification, and AI Explanation Platform for Heterogeneous Research Data.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-000000?style=flat)](https://github.com/langchain-ai/langgraph)
[![SciPy](https://img.shields.io/badge/Stats-SciPy%20%26%20StatsModels-8CA1AF?style=flat&logo=scipy&logoColor=white)](https://scipy.org)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4?style=flat&logo=googlegemini&logoColor=white)](https://deepmind.google/technologies/gemini/)

---

## 1. Problem Description

Researchers and students frequently receive statistical data in **CSV, Excel, PDF, image, or clipboard formats** and need to determine:

* What statistical test is appropriate?
* How should the data be interpreted?
* Are reported statistical values correct?
* Are assumptions and data quality issues satisfied?
* How can the results be presented in a clear report?

Traditional statistical software requires users to manually select tests and understand statistical assumptions. There is also a risk of **incorrect test selection, calculation errors, missing-data issues, and misleading interpretation**.

**StatVerify AI** addresses this by providing an automated statistical analysis and verification workflow that separates:

> **Data processing → Statistical decision → Calculation → Verification → Explanation → Reporting**

The AI layer is specifically designed for **interpretation**, not for performing statistical calculations.

---

## 2. Solution

**StatVerify AI** is a Python-based statistical analysis and verification platform that accepts heterogeneous research datasets, automatically understands their structure, recommends an appropriate statistical test, executes the calculation using established statistical libraries, verifies reported results when available, generates an explanation, and produces a PDF report.

The system is designed to avoid hard-coded assumptions about a particular dataset. It dynamically handles:

* Different variable names
* Continuous, categorical, binary, and ordinal variables
* Missing values & quality checks
* Different group sizes
* Multiple statistical tests (t-test, ANOVA, Chi-Square, Correlations, OLS, Logistic Regression, etc.)
* Heterogeneous input formats (CSV, Excel, PDF, Image, Clipboard)
* Reported vs calculated statistical values
* Statistical assumptions
* Research-oriented result interpretation

---

## 3. Overall Workflow

```text
                USER INPUT
                    │
                    ▼
        ┌──────────────────────┐
        │   Universal Parser   │
        │ CSV / Excel / PDF /  │
        │ Image / Clipboard   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Data Validation    │
        │ Missing / Invalid /  │
        │ Data Quality Checks  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Schema / Metadata    │
        │ Variable Detection   │
        │ Type Classification  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Analysis Planner   │
        │ Understand Variables │
        │ & Relationships      │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Statistical Extractor│
        │ Groups / Numeric /   │
        │ Categorical Data     │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Statistical Decision │
        │       Engine         │
        │ Select Appropriate   │
        │ Statistical Test     │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Statistical Engine   │
        │                      │
        │ t-test / ANOVA /     │
        │ Chi-square / etc.    │
        └──────────┬───────────┘
                   │
             ┌─────┴─────┐
             │           │
             ▼           ▼
       Verification     Result
             │
             ▼
        ┌──────────────────────┐
        │ Comparison /         │
        │ Tolerance Engine     │
        │                      │
        │ Reported vs          │
        │ Calculated Values    │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │    AI Explanation    │
        │                      │
        │ Explain results      │
        │ without recalculating│
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │    Report Engine     │
        │                      │
        │ Structured Report +  │
        │ PDF Export           │
        └──────────────────────┘
```

---

## 4. Tech Stack

| Layer | Technology |
|---|---|
| **Programming Language** | Python 3.9+ |
| **Data Processing** | Pandas |
| **Statistical Computing** | SciPy |
| **Statistical Modeling** | StatsModels (OLS & Logit) |
| **Data Validation / Models** | Pydantic |
| **Workflow Orchestration** | LangGraph |
| **AI Explanation** | Google Gemini (Gemini 2.5 Flash) |
| **PDF Generation** | ReportLab |
| **Logging** | Loguru |
| **Configuration** | Environment variables (`.env`) |
| **Testing** | `unittest` & `pytest` |
| **UI** | Streamlit + Custom Glassmorphism CSS |

---

## 5. Statistical Methods Supported

```text
src/
└── statistical_methods/
    ├── categorical.py     # Chi-Square, Fisher's Exact Test
    ├── correlation.py     # Pearson (r, R², Fisher Z CI), Spearman, Kendall Tau
    ├── non_parametric.py  # Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman
    ├── parametric.py      # Independent Welch t-test, Paired t-test, One-Way ANOVA
    └── regression.py      # OLS Linear Regression, Binary Logistic Regression
```

---

## 6. Quick Start & Execution

### 1. Installation

```bash
# Clone repository
git clone https://github.com/hanumithaag06/statverify-ai.git
cd statverify-ai

# Activate virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set your Gemini API key (optional for AI features):

```bash
cp .env.example .env
```

### 3. Run Web Application

Launch the Streamlit interactive dashboard:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run Test Suites

Run unit tests and end-to-end verification pipeline:

```bash
python -m unittest discover -s tests -p "test_*.py"
python verify_final.py
```

---

## 7. Key Design Principles

1. **No Hard-Coded Dataset Assumptions**: Dynamically inspects variable roles, group levels, and metadata.
2. **Deterministic Calculations**: All math is performed by SciPy/StatsModels. The AI layer never calculates numbers.
3. **Verification Engine**: Tolerance-based numerical comparison between paper-reported and SciPy-calculated values.
4. **Structured PDF Export**: Professional report generation featuring statistical results, assumption checks, and AI explanations.
