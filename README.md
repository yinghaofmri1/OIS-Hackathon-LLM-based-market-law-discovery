# LLM-Based Multi-Family Market Law Discovery

A research pipeline for **discovering and testing interpretable empirical relationships between financial news, historical market features, and future market outcomes**. The workflow uses an instruction-tuned language model to propose mathematical expressions, optionally adapts it with LoRA, fits coefficients on historical training data, and evaluates candidate equations against outcome-specific historical baselines.

> **Research status:** This notebook implements a *hypothesis-generation and statistical-validation workflow*, not a system that establishes causal or universal market laws. Its default historical test period has previously been inspected and should be treated as **exploratory**, not a fresh independent replication. The notebook describes an earlier synthetic end-to-end check without the LLM; real-data execution and GPU optimizer training are not verified by the supplied notebook.

## Notebook

[

Designed to run top-to-bottom in **Google Colab with GPU acceleration**, using Google Drive for inputs and generated results.

## Research questions

Can an LLM propose compact, interpretable equations that improve prediction beyond market-history baselines? Do news-derived features add incremental information? Are promising structures consistent across training folds, validation, and historical test data?

The notebook examines five separately defined target families:

| Family | Prediction target | Units / interpretation |
| --- | --- | --- |
| `excess_return` | Forward excess log return | Basis points (bps) |
| `realized_risk` | Root mean square of future daily log returns | bps/day; **not** demeaned volatility |
| `volume_activity` | Average future trading-volume z-score | Standardized activity; **not** raw volume growth |
| `market_shock` | Absolute equal-weight mean future stock log return in the sampled universe | bps; **not** an exchange-index return |
| `dispersion` | Cross-sectional standard deviation of future cumulative stock log returns | bps |

Equations can include nonlinear transformations, interactions, and lagged predictors. Each family is evaluated relative to its **own** historical-price/state baseline.

## Method overview

```text
Historical stock data + lagged news factors
                  |
                  v
      Calendar alignment and feature engineering
                  |
                  v
    Five outcome-specific modeling datasets
                  |
                  v
  LLM equation proposals + template/random controls
                  |
                  v
  Train-period coefficient fits + purged cross-validation
                  |
                  v
      Optional LoRA adaptation and new search
                  |
                  v
   Validation selection and structure preregistration
                  |
                  v
    Final historical TEST evaluation (once)
                  |
                  v
  Baseline tests + news ablation + global Holm correction
                  |
                  v
    Equations, diagnostics, coefficients and reports
```

Key design choices:

- **Leakage-aware timing:** Configurable forecast horizon and execution/news lags; candidate formulas exclude future-return label columns.
- **Interpretable modeling:** At most six equation terms by default; coefficients estimated with ridge-regularized regression.
- **Model selection:** Purged time-series cross-validation, robustness/complexity preferences, and a separate validation split.
- **LLM-assisted search:** `Qwen/Qwen3-4B-Instruct-2507` proposes candidate expressions; LoRA training can be enabled. The configured default objective is `ei_robust`; other supported objective names include `ei_mse`, `dpo`, and `grpo`.
- **Controls:** Price/history baselines, random/template candidates, and tests of incremental news value.
- **Multiple testing:** Final baseline-comparison and news-incrementality p-values are jointly Holm-adjusted across the evaluated families/candidates.
- **Cautious inference:** The notebook labels insufficient evidence and previously inspected holdouts explicitly; pointwise intervals are not simultaneous confidence bands.

## Setup

1. Open the notebook in Google Colab and select **Runtime → Change runtime type → GPU**.
2. Make sure your Google Drive contains the required real-data inputs (see below), or choose synthetic demonstration mode.
3. Edit the configuration in the notebook's first code cell.
4. Run all cells in order. The notebook installs missing Python dependencies and mounts Google Drive in Colab when using real data.

Typical dependencies installed as necessary:

`numpy`, `pandas`, `scipy`, `matplotlib`, `ipython`, `pyarrow`, `lxml`, `requests`, and (with LLM enabled) `torch`, `transformers>=4.51,<5`, `accelerate`, `bitsandbytes`, `peft`.

A GPU and adequate memory are recommended for the default 4B language model. Actual resource requirements depend on the runtime and configuration.

## Input data and paths

The notebook's **default paths** are:

| Configuration variable | Default | Purpose |
| --- | --- | --- |
| `DRIVE_DATA_PATH` | `/content/drive/MyDrive/llm_hackathon/data/dataset.parquet` | Historical per-ticker market data |
| `NEWS_FACTORS_PATH` | `/content/drive/MyDrive/LLM_hackathon/market_kepler/news_factors/news_factors_trading.csv` | Date-indexed aggregate news factors |
| `SECTOR_CACHE_INPUT` | `/content/drive/MyDrive/llm_hackathon/results/alpha_agent_evo/ticker_sectors.csv` | Cached ticker/sector mapping |
| `OUTPUT_ROOT` | `/content/drive/MyDrive/llm_hackathon/results/equation_discovery` | Results directory |
| `RUN_NAME` | `multi_laws_v3_lora_en` | Name of the run-specific folder |

**Important:** The default directory names contain both `llm_hackathon` and `LLM_hackathon`; Google Colab paths are case-sensitive. Adjust them to match your Drive. These datasets are **not embedded in the notebook** and are not claimed to be part of this repository.

For real market data, the notebook expects fields including `date`, `ticker`, and positive `close` values. Additional historical/volume features are used or derived by the pipeline. News input requires a `date` column plus compatible numeric news-factor fields (e.g., `news_shock`, `news_mood`, topic/sector signals); the available factors determine which features can be used. Check the dataset-loading and feature-engineering cells for precise compatibility.

If a local sector cache is unavailable, the notebook has an optional lookup mechanism controlled by `ALLOW_YFINANCE_LOOKUP` (default `False`).

## Main configuration

```python
DEMO_MODE = False             # True: run on generated synthetic data
FAST = True                  # Smaller search/training/evaluation budgets
USE_LLM = True               # False: skip LLM proposals/training
RESUME = True
SEED = 42

HF_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LAW_FAMILIES = [
    "excess_return", "realized_risk", "volume_activity",
    "market_shock", "dispersion"
]

STUDY_START = "2015-04-01"
VAL_START = "2021-01-01"
TEST_START = "2023-01-01"
HORIZON = 5
EXECUTION_LAG = 1
NEWS_LAG = 1

MAX_TERMS = 6
TOP_FINAL = 2               # At most two selected equations per family
OBJECTIVES = ["ei_robust"]
LORA_R, LORA_ALPHA = 16, 32
HOLDOUT_IS_FRESH = False     # Do not switch to True for an inspected test period
```

### Suggested first run

For an end-to-end *synthetic* smoke test without model downloads or LoRA training:

```python
DEMO_MODE = True
USE_LLM = False
FAST = True
```

For a faster experiment on **your own real data** without LLM generation:

```python
DEMO_MODE = False
USE_LLM = False
FAST = True
```

For the full LLM-assisted workflow, set `DEMO_MODE=False`, `USE_LLM=True`, configure real input paths, and run with a GPU. A successful optimization step does **not** imply a better predictive equation.

## Output files

By default, outputs are written to:

```text
/content/drive/MyDrive/llm_hackathon/results/equation_discovery/
└── multi_laws_v3_lora_en/
    ├── ALL_PREREGISTERED.json
    ├── LAWS.csv
    ├── LAWS.json
    ├── COEFFICIENTS.csv
    ├── LAW_REPORT.md
    ├── excess_return/
    ├── realized_risk/
    ├── volume_activity/
    ├── market_shock/
    └── dispersion/
```

| File | Description |
| --- | --- |
| `ALL_PREREGISTERED.json` | Frozen equation structures and run metadata registered before final TEST evaluation |
| `LAWS.csv` / `LAWS.json` | Evaluated formulas, units, model comparisons, uncertainty, and evidence labels |
| `COEFFICIENTS.csv` | Fitted coefficients and related records |
| `LAW_REPORT.md` | Human-readable research summary of all selected empirical relationships |
| `<family>/model_bundle.json` | Family definition, selected models, baseline, preprocessing/feature metadata |

For training diagnostics, inspect the family-level `training_pool_diagnostics.json`, `training_status.json`, and `train_ei_robust.csv` when those are produced by the corresponding training path. Family subdirectories also contain intermediate search and model records.

## Interpreting results responsibly

A proposed equation is an **empirical statistical relationship**, not a conservation law, theorem, or proven causal mechanism. Evidence should be interpreted with particular care because:

- The notebook's default TEST interval begins in **2023**, but has already been inspected in development, so this is an exploratory historical check.
- News lags are implemented, but upstream timestamp fidelity, news labeling, and survivorship biases are **not fully audited**.
- The `market_shock` target is derived from the sampled stocks, rather than a market index.
- Coefficient intervals condition on the selected structure and preprocessing.
- Synthetic demonstration results do not establish real-world predictability.
- Independent future-period replication, realistic transaction-cost checks, and additional data auditing would be needed before treating the relationships as investable signals.

## Reproducibility

Keep the input-data version, notebook revision, run configuration, package versions, and output metadata together. Preserve the original held-out period and avoid adjusting structures or thresholds based on test results. Use an **actually untouched subsequent dataset** for independent confirmation, rather than merely toggling `HOLDOUT_IS_FRESH`.

## Project status

This README documents the supplied **v3 LoRA notebook**. The notebook itself reports syntax and non-LLM synthetic end-to-end checks for an underlying version, but **does not contain verified real-data discoveries or completed LoRA/GPU training results**. Numerical findings should be added only after reproducing the run and auditing its generated outputs.
