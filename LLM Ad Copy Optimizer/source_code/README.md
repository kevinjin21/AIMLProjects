# Source Code Documentation

Technical reference for the notebook pipeline and application modules.

---

## Notebook Pipeline

### 01_exploration-raw.ipynb
EDA on Meta Ads template dataset (`meta_ads_with_metrics.csv`). Establishes the target schema for enriching raw ad copy data: ad_id, headline_text, body_text, call_to_action, platform, ad_format, category, impressions, clicks, conversions, spend.

### 02_enrich-dataset.ipynb
LLM-powered data enrichment using LangChain + Ollama. Transforms raw Hugging Face ad copy (text + shape only) into the template format by:
- Parsing headline/body/CTA from raw text
- Categorizing ads (ecommerce, saas, education, etc.)
- Generating synthetic performance metrics (impressions, clicks, conversions, spend)
- Chunked processing with retry logic for API stability

### 03_exploration-enriched.ipynb
EDA on the LLM-enriched dataset. Validates data quality, checks distributions, and confirms readiness for feature engineering.

### 04_feature-engineering.ipynb
Applies `AdCopyFeatureEngineer` class to generate ML-ready features. Includes text analysis (length, word count, caps ratio), pattern detection (urgency, positive sentiment, numbers/percentages), categorical encoding, and performance metric derivation. Outputs `ml_ready_ad_features.csv`.

### 05_modeling.ipynb
Trains and evaluates the `CTRPredictor` model. Performs train/test split, hyperparameter tuning via GridSearchCV, and overfitting validation. Saves production model to `/models` as `.joblib`.

### 06_generation-evaluation.ipynb
Core ad optimization workflow:
- Loads trained CTR model and generates improved ad copy via LLM
- Uses `AdCopyGenerator` with Anthropic Claude or Ollama
- Evaluates improvements by comparing original vs. improved CTR predictions
- Outputs evaluation results and improved ad copies to `/data/pipeline`

### 07_results-presentation.ipynb
Visualizes optimization results. Includes CTR distribution comparisons, improvement statistics by category/platform, and sample ad comparisons.

---

## App Modules

### build_ad_dataset.py
Fetches ads from Meta Ad Library API.
- Configurable search terms, countries, and ad limits
- Generates synthetic CTR/impressions/CPC based on category distributions
- **Output**: Raw ad dataset CSV

### compute_metrics.py
Enriches ad data with derived performance metrics.
- **Adds**: CTR, CVR, CPC, CPA, high_performer flag (top 25% CTR)
- **Input**: Raw ad CSV → **Output**: Enriched CSV with metrics

### features.py
`AdCopyFeatureEngineer` class for ML feature engineering.
- **Text features**: Length, word count, caps ratio, punctuation counts
- **Pattern detection**: Urgency words, positive sentiment, numbers, percentages
- **Categorical encoding**: Label encoding + one-hot for platform, format, category
- **Performance features**: CTR/CVR quartiles, engagement score, cost metrics
- **Interaction features**: Text length × CTR, platform-format combinations
- **Methods**: `engineer_text_features()`, `engineer_categorical_features()`, `engineer_performance_features()`, `create_interaction_features()`, `select_final_features()`, `fit_transform()`

### model.py
`CTRPredictor` class for Random Forest CTR prediction.
- **Training**: Automatic feature preparation with data leakage prevention, GridSearchCV tuning, train/test evaluation
- **Validation**: Overfitting detection, R²/RMSE/MAE metrics
- **Inference**: `predict()` for single ads or batches
- **Persistence**: `save_model()` / `load_model()` via joblib
- **Insights**: Feature importance analysis for optimization guidance

### generator.py
`AdCopyGenerator` class for LLM-powered ad optimization.
- **Providers**: Anthropic Claude API or local Ollama models
- **Optimization prompts**: Dynamic prompt construction based on ML model insights, headline length analysis, and category-specific recommendations
- **Output**: Structured JSON with improved headline, body, and improvement reasoning
- **Batch processing**: `generate_improved_ads()` with rate limiting and error handling
- **Model-ready output**: Optional feature engineering on generated ads

### app.py
Streamlit dashboard for results visualization.
- Displays original vs. improved ad comparisons
- CTR improvement metrics and distributions
- Filtering by category, platform, improvement percentage
- Plotly-based interactive visualizations

---

## Pipeline Summary

```
1. build_ad_dataset.py      → raw ads CSV
2. compute_metrics.py       → enriched ads with metrics
3. 02_enrich-dataset.ipynb  → LLM-enriched to template format
4. features.py              → ML-ready features
5. model.py                 → trained CTR predictor (.joblib)
6. generator.py             → LLM-improved ad copies
7. app.py                   → results dashboard
```

---

## Quick Start Examples

For full integration details, see [INTEGRATION.md](INTEGRATION.md).

### Generate & Evaluate Improved Ads

```python
from app.generator import generate_from_csv, LLMProvider
from app.model import evaluate_improved_ads

# Generate improved ads with model-ready features
results, model_ready_df = generate_from_csv(
    csv_path="data/enriched_ads_with_metrics.csv",
    model_path="models/ctr_predictor_production.joblib",
    provider=LLMProvider.ANTHROPIC,
    create_model_ready=True
)

# Evaluate CTR improvements
evaluation = evaluate_improved_ads(
    model_path="models/ctr_predictor_production.joblib",
    improved_ads_csv="data/pipeline/improved_ads_model_ready.csv"
)
print(f"Avg CTR improvement: {evaluation['ctr_improvement_pct'].mean():.1f}%")
```

### Feature Engineering Only

```python
from app.features import AdCopyFeatureEngineer
import pandas as pd

df = pd.read_csv("data/enriched_ads_with_metrics.csv")
engineer = AdCopyFeatureEngineer()
df_features = engineer.fit_transform(df)
```

### CTR Prediction

```python
from app.model import CTRPredictor

predictor = CTRPredictor()
predictor.load_model("models/ctr_predictor_production.joblib")
predictions = predictor.predict(df_features)
```