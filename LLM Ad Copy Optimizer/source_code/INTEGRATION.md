# Generator-Model Integration

This document explains how to use `generator.py` and `model.py` together to generate and evaluate improved ad copies.

## Overview

The integration allows you to:
1. **Generate improved ad copies** using AI (generator.py)
2. **Automatically format them** with proper feature engineering
3. **Evaluate the improvements** using trained ML models (model.py)

## Key Changes Made

### Generator.py Updates
- Added `create_evaluation_ready_dataset()` method to process improved ads through feature engineering
- Updated `generate_from_csv()` to optionally create model-ready datasets
- Added feature engineering integration to ensure compatibility with model.py

### Model.py Updates
- Added `evaluate_ad_improvements()` method to compare original vs improved ad performance
- Added `evaluate_improved_ads()` convenience function for easy evaluation
- Enhanced prediction capabilities to handle both single ads and batches

## Usage Examples

### Basic Workflow

```python
from generator import generate_from_csv, LLMProvider
from model import evaluate_improved_ads

# Step 1: Generate improved ad copies with model-ready format
results, model_ready_df = generate_from_csv(
    csv_path="data/enriched_ads_with_metrics.csv",
    model_path="models/ctr_predictor_production.joblib",
    provider=LLMProvider.ANTHROPIC,  # or OLLAMA
    batch_size=5,
    output_path="data/pipeline/improved_ads.csv",
    create_model_ready=True  # Creates engineered features automatically
)

# Step 2: Evaluate improvements using trained model
evaluation_results = evaluate_improved_ads(
    model_path="models/ctr_predictor_production.joblib",
    improved_ads_csv="data/pipeline/improved_ads_model_ready.csv",
    output_path="data/pipeline/evaluation_results.csv"
)

# Step 3: Review results
print(f"Generated improvements for {len(results)} ads")
print(f"Average CTR improvement: {evaluation_results['ctr_improvement_pct'].mean():.1f}%")
```

### Advanced Usage

```python
from generator import AdCopyGenerator, LLMProvider
from model import CTRPredictor
import pandas as pd

# Initialize components
generator = AdCopyGenerator(provider=LLMProvider.ANTHROPIC)
predictor = CTRPredictor()
predictor.load_model("models/ctr_predictor_production.joblib")

# Load your ad data
df = pd.read_csv("data/enriched_ads_with_metrics.csv")

# Generate improvements for specific ads
improved_ads = generator.generate_batch_improvements(
    df=df.head(10),  # First 10 ads
    batch_size=10
)

# Create model-ready format
model_ready_data = generator.create_evaluation_ready_dataset(
    improved_ads, 
    original_data=df
)

# Evaluate improvements
evaluation = predictor.evaluate_ad_improvements(model_ready_data)

# Analyze results
top_performers = evaluation.nlargest(5, 'ctr_improvement_pct')
print("Top 5 improvements:")
print(top_performers[['ad_id', 'ctr_improvement_pct', 'improved_headline']])
```

## Data Flow

1. **Input**: Original ad data (`enriched_ads_with_metrics.csv`)
2. **Generator**: Creates improved versions + applies feature engineering
3. **Output**: Model-ready dataset with both original and improved versions
4. **Model**: Predicts CTR for both versions and compares performance
5. **Results**: Evaluation showing which improvements are most effective

## File Formats

### Generator Output
- `improved_ads.csv`: Basic improved ad copies with reasoning
- `improved_ads_model_ready.csv`: Same ads with full feature engineering for model evaluation

### Model Output
- `evaluation_results.csv`: Comparison of predicted CTR for original vs improved ads

## Requirements

- Trained CTR prediction model (from `model.py`)
- LLM access (Anthropic API key or Ollama installation)
- Feature engineering dependencies (`features.py`)

## Testing

Run the integration test to verify everything works:

```bash
cd source_code/app
python integration_test.py
```

This will test the complete pipeline with sample data and show expected outputs.

## Troubleshooting

### Common Issues

1. **Feature engineering not available**: Install required dependencies and ensure `features.py` is accessible
2. **Model not found**: Train a model first using `model.py` or check the model path
3. **LLM connection issues**: Verify API keys for Anthropic or ensure Ollama is running locally
4. **Column mismatch**: Ensure input data has the expected columns (`headline_text`, `body_text`, etc.)

### Debug Tips

- Use small batch sizes (3-5) for testing
- Check that feature columns match between training and evaluation data
- Verify that both original and improved versions are created with proper `_original` and `_improved` suffixes