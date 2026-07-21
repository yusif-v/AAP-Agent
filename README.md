# AAP-Agent - Autonomous Agent Prediction

AutoML agent for Kaggle's Autonomous Agent Prediction competition.

## Structure

```
submission/
├── agent.yaml           # Main agent config
├── prompts/
│   └── system.md        # System instructions
├── configs/
│   └── inference.yaml   # LLM config
└── skills/
    ├── data_analysis/   # Dataset inspection
    ├── model_training/  # ML training scripts
    └── submission_strategy/ # Selection logic
```

## How it works

The agent operates in a Kaggle sandbox with 30 submission limit, 60 min timeout, and $2 LLM budget.

### Strategy
1. **Quick EDA**: Inspect dataset size and feature types
2. **Adaptive Training**: Choose model complexity based on data size
3. **Ensemble Prediction**: Combine multiple models
4. **Strategic Submission**: Submit iteratively, track scores

### Model Selection by Data Size
- **< 1000 rows**: RF (n=100, depth=5) - avoid overfitting
- **1000-20k rows**: RF + XGBoost + LightGBM
- **> 20k rows**: Full ensemble with tuned params

## Local Testing

```bash
cd data
pip install -r requirements.txt
python run_local_eval.py --submission-dir ../submission --dataset train_01
```

## Competition Constraints

- 30 submission limit per session
- $2.00 LLM token budget (gemini-2.5-flash-lite: ~$0.25/M input)
- 60 minute session timeout
- Binary classification with AUC ROC metric