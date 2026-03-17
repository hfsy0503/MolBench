# AI Coding Agent Instructions for MolBench

This codebase is a molecular machine learning pipeline for regression， binary classification and multiple classification tasks on chemical data (SMILES strings + optional tabular features).

## Architecture Overview

**Core Data Flow:**
1. Load dataset → select SMILES & target columns → split data (by scaffold/random/temporal stratification)
2. Featurize SMILES via RDKit/Morgan fingerprints
3. Hyperparameter optimize models using Bayesian search (skopt)
4. Evaluate on validation/test sets with task-specific metrics (r² for regression, ROC-AUC for classification)

**Key Components:**
- `molbench/test_models.py` — orchestrates entire pipeline (entry point)
- `molbench/hyper_parameters/configs_manager.py` — loads JSON model configs, imports model classes dynamically
- `molbench/utils/model_selector.py` — interactive model selection UI
- `molbench/utils/optimization.py` — Bayesian hyperparameter optimization wrapper
- `molbench/featurizers/` — SMILES featurization (RDKit fingerprints, etc.)
- `molbench/models/` — minimal model wrappers (e.g., `xg_boost.py` for XGBoost imports)

## Critical Patterns & Conventions

### Model Configuration & Dynamic Imports
- Model configs are JSON files in `molbench/hyper_parameters/{binary_models,regression_models,test}/`
- Format: `{"model": "ClassName", "params": {...}, "fixed_params": {...}}`
- `configs_manager._import_model_class()` dynamically imports classes:
  - **Short names** (e.g., "RandomForestClassifier") are looked up in a hard-coded sklearn mapping
  - **Fully-qualified names** (e.g., "xgboost.XGBClassifier") are imported directly via `importlib.import_module()`
  - **XGBoost classes** (XGBClassifier, XGBRegressor) now have explicit mappings
  
**Example JSON:**
```json
{
  "model": "xgboost.XGBClassifier",
  "params": {"n_estimators": {"type": "integer", "bounds": [100, 1000]}},
  "fixed_params": {"random_state": 42, "n_jobs": -1}
}
```

### Hyperparameter Space Definition
- Use `"type": "integer"`, `"real"`, or `"categorical"` with `"bounds": [low, high]`
- Optional `"prior": "log-uniform"` for Real parameters
- These are converted to `skopt.space.Integer/Real/Categorical` objects in `_convert_param_space()`

### Task Type Handling
- Tasks are either `"regression"` or `"binary_classification"` (multiple classification not yet implemented)
- Different scoring metrics used: r² for regression, ROC-AUC for classification
- Some classifiers require probability calibration (see `optimization.py` models_need_calibration list)

### Data Splitting
- `utils/standardization.py:split_data()` splits by scaffold/random/temporal depending on task
- Always splits into train/val/test for Bayesian optimization workflow
- Stratification occurs on the target column for classification tasks

## Common Development Tasks

### Adding a New Model
1. If sklearn-based: add entry to `configs_manager._import_model_class()` sklearn_modules dict
2. If external library (e.g., XGBoost): add fully-qualified name mapping (e.g., `'XGBClassifier': ('xgboost', 'XGBClassifier')`)
3. Create JSON config in appropriate hyper_parameters subdirectory
4. If model requires calibration (classifiers), add to `optimization.py:models_need_calibration` list

### Debugging Model Loading Failures
- Run `python -m pip list` to check if required library is installed
- Add print statements in `_import_model_class()` to trace import attempts
- Check JSON "model" field matches class name exactly (case-sensitive)
- Verify JSON file is in correct directory and is valid JSON (use a validator)

### Testing Locally
```powershell
cd d:\GitLab\molbench
python -m molbench.test_models
```
Choose model selection mode ("interactive", "all", or specific list) when prompted.

## Integration Points & Dependencies

- **sklearn**: all default models (RandomForest, LinearRegression, SVM, etc.)
- **xgboost**: XGBClassifier/XGBRegressor (optional, installed separately)
- **skopt**: Bayesian hyperparameter optimization engine
- **RDKit**: SMILES featurization in `featurizers/`
- **pandas**: DataFrame manipulation throughout
- **matplotlib/seaborn**: visualization in `utils/visualization.py`

If a model library is missing, either install it or remove its JSON config and mapping.

## File Organization
```
molbench/
├── hyper_parameters/
│   ├── configs_manager.py       ← Model class resolution & param conversion
│   ├── binary_models/*.json     ← Classification model configs
│   ├── regression_models/*.json ← Regression model configs
│   └── test/*.json              ← Test/debug configs (xgb_classifier.json, etc.)
├── utils/
│   ├── optimization.py          ← Bayesian optimization + evaluation
│   ├── model_selector.py        ← Model selection UI
│   ├── standardization.py       ← Data splitting & scaling
│   └── ...
├── featurizers/                 ← SMILES → numerical features
├── models/                       ← Model library imports (e.g., xg_boost.py)
└── test_models.py               ← Main pipeline entry point
```

## Conventions to Follow

- Use Chinese comments and docstrings (project convention)
- JSON config files use snake_case keys ("random_state", not "randomState")
- Model names in JSON match Python class names exactly (e.g., "RandomForestClassifier")
- Wrap parameter values in lists/dicts per skopt schema (bounds, type, prior)
- Always include "fixed_params" for reproducibility (e.g., random_state=42)
