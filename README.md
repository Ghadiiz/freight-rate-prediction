# Freight Rate Prediction

Predicts the posted rate (USD) of freight loads from the pickup and delivery city, distance,
equipment type, weight and date. Built for the Spotter Machine Learning Engineer assessment.

## Project structure

```
freight-rate-prediction/
├── data/
│   ├── train_test.csv                       # labeled development data (Jan–Oct 2025)
│   ├── validation.csv                       # 12,000 loads to predict (Nov–Dec 2025)
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv            # filled with predictions by the pipeline
├── notebooks/
│   └── eda.ipynb                            # exploratory data analysis
├── src/
│   └── train_and_predict.py                 # full pipeline: clean, split, train, predict
├── score.py                                 # Spotter's scorer (unchanged)
├── validation_predictions.csv               # final predictions (output)
└── requirements.txt
```

## How to run

Tested with Python 3.9.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Train, evaluate, and generate both prediction files
python src/train_and_predict.py

# 4. Validate outputs and create the December chart
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Step 3 prints the test-set results and writes `validation_predictions.csv` and the filled
`data/december_chart_inputs.csv`. Step 4 creates `scorer_results/candidate_december.png`.

## Approach

**Data quality**

- 292 negative weights were sign errors (their absolute values match normal weights), so they
  were corrected with the absolute value. The same fix is applied to all files.
- 653 training loads (1.36%) priced below 0.5× or above 3× their own quote were treated as
  entry errors and removed from training only. Validation rows are never removed.
- Missing weights (<1%) are left as missing; the model handles them natively.

**Features**

- Pickup city, delivery city, equipment (categorical), distance, weight, and date features
  (day of week, day of month, month).
- `market_index`, `quote_signal` and the coordinates are not used: they are absent from the
  December file, and a test showed that adding them did not improve accuracy.

**Validation (time-based split)**

- The validation loads (Nov–Dec) all come after the training data (Jan–Oct), so the model is
  evaluated the same way: trained on Jan–Aug and tested on Sep–Oct (20% of the data).
- The test months are left untouched, including outliers, to give a realistic estimate.

**Model**

- `HistGradientBoostingRegressor` (scikit-learn) with absolute-error loss: strong on tabular
  data, handles categories and missing values natively, and is robust to extreme prices.
- The final model is retrained on all labeled data (Jan–Oct) before predicting.

## Results (test months Sep–Oct)

| Model                                                       | MAE         | MAPE      |
| ----------------------------------------------------------- | ----------- | --------- |
| Baseline: median rate per mile of the same lane + equipment | $144.85     | 6.31%     |
| Gradient boosting                                           | **$107.22** | **4.74%** |

**Limitations:** the training data contains no December, so the model cannot learn holiday
effects. The December chart reflects the weekly pattern (higher midweek, lower on weekends)
around the recent price level. Eight cities in the validation set never appear in training
(about 12% of validation loads); for these, the model relies on distance, equipment and date.
