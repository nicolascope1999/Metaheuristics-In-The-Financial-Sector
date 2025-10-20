# Terminology (grouped)

## Environment & Libraries

<details><summary><strong>pip install</strong></summary>
Purpose: Python package installation command from the Python Package Index (PyPI) or local requirements file.  
Usage in project: Used at the top of notebooks (%pip install -r requirements.txt) to make the execution environment reproducible across runs and timestamps (run_YYYYMMDD_HHMMSS/).  
Why: Ensures every GA or model experiment runs with the exact dependency versions (avoids pickle incompatibility and drift).  
When: Before importing libraries or re-running experiments on a fresh kernel / server.  
Where: In Jupyter cells (magic %pip) or terminal.  
Example:
```
%pip install -r requirements.txt
```
Relevance: Guarantees deterministic feature engineering, model training, and GA behavior by locking dependency versions.
</details>

<details><summary><strong>numpy</strong></summary>
Purpose: Core numerical array library for fast vectorized operations.  
Usage in project: Computing technical indicators (rolling means, returns), GA chromosome masks (binary arrays), mutation/crossover operations, volatility metrics, and fitness aggregation.  
Why: Speed + memory efficiency vs pure Python loops; essential for large feature matrices.  
When: Any transformation on OHLCV arrays or feature engineering calculations.  
Where: Imported as `import numpy as np`.  
Example – binary GA individual mutation:
```python
individual = (np.random.rand(len(feature_names)) < 0.5).astype(int)
```
Relevance: Enables efficient population evolution and indicator generation at scale.
</details>

<details><summary><strong>pandas</strong></summary>
Purpose: Tabular & time-series data handling (DataFrame / Series).  
Usage in project: Loading OHLCV CSVs, time index alignment, rolling windows for indicators (SMA, ATR, RSI), time-series splits, saving engineered_features.csv.  
Why: Built-in time-aware indexing prevents accidental lookahead; simplifies chaining of transformations.  
When: Every stage from raw ingestion → feature engineering → target creation.  
Where: Imported as `import pandas as pd`.  
Example – time-aware slice:
```python
filtered = df.loc[start_date:end_date]
```
Relevance: Central structure passed into GA for feature subsets (columns).
</details>

<details><summary><strong>datetime</strong></summary>
Purpose: Standard library module for timestamps, durations, scheduling splits, run folder naming.  
Usage in project: Generating run directory (`datetime.now()`), dynamic date filtering (last 6 months + 100 hours).  
Why: Ensures temporal integrity and reproducibility of experiment metadata.  
Example:
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```
Relevance: Encodes experiment provenance in directory layout (run_YYYYMMDD_HHMMSS/).
</details>

<details><summary><strong>matplotlib</strong></summary>
Purpose: Static plotting library.  
Usage in project: Performance comparison bar charts, convergence plots (fitness vs generations), distribution summaries.  
Why: Quick, scriptable, exportable (PNG) for audit trails.  
Example:
```python
plt.plot(ga.fitness_history)
plt.savefig("fitness.png")
``>
Relevance: Visual diagnostic of GA convergence and model improvements.
</details>

<details><summary><strong>seaborn</strong></summary>
Purpose: Statistical plotting layer over matplotlib with improved aesthetics.  
Usage in project: Return distributions, heatmaps, volatility histograms.  
Why: Faster exploratory visuals with consistent style (`sns.set_style('whitegrid')`).  
Example:
```python
sns.histplot(data=returns, bins=50, kde=True)
```
Relevance: Helps validate stationarity assumptions and class balance before GA.
</details>

<details><summary><strong>plotly</strong></summary>
Purpose: Interactive charts (zoom, hover) for OHLCV, indicators, multi-panel EDA.  
Usage in project: Candlestick + Bollinger Bands + RSI/MACD dashboards.  
Why: Interactive inspection of engineered signals and potential regime shifts.  
Example:
```python
go.Candlestick(open=df.Open, high=df.High, low=df.Low, close=df.Close)
```
Relevance: Qualitative validation that GA-selected features align with observable structure.
</details>

<details><summary><strong>sklearn (scikit-learn)</strong></summary>
Purpose: Core ML toolkit (models, preprocessing, CV, metrics).  
Usage in project: RandomForestClassifier, TimeSeriesSplit, StandardScaler, cross_val_score in GA fitness, GridSearchCV for tuned enhanced models.  
Why: Provides consistent API for model evaluation and stability metrics (cv_stability).  
Example – time-series CV inside GA:
```python
cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_sub, y, cv=cv)
```
Relevance: Supplies fitness backbone (accuracy + stability − complexity penalty).
</details>

<details><summary><strong>pickle</strong></summary>
Purpose: Python object serialization (byte streams).  
Usage in project: Saving models, selected feature masks, scaler objects, run_summary.pkl into run_YYYYMMDD_HHMMSS/.  
Why: Enables later reload for comparison / reproducibility; lightweight vs full pipeline export.  
Example:
```python
with open("rf_baseline.pkl", "wb") as f:
    pickle.dump(model, f)
```
Relevance: Persists GA outputs (selected_features_*.pkl) for downstream analysis.
</details>

<details><summary><strong>joblib</strong></summary>
Purpose: Parallel execution & efficient persistence of large numpy objects.  
Usage in project: Parallel fitness evaluation (`Parallel(n_jobs=-1)(...)`) to speed GA generation cycles.  
Why: Reduces wall-clock time for evaluating many feature subsets.  
Example:
```python
from joblib import Parallel, delayed
fitness = Parallel(n_jobs=-1)(delayed(f)(ind) for ind in population)
```
Relevance: Scales GA evaluation to larger feature spaces without rewriting concurrency primitives.
</details>

<details><summary><strong>dask</strong></summary>
Purpose: Parallel / distributed computing for larger-than-memory or chunked workflows.  
Usage in project (potential / optional): Could scale feature engineering or CV across cores / cluster when dataset grows beyond single-machine capacity.  
Why: Future-proofing; integrates with pandas-like syntax.  
Example (conceptual):
```python
import dask.dataframe as dd
ddf = dd.read_csv("raw_data/*.csv")
```
Relevance: Extension path when expanding from single FX pair to multi-asset or higher frequency tick data.
</details>

## Python

<details><summary><strong>run_YYYYMMDD_HHMMSS/</strong></summary>
Purpose: Timestamped experiment/run directory (e.g. run_20251009_143522) that isolates all artifacts from a single end‑to‑end execution (feature engineering, GA selection, model training).  
Why: Guarantees provenance, auditability, and prevents overwriting prior results (critical for comparing GA parameter impacts).  
When: Created once at the start of a notebook/workflow before any models or artifacts are persisted.  
Where: Project root (same level as notebooks/scripts).  
How Used: All pickles, JSON summaries, plots, and engineered feature metadata are written inside this folder. Downstream analysis scripts glob over run_* to build longitudinal performance dashboards.  
Relevance: Central anchor tying together selected features, engineered features (AMA/FDI), model baselines vs GA-enhanced models. Enables reproducibility and rollback.  
Example:
run_20251009_143522/
 ├─ rf_baseline.pkl  
 ├─ rf_enhanced_tuned.pkl  
 ├─ selected_features_rf.pkl  
 └─ run_summary.json
</details>

<details><summary><strong>*_model.pkl</strong></summary>
Purpose: Pickled baseline model objects (e.g. rf_baseline.pkl, xgb_baseline.pkl) trained on the full (non‑GA‑reduced) feature set.  
Why: Serves as control group for evaluating GA feature selection + engineered indicators impact.  
When: After initial preprocessing & baseline training phase (before GA selection).  
Where: Inside the run directory.  
How Used: Loaded later to compute performance deltas vs *_model_ga.pkl or *_enhanced*.  
Relevance: Establishes benchmark accuracy / stability / feature count.  
Example: rf_baseline.pkl loaded to compare f1_score against Random Forest enhanced model.
</details>

<details><summary><strong>*_model_ga.pkl</strong></summary>
Purpose: Pickled models trained on GA-selected (and possibly engineered) feature subsets (legacy naming if not using “_enhanced”).  
Why: Captures the performance uplift attributable solely to GA-driven dimensionality reduction (and later addition of AMA/FDI).  
When: After GA feature selection (before or after feature engineering, depending on workflow stage).  
Where: In the same timestamp run directory.  
How Used: Compared against *_model.pkl to quantify feature reduction %, accuracy delta, stability improvements (cv_stability).  
Relevance: Core evidence that metaheuristics improve generalization with fewer features.  
Example: xgb_model_ga.pkl vs xgb_baseline.pkl showing +1.8% f1_score with 55% fewer features.
</details>

<details><summary><strong>selected_features_*.pkl</strong></summary>
Purpose: Serialized tuple or structure containing (best_mask, best_fitness, selected_feature_names) from GA runs per model type (e.g. selected_features_rf.pkl).  
Why: Decouples selection results from model objects; allows retraining under new hyperparameters without rerunning GA.  
When: Immediately after GA .run() completes.  
Where: Stored in run directory; referenced by enhanced model training cell.  
How Used: Loaded to build reduced DataFrames (X[selected_features]) and to audit which categories (volatility, momentum, engineered) are favored.  
Relevance: Key artifact enabling reproducibility of the exact subset that produced reported metrics.  
Example: Inspecting len(pickle.load(f)[2]) to compute feature reduction percentage.
</details>

<details><summary><strong>X_*_scaled.pkl</strong></summary>
Purpose: Pickled pandas DataFrames of scaled feature matrices for each temporal split (X_train_scaled.pkl, X_val_scaled.pkl, X_test_scaled.pkl).  
Why: Preserves exact preprocessing (scaling parameters tied to training window) to avoid data leakage and allow consistent re-evaluation.  
When: After StandardScaler fit on training set and transforms on validation/test.  
Where: Inside run directory alongside scaler.pkl (the fitted scaler object).  
How Used: Reloaded for rapid model re-training or metric recalculation without re-running feature engineering.  
Relevance: Ensures identical numeric inputs for GA fitness replay or alternative model benchmarking.  
Example: pd.read_pickle('run_.../X_test_scaled.pkl') used to re-score a newly tuned XGBoost model.
</details>

<details><summary><strong>y_*.pkl</strong></summary>
Purpose: Serialized target Series for each split (y_train.pkl, y_val.pkl, y_test.pkl).  
Why: Locks the temporal segmentation—critical for fair comparison across different model or GA configurations.  
When: Generated concurrently with X_*_scaled.pkl.  
Where: In the same run directory.  
How Used: Reloaded to evaluate alternative classifier architectures or recalculated metrics (precision/recall drift analysis).  
Relevance: Guarantees target alignment with stored feature matrices (prevents accidental re-splitting with different boundaries).  
Example: y_test length check equals X_test_scaled rows to validate integrity.
</details>

<details><summary><strong>feature_names.pkl</strong></summary>
Purpose: Pickled list of all original engineered feature column names (excluding Target).  
Why: Provides immutable reference for mapping GA chromosome indices → column names.  
When: Right after saving engineered_features.csv and before GA initialization.  
Where: Saved once per run.  
How Used: Passed into GeneticAlgorithm(...) and for interpreting selected_features_*.pkl masks.  
Relevance: Ensures consistent ordering; prevents silent misalignment if DataFrame column order changes in future revisions.  
Example: feature_names[42] used to decode a chromosome bit set to 1.
</details>

<details><summary><strong>README.txt</strong></summary>
Purpose: Human-readable run manifest summarizing workflow steps, file roles, timestamps.  
Why: Lightweight provenance layer aiding audits and quick triage without unpickling artifacts.  
When: Created immediately after run directory creation (before heavy processing).  
Where: Root of the run_YYYYMMDD_HHMMSS/ folder.  
How Used: Reviewed to confirm GA parameters, model types, and artifact naming conventions during post-run review.  
Relevance: Bridges automated artifacts and analyst comprehension, supporting reproducibility claims.  
Example Excerpt:
GA Steps: Baseline → GA Selection (RF,XGB) → AMA/FDI → Enhanced Training
</details>

<details><summary><strong>artifact store</strong></summary>
Purpose: Conceptual aggregation of all persistent run artifacts (models, feature subsets, metrics, plots). Could be the filesystem (current setup) or an external registry (S3, MLflow artifact store).  
Why: Centralizes reproducible units enabling differential analysis, rollback, and governance.  
When: Implicitly populated throughout the workflow steps.  
Where: Currently the set of run_* directories; future extension: remote object storage.  
How Used: Scripts can iterate over run_* to produce longitudinal performance dashboards (e.g. trend of f1_score vs feature count).  
Relevance: Foundation for scaling toward MLOps (versioned artifacts + metadata).  
Example: Glob pattern run_*/run_summary.json to assemble experiment matrix.
</details>

<details><summary><strong>experiment tracking (MLflow, Weights & Biases)</strong></summary>
Purpose: Externalized logging of parameters, metrics, artifacts, and lineage (optional extension beyond local pickles).  
Why: Enables comparison across runs, hyperparameter sweeps, GA setting experiments (mutation_rate, min_features), and facilitates collaborative review.  
When: Would be integrated at each stage: after GA selection (log feature count, fitness), after model evaluation (log metrics), after feature engineering (log AMA/FDI params).  
Where: Remote tracking server (MLflow) or hosted service (Weights & Biases) alongside local artifacts.  
How Used: Query dashboards to identify which GA strategies (ILM/DHC vs baseline) yield highest cv_stability with minimal features.  
Relevance: Eases iterative improvement and governance; complements static README.txt.  
Example (MLflow pseudo):
mlflow.log_param("mutation_rate", 0.2)
mlflow.log_metric("rf_f1_enhanced", 0.7135)
</details>

## Data & DataFrame Operations

<details><summary><strong>dataset</strong></summary>
Purpose: Cohesive collection of observations (rows) and variables/features (columns) covering a contiguous temporal window.  
Usage in project: Hourly OHLCV plus engineered indicators loaded from engineered_features.csv or raw_data/*.csv before GA.  
Why: Defines the scope over which feature engineering, splitting, and GA selection operate to preserve temporal integrity.  
When: Loaded once per run; then filtered to last N months + buffer hours.  
Where: raw_data/, working_data/, run_*/ cached pickles.  
Relevance: Correctly scoping the dataset avoids leakage from future timestamps.  
Example: EURUSD hourly candles (≈ 6 months + 100 hours buffer) feeding GA feature selection.
</details>

<details><summary><strong>dataframe</strong></summary>
Purpose: pandas 2D labeled data structure holding columns of potentially heterogeneous types.  
Usage: Core container for OHLCV, engineered indicators, target variable (Target).  
Why: Time-index alignment enables safe rolling windows & forward-looking horizon shifts.  
When: Every transformation step (returns, ATR, RSI, GA masks).  
Where: In-memory object passed to GeneticAlgorithm and feature engineering GA.  
Relevance: Preserves column order used to map chromosome bits → feature names.  
Example:
```python
df = pd.read_csv("working_data/engineered_features.csv", index_col="Datetime", parse_dates=True)
```
</details>

<details><summary><strong>datatype</strong></summary>
Purpose: Underlying type of a column (float64, int64, category).  
Usage: Ensuring numeric dtypes before scaling / model fit; avoiding object columns in GA mask selection.  
Why: Models & scalers require numeric arrays; improper dtypes cause errors or silent coercions.  
When: After feature engineering, before scaling and GA.  
Relevance: Mixed or object dtypes would distort StandardScaler and CV scores.  
Example: `df.dtypes`
</details>

<details><summary><strong>nan</strong></summary>
Purpose: Marker for missing/undefined numeric values.  
Usage: Produced by rolling windows, shifts, percent changes at series starts.  
Why: Must be removed or imputed before model training to avoid dropped rows inconsistently.  
When: After indicator creation but before target generation & splitting.  
Relevance: Unhandled NaN rows create misalignment between X and y.  
Example: `df = df.dropna()`
</details>

<details><summary><strong>constant</strong></summary>
Purpose: Feature whose values do not change (zero variance).  
Usage: Detected to avoid wasting chromosome bits and model capacity.  
Why: Provides no predictive signal; can inflate dimensionality penalty.  
When: Post-engineering QA; during GA fitness (variance checks).  
Relevance: Constant features reduce effective diversity and may bias stability metrics.  
Example: A Bollinger width column all zero if volatility = 0 early in series.
</details>

<details><summary><strong>variable</strong></summary>
Purpose: Any measurable attribute (column) that takes different values over time.  
Usage: All engineered indicators are variables considered for selection.  
Why: Variability enables discrimination among target classes.  
Relevance: Low-variance variables may still be predictive (e.g., directional ratio) but are evaluated via stability.  
Example: `ATR_14`, `RSI_14`.
</details>

<details><summary><strong>feature</strong></summary>
Purpose: Predictor column used as model input.  
Usage: Each binary chromosome bit toggles inclusion.  
Why: Combining informative features increases predictive power; pruning reduces overfitting.  
Relevance: Defines search space for GA.  
Example: `MACD_Histogram`.
</details>

<details><summary><strong>feature set</strong></summary>
Purpose: Complete collection of candidate features before selection.  
Usage: Input to GA initialization.  
Why: Establishes maximum possible dimensionality for masks.  
Relevance: Directly impacts search complexity (min_features / max_features bounds).  
Example: 300 engineered indicators + AMA/FDI (after engineering phase).
</details>

<details><summary><strong>feature subset</strong></summary>
Purpose: Chosen subset (mask=1) from full feature set.  
Usage: Evaluated via CV to compute fitness = accuracy + stability − penalty.  
Why: Balances parsimony vs performance.  
Relevance: Central GA output stored in selected_features_*.pkl.  
Example: 43 of 300 features (≈ 85% reduction).
</details>

<details><summary><strong>target feature</strong></summary>
Purpose: Label column (classification) the models predict (Target).  
Usage: 3-class (0 down, 1 up, 2 sideways) built from future returns vs volatility thresholds.  
Why: Captures directional regime while controlling noise.  
Relevance: All fitness calculations reference this series; no leakage allowed.  
Example: `feature_df['Target']`.
</details>

<details><summary><strong>.rolling()</strong></summary>
Purpose: pandas method for window-based computations (mean, std, skew).  
Usage: SMA, ATR, volatility, RSI averages.  
Why: Encodes temporal context & smooths noise.  
When: Feature engineering phase.  
Relevance: Rolling windows introduce initial NaNs trimmed before training.  
Example: `df['SMA_20'] = df['Close'].rolling(20).mean()`
</details>

<details><summary><strong>.std()</strong></summary>
Purpose: Standard deviation calculation (overall or rolling).  
Usage: Volatility estimates, z-scores, Bollinger Bands width.  
Why: Measures dispersion for normalization & thresholding.  
Relevance: Drives volatility-based target thresholds.  
Example: `rolling_std = close.rolling(20).std()`
</details>

<details><summary><strong>.pct_change()</strong></summary>
Purpose: Percentage change between current and prior value.  
Usage: Returns, Percentage_Change, future horizon movement for target.  
Why: Scale-invariant measurement of price movement magnitude.  
Relevance: Underpins volatility & direction classification.  
Example: `df['Percentage_Change'] = df['Close'].pct_change()`
</details>

<details><summary><strong>.shift()</strong></summary>
Purpose: Offset series by n periods.  
Usage: Lagging closes for returns, detecting crossovers, future target.  
Why: Enables label creation without reindex merges.  
Relevance: Wrong shift direction causes lookahead leakage.  
Example: `future = close.pct_change(5).shift(-5)`
</details>

<details><summary><strong>fill forward</strong></summary>
Purpose: Forward-fill missing values (`ffill`).  
Usage: Handling intermittent NaNs (e.g., due to partial bars).  
Why: Maintains continuity when gaps short & non-systematic.  
Relevance: Avoids dropping large contiguous blocks.  
Example: `df.ffill()`
</details>

<details><summary><strong>fill backward</strong></summary>
c
</details>

<details><summary><strong>.fit()</strong></summary>
Purpose: Learn parameters from training data (mean/std for scaler).  
Usage: StandardScaler fit on training split only.  
Why: Prevents leakage from validation/test distribution.  
Relevance: GA CV uses transformed subsets consistent with temporal order.  
Example: `scaler.fit(X_train)`
</details>

<details><summary><strong>.fit_transform()</strong></summary>
Purpose: Convenience to fit then transform in one call.  
Usage: Often for training set scaling.  
Why: Reduces code & ensures atomic fit/transform.  
Relevance: Replace with explicit fit/transform if reproducibility logging needed.  
Example: `X_train_scaled = scaler.fit_transform(X_train)`
</details>

<details><summary><strong>.transform()</strong></summary>
Purpose: Apply learned parameters to new data (validation/test).  
Usage: Scaling val/test sets, GA fold splits.  
Why: Ensures identical scaling basis.  
Relevance: Mandatory for stable fitness evaluation.  
Example: `X_test_scaled = scaler.transform(X_test)`
</details>

<details><summary><strong>scaler</strong></summary>
Purpose: Object performing numerical feature normalization (e.g., StandardScaler).  
Usage: Standardizes columns to zero mean/unit variance before model & GA evaluation.  
Why: Models (especially gradient-based / distance-based) benefit from normalized scale; stabilizes CV variance.  
Relevance: Stored (scaler.pkl) for reproducibility and later inference.  
Example: `scaler = StandardScaler()`
</details>

<details><summary><strong>StandardScaler()</strong></summary>
Purpose: sklearn transformer computing (x - mean)/std per feature.  
Usage: Primary scaler for all numeric features in this project.  
Why: Keeps distribution centered; robust enough for tree + XGBoost comparability and logistic inside feature GAs.  
Relevance: Consistent scaling critical for comparing baseline vs GA-enhanced.  
Example:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
</details>

<details><summary><strong>z-score</strong></summary>
Purpose: Normalized value (x - mean)/std.  
Usage: Returns_Z_Score_, Price_Z_Score_, anomaly & regime detection.  
Why: Standardizes across periods enabling comparability.  
Relevance: Additional standardized features expand GA search diversity.  
Example: `df['Price_Z_Score_20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()`
</details>

<details><summary><strong>winsorize / clip</strong></summary>
Purpose: Limit extreme outliers to percentile thresholds.  
Usage: Clipping tick volume spikes (`TickVol_Clipped`).  
Why: Reduces undue influence on scaling & model splits.  
Relevance: Prevents instability in cv_stability due to rare extremes.  
Example: `df['TickVol_Clipped'] = df['Tick_Volume'].clip(upper=df['Tick_Volume'].quantile(0.99))`
</details>

<details><summary><strong>imputation</strong></summary>
Purpose: Strategy to fill missing values (mean, ffill, model-based).  
Usage: Simple forward/backward fills for initial NaNs; more complex imputation avoided to prevent bias.  
Why: Preserve temporal continuity while minimizing synthetic signal.  
Relevance: Ensures consistent sample counts across CV folds.  
Example: `df = df.ffill().bfill()`
</details>

<details><summary><strong>resampling</strong></summary>
Purpose: Adjust dataset sampling (time aggregation or balancing classes).  
Usage: Potential future extension (hour→4H) or balancing rare classes.  
Why: Manage class imbalance & temporal granularity.  
Relevance: Affects target distribution stability used in fitness.  
Example: `df.resample('4H').agg({...})`
</details>

<details><summary><strong>stratified sampling</strong></summary>
Purpose: Maintain class proportions across splits.  
Usage: Not applied (time-series split overrides) but conceptually important.  
Why: Standard random stratification disallowed due to temporal order.  
Relevance: Mention clarifies why TimeSeriesSplit is chosen instead.  
Example: (Not used) `StratifiedKFold` (would introduce leakage).
</details>

<details><summary><strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong></summary>
Purpose: Generate synthetic samples for minority classes.  
Usage: Placeholder cell (# SMOTE HERE) for possible handling of rare Up/Down vs Sideways.  
Why: Improve classifier recall on underrepresented classes.  
When: Only after temporal split on training set to avoid leakage.  
Relevance: Could alter class balance affecting GA fitness comparability if misapplied.  
Example:
```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X_train, y_train)
```
</details>

<details><summary><strong>class imbalance</strong></summary>
Purpose: Disproportion of samples per target class (often Sideways dominating).  
Usage: Monitored via value_counts in each temporal split.  
Why: Imbalance skews accuracy; F1-weighted & stability help mitigate.  
Relevance: GA must not overfit dominant class; stability metric penalizes volatility in minority class predictions.  
Example: {2: 60%, 1: 20%, 0: 20%}
</details>

<details><summary><strong>undersampling</strong></summary>
Purpose: Reduce majority class samples.  
Usage: Not currently applied (would reduce temporal continuity).  
Why: Avoid losing sequence information; risk of removing regimes.  
Relevance: Discouraged for time series; prefer model weighting.  
Example: Dropping random Sideways rows (not used).
</details>

<details><summary><strong>oversampling</strong></summary>
Purpose: Duplicate or synthesize minority samples.  
Usage: Candidate via SMOTE on training only.  
Why: Boost signal for minority movements (Up/Down).  
Relevance: Must be restricted to training to avoid target leakage across time.  
Example: Simple random oversampling vs SMOTE.
</details>

<details><summary><strong>bootstrapping</strong></summary>
Purpose: Resampling with replacement to estimate variability.  
Usage: Potential extension for confidence intervals on metrics.  
Why: Quantify uncertainty of GA-selected subset performance.  
Relevance: Could enhance run_summary with interval estimates.  
Example: 100 bootstrap draws of test predictions → F1 CI.
</details>

<details><summary><strong>noisy data</strong></summary>
Purpose: High-frequency random fluctuations unrelated to signal.  
Usage: Smoothed by moving averages, volatility normalization.  
Why: Reducing noise improves fitness stability.  
Relevance: GA indirectly filters noise by preferring stable cross-validation performance.  
Example: Tick-volume spikes without price follow-through.
</details>

<details><summary><strong>unstable data</strong></summary>
Purpose: Data whose statistical properties shift rapidly (non-stationary).  
Usage: Observed via volatility clustering; addressed with adaptive indicators (AMA, FDI).  
Why: Static features degrade; adaptive engineered ones maintain relevance.  
Relevance: cv_stability explicitly rewards robustness under instability.  
Example: Sudden regime shift during macro announcements.
</details>

<details><summary><strong>heteroskedasticity</strong></summary>
Purpose: Time-varying variance in residuals or returns.  
Usage: Implicit in ATR, rolling std; target thresholds scale by recent volatility.  
Why: Normalizing by local variance avoids mislabeling minor moves in calm periods vs major moves in volatile ones.  
Relevance: Dynamic thresholds produce more balanced actionable classes.  
Example: Rolling std spike → wider classification thresholds.
</details>

## Time Series Concepts & Validation

<details><summary><strong>time-series aware splitting</strong></summary>
Purpose: Method of dividing data into train/validation/test while preserving chronological order.  
How/When: Applied before modeling and GA so future samples never influence earlier model fits.  
Where: In notebooks (time-series split section) and inside GA (TimeSeriesSplit in fitness).  
Why: Prevents leakage by avoiding shuffling typical in iid tasks.  
Relevance: Fitness (= accuracy + stability − penalty) would be inflated if random splits were used.  
Example: Use first 80% train, next 10% validation, last 10% test (chronological).
</details>

<details><summary><strong>temporal order</strong></summary>
Purpose: Natural chronological sequencing of observations.  
How/When: Maintained in indexing (Datetime index) and all splits.  
Why: Price depends on prior states; reordering breaks causal structure.  
Relevance: GA evaluation assumes forward-only information flow.  
Example: df.sort_index() before rolling indicators.
</details>

<details><summary><strong>training set</strong></summary>
Purpose: Subset used to fit scalers, models, and GA candidate feature subsets.  
When: First (earliest) segment after filtering period.  
Why: Establishes parameter estimates without peeking ahead.  
Relevance: StandardScaler fit and GA CV folds draw only from this range.  
Example: X_train = X.iloc[:int(0.8*n)].
</details>

<details><summary><strong>validation set</strong></summary>
Purpose: Interim segment for hyperparameter tuning / early comparison.  
When: Immediately follows training window.  
Why: Guides model selection without touching final test.  
Relevance: GridSearchCV merges train+val in some steps (documented) but original val stats still compared.  
Example: Middle 10% of timeline.
</details>

<details><summary><strong>test set</strong></summary>
Purpose: Final untouched segment for unbiased performance reporting.  
When: Last chronological slice.  
Why: Simulates unseen future market conditions.  
Relevance: Baseline vs GA-enhanced metrics reported on this partition.  
Example: Last 10% rows after engineering.
</details>

<details><summary><strong>TimeSeriesSplit (usage/concept)</strong></summary>
Purpose: Expanding-window cross-validation iterator preserving order.  
How: Successive folds: train = [:t_k], test = (t_k : t_{k+1}].  
Why: Measures stability (cv_stability) under temporal drift.  
Relevance: Core to GA fitness; prevents optimistic variance estimates.  
Example:
```python
cv = TimeSeriesSplit(n_splits=5)
for tr, te in cv.split(X):
    model.fit(X[tr], y[tr])
```
</details>

<details><summary><strong>walk-forward validation</strong></summary>
Purpose: Evaluate model by iteratively moving the training window forward and predicting the next block.  
When: Advanced evaluation / production simulation.  
Why: Mimics live re-training cadence.  
Relevance: Could replace or complement TimeSeriesSplit for more granular robustness analysis.  
Example: Retrain every 50 bars, predict next 10.
</details>

<details><summary><strong>rolling-window CV</strong></summary>
Purpose: Sliding fixed-length train window plus forward test window (vs expanding).  
Why: Limits concept drift accumulation from very old data.  
Relevance: Alternative if long histories reduce relevance for fast regimes.  
Example: Train last 2 months → predict next week repeatedly.
</details>

<details><summary><strong>backtesting</strong></summary>
Purpose: Simulated historical strategy or model performance using only past info at each step.  
Why: Validate predictive power before deployment.  
Relevance: Model classification outputs could feed a rule engine for later PnL evaluation.  
Example: Iterate rows; update signals; record hypothetical returns.
</details>

<details><summary><strong>lookahead bias</strong></summary>
Purpose: Error where future data influences training or feature computation.  
How Avoided: Shifted targets, truncating last horizon rows, strictly chronological splits.  
Relevance: GA-selected features must not encode future moves (e.g., forward-filled future values).  
Example: Removing last PREDICTION_HORIZON rows after creating Target.
</details>

<details><summary><strong>data leakage</strong></summary>
Purpose: Any unintended information from outside the training slice leaking into model fitting.  
Forms: Scaling on full dataset, using post-period statistics, overlapping windows across folds.  
Relevance: Would inflate fitness causing poor live generalization.  
Example: Fitting StandardScaler only on X_train prevents leakage.
</details>

<details><summary><strong>stationarity</strong></summary>
Purpose: Statistical property where distribution (mean/variance) is time-invariant.  
Why: Some algorithms / tests assume stationarity; volatility-based thresholds adapt when not.  
Relevance: Indicators (returns, z-scores) partially stabilize series for learning.  
Example: Returns often closer to stationary than raw prices.
</details>

<details><summary><strong>non-stationarity</strong></summary>
Purpose: Changing distribution over time (drift, regime shifts).  
Why: Drives model decay; motivates adaptive features (AMA, FDI) and stability metric.  
Relevance: cv_stability rewards subsets less sensitive to shifts.  
Example: Volatility spike during macro event alters variance.
</details>

<details><summary><strong>time-windows</strong></summary>
Purpose: Fixed lookback spans for rolling calculations (e.g., 20, 50).  
Why: Encode recent context; trade-off responsiveness vs noise.  
Relevance: GA may favor combinations of complementary window lengths.  
Example: SMA_20 vs SMA_50 crossover.
</details>

<details><summary><strong>noisy data</strong></summary>
Purpose: High-frequency randomness obscuring signal.  
Why: Reduces predictive clarity; mitigated with smoothing, volatility normalization.  
Relevance: GA implicitly filters features whose noise lowers stability.  
Example: Raw tick-volume spikes not followed by price move.
</details>

<details><summary><strong>unstable data</strong></summary>
Purpose: Rapidly shifting statistical structure (regime changes).  
Why: Causes model drift; adaptive engineered features aim to track it.  
Relevance: Stability component penalizes volatile CV scores caused by instability.  
Example: Trend → range transition.
</details>

<details><summary><strong>hetroskedasticity</strong></summary>
Purpose: Misspelling often seen for heteroskedasticity.  
Relevance: Documented to clarify correct term; same implications.  
Example: Replace with heteroskedasticity in analysis scripts.
</details>

<details><summary><strong>heteroskedasticity</strong></summary>
Purpose: Time-varying variance in residuals/returns.  
Why: Affects thresholding for targets and feature scaling impact.  
Relevance: Dynamic volatility thresholds (std * MULTIPLIER) reduce class imbalance.  
Example: Increased ATR_14 widens Up/Down thresholds.
</details>

<details><summary><strong>detrending</strong></summary>
Purpose: Remove long-term drift to isolate mean-reverting or high-frequency structure.  
How: Differences, moving-average subtraction, log returns.  
Relevance: Using returns/log returns serves as implicit detrending aiding model learning.  
Example: df['Log_Return'] = np.log(Close/Close.shift(1))
</details>

<details><summary><strong>seasonality</strong></summary>
Purpose: Repeating temporal patterns (hour-of-day, day-of-week).  
Why: Can inform engineered cyclical features.  
Relevance: Not yet exploited; potential future categorical time feature for GA.  
Example: Add sin(hour/24*2π).
</details>

<details><summary><strong>cointegration</strong></summary>
Purpose: Stable linear combination between non-stationary series.  
Why: Enables spread trading / mean reversion signals.  
Relevance: Future extension if multiple FX pairs added (multi-asset feature creation).  
Example: EURUSD & GBPUSD spread feature.
</details>

<details><summary><strong>unit root</strong></summary>
Purpose: Characteristic of a stochastic process with persistence (random walk).  
Why: Presence implies non-stationarity; motivates differencing/log returns.  
Relevance: Price series likely has unit root; returns mitigate.  
Example: Augmented Dickey-Fuller test on Close.
</details>

<details><summary><strong>differencing</strong></summary>
Purpose: Transform x_t → x_t - x_{t-1} to remove unit root/trend.  
Why: Achieve (approx) stationarity for modeling.  
Relevance: Percentage_Change and Log_Return are scaled differences.  
Example:
```python
df['Diff_Close'] = df['Close'].diff()
```
</details>

<details><summary><strong>slippage</strong></summary>
Purpose: Execution price deviation from intended price.  
Why: Reduces real-world strategy returns vs paper predictions.  
Relevance: Not modeled yet; future backtest adjustment for realistic metrics.  
Example: Adjust predicted trade entry by -0.5 pip.
</details>

<details><summary><strong>transaction costs</strong></summary>
Purpose: Fees/spread/commission applied per trade.  
Why: Converts raw accuracy into net profitability.  
Relevance: Could be subtracted post-classification to assess economic value of GA improvements.  
Example: Net_return = gross_return - spread_cost.
</details>

<details><summary><strong>overnight gap</strong></summary>
Purpose: Price jump between last bar of one session and first of next.  
Why: Introduces discontinuities affecting indicators.  
Relevance: Hourly FX often continuous, but commodities / equities data may need gap features.  
Example: Gap_Pct used as engineered feature.
</details>

<details><summary><strong>latency</strong></summary>
Purpose: Delay between signal generation and execution.  
Why: Degrades realized performance for fast regimes.  
Relevance: Model not latency-aware; future deployment metric.  
Example: 200 ms delay shifts fill price during volatile spikes.
</details>

<details><summary><strong>tick vs candle aggregation</strong></summary>
Purpose: Tick = every trade/quote; Candle = aggregated OHLCV over interval.  
Why: Aggregation smooths noise but may hide microstructure signals.  
Relevance: Current workflow uses hourly candles; engineered tick-volume proxies approximate finer activity.  
Example: Converting ticks to 1H OHLCV before feature engineering.
</details>

## Descriptive Statistics & Distributions

<details><summary><strong>count</strong></summary>
Purpose: Number of non-missing observations in a series/column.  
How/When: Used during EDA (df.describe()) and after dropping NaNs from engineered indicators to verify sample sufficiency for TimeSeriesSplit folds.  
Where: Validation of feature completeness before GA initialization.  
Why: Too few rows after filtering/engineering can make CV unstable (folds too small).  
Relevance: Ensures enough chronological samples for 5-fold TimeSeriesSplit (each fold must have train/test size > 0).  
Example:  
```python
valid_rows = X_train.shape[0]  # count
```
</details>

<details><summary><strong>mean</strong></summary>
Purpose: Arithmetic average; central tendency of a feature.  
How/When: Used implicitly by StandardScaler (stores mean per feature); inspected in descriptive stats to detect drift (comparison across runs).  
Why: Centering improves model convergence (esp. logistic/XGBoost internal splits benefit from normalized features).  
Relevance: Stable mean across train/val/test reduces leakage suspicion.  
Example: `feature_means = X_train.mean()`
</details>

<details><summary><strong>std</strong></summary>
Purpose: Standard deviation (dispersion) of values.  
How/When: Used explicitly in volatility features, Bollinger Bands, z-scores, dynamic target thresholds (rolling std).  
Why: Scales thresholds for Up/Down classification to adapt to regime volatility (reduces class imbalance).  
Relevance: Drives Target generation; feeds stability fitness indirectly.  
Example: `vol = close.pct_change().rolling(20).std()`
</details>

<details><summary><strong>min</strong></summary>
Purpose: Minimum observed value in a series/window.  
How/When: Rolling minima in stochastic oscillator, high/low range, and gap analysis.  
Why: Bounds help construct oscillators (e.g., %K).  
Relevance: Feature diversity: extremal values help classify compression vs expansion regimes.  
Example: `low_roll = low.rolling(14).min()`
</details>

<details><summary><strong>max</strong></summary>
Purpose: Maximum observed value in a series/window.  
Usage mirrors min (stochastic, range, volatility extremes).  
Why: Captures recent extrema for momentum/mean reversion signals.  
Relevance: Essential for normalized indicators (e.g., %B).  
Example: `high_roll = high.rolling(14).max()`
</details>

<details><summary><strong>iqr</strong></summary>
Purpose: Interquartile range (Q3 - Q1); robust dispersion metric.  
How/When: Optional robust scaling / outlier diagnostics in pre-engineering QA.  
Why: Less sensitive to spikes than std; helps decide clipping thresholds (tick volume).  
Relevance: Potential improvement for stability if volatility outliers inflate std-based features.  
Example:  
```python
q1, q3 = s.quantile([0.25,0.75])
iqr = q3 - q1
```
</details>

<details><summary><strong>skewness</strong></summary>
Purpose: Asymmetry of distribution (positive = right-tail).  
How/When: Reported for returns to understand tail risk bias.  
Why: Non-zero skew affects model calibration & class threshold tuning.  
Relevance: Helps justify using balanced metrics (F1) vs raw accuracy.  
Example: `returns.skew()`
</details>

<details><summary><strong>kurtosis</strong></summary>
Purpose: Tail/heaviness & peakness (excess > 0 ⇒ fat tails).  
How/When: Evaluated on returns distribution (EDA).  
Why: Fat tails imply higher rare-move probability; motivates volatility-adjusted targets.  
Relevance: Supports design choice of volatility-based dynamic thresholds instead of fixed %.  
Example: `returns.kurt()`
</details>

<details><summary><strong>z-score</strong></summary>
Purpose: Standardized value: (x - mean) / std.  
How/When: Used to form Price_Z_Score, Returns_Z_Score, anomaly detection.  
Why: Normalization enables comparability across time and features with different scales.  
Relevance: Enriches GA search with scale-free features improving stability metric.  
Example: `df['Price_Z_Score_20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()`
</details>

<details><summary><strong>distribution</strong></summary>
Purpose: Overall shape (frequency) of variable values.  
How/When: Visualized (hist/kde) for returns & engineered indicators.  
Why: Identifies transformations (log, z-score) to stabilize variance.  
Relevance: Guides feature preprocessing choices that affect GA fitness consistency.  
Example: `sns.histplot(returns, kde=True)`
</details>

<details><summary><strong>normal distribution</strong></summary>
Purpose: Gaussian reference model (used in z-score assumptions & some statistical tests).  
Why: Deviations (fat tails, skew) inform risk-aware feature engineering.  
Relevance: Non-normality justifies robustness emphasis (stability term in fitness).  
Example: Q-Q plot vs normal to show heavy tails in returns.
</details>

<details><summary><strong>left skewed</strong></summary>
Purpose: Distribution with longer/larger left tail (negative skew).  
Why: Indicates more extreme negative returns; risk considerations for thresholds.  
Relevance: May motivate asymmetric class boundaries (future extension).  
Example: If returns.skew() < 0.
</details>

<details><summary><strong>right skewed</strong></summary>
Purpose: Distribution with longer right tail (positive skew).  
Why: More frequent large upward moves; can bias class balance.  
Relevance: Helps interpret imbalance in Up vs Down classes after target creation.  
Example: returns.skew() > 0.
</details>

<details><summary><strong>autocorrelation</strong></summary>
Purpose: Correlation of a series with its lagged self.  
How/When: Checked for returns & absolute returns (volatility clustering).  
Why: Presence suggests momentum or mean reversion features worth including.  
Relevance: GA may prefer lag-derived indicators when autocorrelation > 0.  
Example: `returns.autocorr(lag=1)`
</details>

<details><summary><strong>correlation</strong></summary>
Purpose: Linear relationship (Pearson) between two variables.  
How/When: Used in indicator correlation heatmaps to spot redundancy.  
Why: High multicollinearity inflates model variance; GA implicitly prunes redundant features.  
Relevance: Justifies feature subset optimization vs full set.  
Example: `df[features].corr()`
</details>

<details><summary><strong>VIF (Variance Inflation Factor)</strong></summary>
Purpose: Quantifies multicollinearity: VIF = 1 / (1 - R²).  
How/When: Optional diagnostic pre-GA to flag redundant features.  
Why: High VIF (>10) indicates instability risk in linear/logistic models.  
Relevance: Though tree models are robust, removing extreme VIF features can reduce GA search space.  
Example:  
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = variance_inflation_factor(X.values, i)
```
</details>

<details><summary><strong>condition number</strong></summary>
Purpose: Ratio of largest to smallest singular value (matrix stability).  
How/When: Optional numeric stability check on standardized feature matrix.  
Why: Large values indicate near-linear dependencies.  
Relevance: High condition number may degrade ANN/logistic (if added later).  
Example: `np.linalg.cond(X_train_scaled.values)`
</details>

<details><summary><strong>confidence interval</strong></summary>
Purpose: Range likely containing true metric (e.g., accuracy) at chosen confidence level.  
How/When: Future enhancement via bootstrap on test predictions.  
Why: Quantifies uncertainty of GA improvement claims.  
Relevance: Adds statistical rigor to baseline vs enhanced comparison.  
Example: Bootstrap 1000 F1 scores → 95% CI.
</details>

<details><summary><strong>p-value</strong></summary>
Purpose: Probability (under null) of observing result as extreme.  
How/When: Could test whether enhanced model improvement is statistically significant.  
Why: Avoid overinterpreting random uplifts.  
Relevance: Supports evidence that GA selection materially improves performance.  
Example: Paired test on per-fold accuracies.
</details>

<details><summary><strong>hypothesis testing (t-test, chi-square, Mann-Whitney)</strong></summary>
Purpose: Formal tests comparing means (t), independence (chi-square), or distributions (Mann-Whitney non-parametric).  
How/When: Potential evaluation of feature distributions pre/post selection or class distribution shifts.  
Why: Validate that resampling or engineering does not distort structure adversely.  
Relevance: Could confirm engineered AMA/FDI shift discriminative power vs baseline features.  
Example: `scipy.stats.mannwhitneyu(f_old, f_new)`
</details>

<details><summary><strong>multiple testing correction (Bonferroni, FDR)</strong></summary>
Purpose: Adjust p-values when many hypotheses tested to control Type I error.  
How/When: If running many indicator significance tests.  
Why: Prevent false discoveries inflating feature importance narrative.  
Relevance: Ensures statistical discipline if adding filter-based pre-selection before GA.  
Example: `adjusted_p = raw_p * m  # Bonferroni`
</details>

## Returns & Price Transformations

<details><summary><strong>Percentage_Change</strong></summary>
Purpose: Raw simple return for one bar: (Close_t - Close_{t-1}) / Close_{t-1}.  
How/When: Computed immediately after loading & sorting OHLCV before any volatility or target engineering.  
Where: Stored as feature_df['Percentage_Change'] and reused for volatility (rolling std), target thresholds, distribution diagnostics, and higher‑order stats (skew/kurtosis).  
Why: Scale‑invariant measure of directional magnitude; forms the base series for volatility, z-scores, cumulative returns, and classification target logic.  
Relevance: Target generation (dynamic thresholds) uses the rolling std of this series; GA feature selection may keep or drop it and its derivatives.  
Example:
```python
df['Percentage_Change'] = df['Close'].pct_change()
```
Edge Notes: First value is NaN (dropped later); susceptible to gaps—can be paired with Gap_Pct for context.
</details>

<details><summary><strong>Log_Percentage_Change</strong></summary>
Purpose: Log return ln(Close_t / Close_{t-1}).  
How/When: Computed alongside Percentage_Change during baseline feature engineering.  
Where: feature_df['Log_Return'] (alias).  
Why: Additivity over time (log returns sum), symmetry for small moves, reduces compounding bias in modeling.  
Relevance: Alternative return representation; sometimes yields more stable variance improving cv_stability in fitness.  
Example:
```python
df['Log_Percentage_Change'] = np.log(df['Close'] / df['Close'].shift(1))
```
Tip: For very small returns log and simple returns are numerically close; GA may choose one.
</details>

<details><summary><strong>Price_Z_Score</strong></summary>
Purpose: Standardized deviation of price from its rolling mean: (Close - MA(window)) / rolling_std(window).  
How/When: Created after rolling means & std are available (e.g., with 20 / 50 period windows).  
Where: Columns like Price_Z_Score_20, Price_Z_Score_50 in engineered features.  
Why: Normalizes price regime (overextension vs mean) making cross-period comparability easier.  
Relevance: Helps GA capture mean-reversion / breakout contexts without raw price scale.  
Example:
```python
w = 20
roll_mean = df['Close'].rolling(w).mean()
roll_std  = df['Close'].rolling(w).std()
df[f'Price_Z_Score_{w}'] = (df['Close'] - roll_mean) / roll_std
```
Caution: Initial window rows NaN → dropped prior to splits to avoid leakage/misalignment.
</details>

<details><summary><strong>Returns_Z_Score</strong></summary>
Purpose: Standardized recent return relative to its rolling return distribution.  
How/When: After Percentage_Change is computed; use rolling mean/std on returns.  
Where: Columns Returns_Z_Score_{period}.  
Why: Highlights unusually large moves (potential volatility regime shifts) beyond raw return magnitude.  
Relevance: Aids GA in selecting shock / anomaly indicators that improve stability when combined with trend features.  
Example:
```python
r = df['Percentage_Change']
w = 20
df[f'Returns_Z_Score_{w}'] = (r - r.rolling(w).mean()) / r.rolling(w).std()
```
Interpretation: |Z| > 2 often signals elevated volatility or potential turning points.
</details>

<details><summary><strong>Returns_Skewness_{period}</strong></summary>
Purpose: Rolling third standardized moment of returns over a window; measures asymmetry.  
How/When: After returns computed; rolling(window).skew() for each defined period.  
Where: Returns_Skewness_20, Returns_Skewness_50 etc.  
Why: Captures directional tail bias—can hint at persistent drift or crash risk.  
Relevance: Provides regime characterization features; GA may combine with volatility to refine class boundaries indirectly.  
Example:
```python
w = 50
df[f'Returns_Skewness_{w}'] = df['Percentage_Change'].rolling(w).skew()
```
Note: Unstable for very short windows—choose windows ensuring enough samples.
</details>

<details><summary><strong>Returns_Kurtosis_{period}</strong></summary>
Purpose: Rolling excess kurtosis (peakedness / tail heaviness) of returns.  
How/When: Similar timing to skewness; rolling(window).kurt().  
Where: Returns_Kurtosis_{period}.  
Why: High kurtosis ⇒ fat tails / jump risk; influences adaptive threshold rationale.  
Relevance: Helps GA select features that signal regime transitions (tail clustering) improving cv_stability.  
Example:
```python
w = 50
df[f'Returns_Kurtosis_{w}'] = df['Percentage_Change'].rolling(w).kurt()
```
Interpretation: >0 indicates heavier tails than normal; may precede volatility expansion.
</details>

<details><summary><strong>Open_Close_Diff</strong></summary>
Purpose: Intra-bar directional body: Close - Open.  
How/When: Early in candle feature block.  
Where: feature_df['Open_Close_Diff'].  
Why: Captures net directional movement ignoring wicks; signals bullish/bearish pressure intensity.  
Relevance: Combines well with range metrics (Body_To_Range_Ratio). A low-variance but directional feature GA may retain.  
Example:
```python
df['Open_Close_Diff'] = df['Close'] - df['Open']
```
Positive: bullish bar; negative: bearish bar.
</details>

<details><summary><strong>High_Low_Diff</strong></summary>
Purpose: Full candle range: High - Low.  
How/When: With other basic price transforms.  
Where: feature_df['High_Low_Diff'] (alias Candle_Range).  
Why: Proxy for realized intrabar volatility.  
Relevance: Used in ATR, volatility normalization, body/range ratios feeding pattern recognition; GA often keeps one of related range features.  
Example:
```python
df['High_Low_Diff'] = df['High'] - df['Low']
```
Large values may precede trend continuations or reversals depending on context.
</details>

<details><summary><strong>High_Close_Ratio</strong></summary>
Purpose: High / Close ratio (dimensionless).  
How/When: After base OHLC columns accessible.  
Where: feature_df['High_Close_Ratio'].  
Why: Indicates how far price closed from session high (closeness to upper extreme).  
Relevance: Helps classify exhaustion vs strength; complementary to Upper_Shadow.  
Example:
```python
df['High_Close_Ratio'] = df['High'] / df['Close']
```
Interpretation: ~1 implies close near high; >1.001 suggests fade off highs.
</details>

<details><summary><strong>Low_Close_Ratio</strong></summary>
Purpose: Low / Close ratio.  
How/When: Together with High_Close_Ratio for symmetry.  
Where: feature_df['Low_Close_Ratio'].  
Why: Closeness to the low; potential reversal if clustering near lows then momentum shifts.  
Relevance: Combined with High_Close_Ratio can encode candle body positioning; GA may select only one ratio or a derived combination.  
Example:
```python
df['Low_Close_Ratio'] = df['Low'] / df['Close']
```
Values close to 1 indicate close near low.
</details>

<details><summary><strong>Open_Close_Ratio</strong></summary>
Purpose: Open / Close ratio capturing direction and magnitude in relative form.  
How/When: After OHLC ingestion.  
Where: feature_df['Open_Close_Ratio'].  
Why: Normalizes Open_Close_Diff by close level to reduce scale drift across price regimes.  
Relevance: Provides scale-stable directional signal aiding stability across long backtests.  
Example:
```python
df['Open_Close_Ratio'] = df['Open'] / df['Close']
```
>1 bearish bar; <1 bullish bar.
</details>

<details><summary><strong>log transform</strong></summary>
Purpose: Apply natural log (or log1p) to compress skewed distributions (e.g., Volume, Tick_Volume).  
How/When: During feature engineering (e.g., TickVol_Log1p).  
Where: feature_df['TickVol_Log1p'] etc.  
Why: Stabilizes variance, mitigates extreme outliers impacting scaler and model splits.  
Relevance: Improves cv_stability by reducing fold-wise variance spikes.  
Example:
```python
df['TickVol_Log1p'] = np.log1p(df['Tick_Volume'])
```
Note: Use log1p to safely handle zeros.
</details>

<details><summary><strong>cumulative returns</strong></summary>
Purpose: Compounded growth series (1 + r).cumprod(); tracks equity curve of naive holding.  
How/When: In EDA or evaluation (drawdown calc). Not usually a direct model input (can leak future).  
Where: Temporary series in analysis cells.  
Why: Assesses regime quality, drawdowns, volatility clustering context.  
Relevance: Guides interpretation of target distributions and performance improvements post GA (qualitative check).  
Example:
```python
cum_ret = (1 + df['Percentage_Change']).cumprod()
```
Avoid feeding directly into GA to prevent forward path leakage.
</details>

<details><summary><strong>drawdown</strong></summary>
Purpose: Measure of decline from running peak: equity / rolling_max - 1.  
How/When: Post cumulative return computation during EDA.  
Where: Derived series for risk diagnostics (not a feature for selection).  
Why: Evaluates tail and risk exposure; informs if enhanced model may reduce adverse sequences (future extension).  
Relevance: Helps validate if GA-selected features potentially lower volatility in hypothetical strategy layer.  
Example:
```python
equity = (1 + df['Percentage_Change']).cumprod()
dd = equity / equity.cummax() - 1
max_dd = dd.min()
```
Not included in training to avoid label leakage.
</details>

<details><summary><strong>rolling returns</strong></summary>
Purpose: Aggregated return over a fixed lookback window (e.g., N-period forward/backward).  
How/When: Could be computed as (Close / Close.shift(N) - 1) or rolling sum of log returns; used in target derivation (future horizon returns).  
Where: Intermediate series for target (future pct change over PREDICTION_HORIZON).  
Why: Captures multi-bar momentum vs single-bar noise.  
Relevance: Directly underpins Target classification (future movement vs dynamic volatility thresholds) ensuring label coherence.  
Example (future 5-bar return used for Target):
```python
h = 5
future_ret = df['Close'].pct_change(h).shift(-h)
```
Caution: Always shift forward (negative shift) then truncate last h rows to avoid lookahead bias.
</details>

## Candles & Price Features

<details><summary><strong>Candle_Range</strong></summary>
Purpose: Total intrabar volatility span: High - Low for a single candle (hour in this project).  
How/When/Where: Computed early in baseline feature engineering right after OHLC data is loaded and sorted. Stored as High_Low_Diff / Candle_Range.  
Why: Captures realized movement magnitude independent of direction; foundation for ATR, volatility context, and pattern ratios.  
Relevance (project): GA may retain Candle_Range or a derived volatility proxy; helps distinguish quiet vs expansion phases affecting target class transitions (Sideways vs Up/Down). Stability improves when features encode regime amplitude.  
Example:
```python
feature_df['Candle_Range'] = feature_df['High'] - feature_df['Low']
```
</details>

<details><summary><strong>Body_Size</strong></summary>
Purpose: Net directional move within the candle body: abs(Close - Open).  
How/When/Where: Calculated with other basic price transforms before advanced indicators.  
Why: Measures directional conviction stripped of wicks; small bodies with large range imply indecision (doji-type).  
Relevance: Useful in combination ratios (Body_To_Range_Ratio) for pattern discrimination; GA can choose compact encodings of price structure that aid classification stability.  
Example:
```python
feature_df['Body_Size'] = (feature_df['Close'] - feature_df['Open']).abs()
```
</details>

<details><summary><strong>Upper_Shadow</strong></summary>
Purpose: Distance from the higher of Open/Close to High (top wick length).  
How/When/Where: Created in candle anatomy block alongside Body_Size.  
Why: Indicates intrabar rejection above the real body (potential resistance or exhaustion).  
Relevance: Helps differentiate failed breakouts vs continuation bars; can indirectly improve Up vs Sideways discrimination in GA-selected subsets.  
Example:
```python
feature_df['Upper_Shadow'] = feature_df['High'] - feature_df[['Open','Close']].max(axis=1)
```
</details>

<details><summary><strong>Lower_Shadow</strong></summary>
Purpose: Distance from Low to the lower of Open/Close (bottom wick length).  
How/When/Where: Same stage as Upper_Shadow.  
Why: Reflects buying pressure rejecting lower prices (potential support).  
Relevance: Complements Upper_Shadow for asymmetry signals; GA may select one or the ratio interplay to capture reversal probability affecting Target movement horizon.  
Example:
```python
feature_df['Lower_Shadow'] = feature_df[['Open','Close']].min(axis=1) - feature_df['Low']
```
</details>

<details><summary><strong>Body_To_Range_Ratio</strong></summary>
Purpose: Normalized conviction metric: Body_Size / Candle_Range.  
How/When/Where: Calculated after Body_Size and Candle_Range are available.  
Why: Scale-invariant measure of directional efficiency (close-to-open move vs total volatility).  
Relevance: High ratio ⇒ directional bar; Low ratio ⇒ indecision or mean-reverting noise. Helps GA prefer parsimonious normalized features robust across volatility regimes (improves cv_stability).  
Example:
```python
feature_df['Body_To_Range_Ratio'] = feature_df['Body_Size'] / feature_df['Candle_Range']
```
Edge Note: Guard against division by zero if High == Low (rare in liquid hourly FX/commodities).
</details>

<details><summary><strong>Candle</strong></summary>
Purpose: A single OHLCV bar representing aggregated trades over the chosen period (1H).  
How/When/Where: Fundamental data unit loaded from raw_data and filtered to the analysis window.  
Why: Aggregation smooths tick noise while preserving broad directional and volatility structure needed for indicator computation.  
Relevance: Every engineered feature derives ultimately from candle sequences; understanding anatomy (body, shadows) informs creation of composite candle metrics for GA selection.  
Example Fields: Open, High, Low, Close, Volume (or Tick_Volume).  
</details>

<details><summary><strong>Bullish</strong></summary>
Purpose: Qualitative label for a candle closing above its open (Close > Open).  
How/When/Where: Implicitly used when interpreting Open_Close_Diff > 0; can be encoded as (Close > Open).astype(int) if needed.  
Why: Directional bias per bar assists pattern formation (e.g., sequences of bullish bars).  
Relevance: May become a lightweight categorical feature if added; influences local momentum that impacts future return classification (Target).  
Example:
```python
feature_df['Bullish_Flag'] = (feature_df['Close'] > feature_df['Open']).astype(int)
```
</details>

<details><summary><strong>Bearish</strong></summary>
Purpose: Candle closing below its open (Close < Open).  
How/When/Where: Complement of Bullish classification; derivable without added storage.  
Why: Identifies downward directional bars contributing to short-term momentum or exhaustion patterns.  
Relevance: Potential binary signal for GA if included; interacts with volatility and shadow metrics in feature subset selection.  
Example:
```python
feature_df['Bearish_Flag'] = (feature_df['Close'] < feature_df['Open']).astype(int)
```
</details>

<details><summary><strong>Partial_Bar_Flag</strong></summary>
Purpose: Indicator that an hourly bar contains fewer ticks than expected (possible session boundary / early closure / data gap).  
How/When/Where: Created only when VOLUME_TYPE_TICKER is True using threshold (e.g., Tick_Volume < EXPECTED_TICKS_PER_HOUR).  
Why: Partial bars can distort volatility/indicator calculations (under-represent activity) and induce artificial NaNs or misleading ranges.  
Relevance: GA can down‑weight subsets sensitive to irregular sampling if this flag is included; improves robustness by explaining anomalies in feature behavior.  
Example:
```python
EXPECTED_TICKS_PER_HOUR = 60
feature_df['Partial_Bar_Flag'] = (feature_df['Tick_Volume'] < EXPECTED_TICKS_PER_HOUR).astype(int)
```
</details>

<details><summary><strong>wick</strong></summary>
Purpose: Generic term for candle shadow (upper or lower line beyond body).  
How/When/Where: Derived implicitly via Upper_Shadow and Lower_Shadow computations.  
Why: Shows intrabar price extremes rejected by the close; long wicks may signal reversals or failed breakouts.  
Relevance: Separating body vs wick intensifies pattern granularity; GA might choose shadow features over raw High/Low for parsimony and stability.  
Example Interpretation: Long lower wick + small body ⇒ potential bullish reversal under volatility normalization.  
</details>

<details><summary><strong>engulfing pattern (concept)</strong></summary>
Purpose: Two-candle reversal formation where the second candle’s body fully engulfs the prior candle’s body (bullish: down then larger up body; bearish: up then larger down body).  
How/When/Where: Not yet explicitly engineered; could be added as a binary feature scanning consecutive candles after baseline features.  
Why: Encodes abrupt momentum shift & liquidity absorption, potentially predictive of short-term directional follow‑through.  
Relevance: Future feature engineering extension—binary engulfing flags may enrich GA search space with discrete pattern content beyond continuous indicators, aiding class separation for Up/Down vs Sideways.  
Example (bullish engulfing detection sketch):
```python
prev = feature_df.shift(1)
bullish_engulf = (
    (prev['Close'] < prev['Open']) &  # previous bearish
    (feature_df['Close'] > feature_df['Open']) &  # current bullish
    (feature_df['Close'] >= prev['Open']) &
    (feature_df['Open'] <= prev['Close'])
).astype(int)
```
</details>

## Moving Averages & Trend Indicators

<details><summary><strong>SMA / <code>SMA_{period}</code> / <code>SMA_Dist_{period}</code></strong></summary>
Purpose: Simple Moving Average (SMA) is the unweighted mean of the last N closes; SMA_{period} denotes window length; SMA_Dist_{period} is the % distance of current Close from its SMA (momentum / mean‑reversion context).  
How/When/Where: Computed during baseline feature engineering after OHLC is cleaned. Created for periods [5,10,20,50,100]. Distance variant: (Close - SMA)/SMA * 100 stored as SMA_Dist_{period}.  
Why: Smooths noise (reduces high‑frequency variance) while retaining directional drift. Distance converts raw deviation to scale‑free feature aiding comparison across price levels.  
Relevance (project): GA can choose between raw level (SMA_20), deviation (SMA_Dist_20), or combinations for regime and overextension detection—supports Target classification stability (cv_stability).  
Example:
```python
period = 20
df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
df[f'SMA_Dist_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}'] * 100
```
Edge: Initial (period-1) rows NaN → dropped before splitting to avoid leakage via inconsistent alignment.
</details>

<details><summary><strong>EMA / <code>EMA_{period}</code> / <code>EMA_Dist_{period}</code></strong></summary>
Purpose: Exponential Moving Average applies exponentially decaying weights (recent prices emphasized). EMA_Dist_{period} expresses % divergence from adaptive baseline.  
How/When/Where: Generated alongside SMAs for identical periods; uses pandas ewm(span=period, adjust=False).  
Why: Faster reaction to regime shifts than SMA; distance variant highlights momentum acceleration or exhaustion earlier.  
Relevance: Helps GA balance lag (SMA) vs responsiveness (EMA). Distance features often correlate with turning points → improved predictive F1 with fewer features (feature reduction).  
Example:
```python
p = 50
df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
df[f'EMA_Dist_{p}'] = (df['Close'] - df[f'EMA_{p}']) / df[f'EMA_{p}'] * 100
```
Note: High collinearity across overlapping periods—GA selection prunes redundancy.
</details>

<details><summary><strong>TEMA / <code>TEMA_{period}</code> / <code>TEMA_Dist_{period}</code></strong></summary>
Purpose: Triple Exponential Moving Average reduces lag further by combining single, double, triple EMAs: TEMA = 3*EMA1 - 3*EMA2 + EMA3. Distance variant normalizes deviation.  
How/When/Where: Built for periods [10,20,50] after EMA block.  
Why: Enhances responsiveness while smoothing; captures emerging trend shifts earlier than SMA/EMA alone.  
Relevance: Offers alternative trend proxy; GA may retain a single TEMA_Dist period instead of multiple EMA distances, improving sparsity.  
Example:
```python
p = 20
ema1 = close.ewm(span=p, adjust=False).mean()
ema2 = ema1.ewm(span=p, adjust=False).mean()
ema3 = ema2.ewm(span=p, adjust=False).mean()
df[f'TEMA_{p}'] = 3*ema1 - 3*ema2 + ema3
df[f'TEMA_Dist_{p}'] = (close - df[f'TEMA_{p}']) / df[f'TEMA_{p}'] * 100
```
Tip: Distance > 0 and rising + narrowing volatility often precedes sustained Up (Target=1).
</details>

<details><summary><strong><code>SMA_5_10_Cross</code></strong></summary>
Purpose: Binary flag = 1 when short SMA (5) crosses above medium SMA (10); else 0.  
How/When/Where: Computed immediately after SMA series exist using prior bar states to avoid same-bar bias.  
Why: Encodes momentum regime shift with minimal information entropy vs raw values.  
Relevance: Compact categorical trend initiation signal; GA can favor cross flags over multiple moving averages, reducing dimensionality.  
Example:
```python
cross = (df['SMA_5'] > df['SMA_10']) & (df['SMA_5'].shift(1) <= df['SMA_10'].shift(1))
df['SMA_5_10_Cross'] = cross.astype(int)
```
</details>

<details><summary><strong><code>SMA_10_20_Cross</code></strong></summary>
Purpose: Binary flag for SMA_10 crossing above SMA_20 (short–intermediate trend alignment).  
How/When/Where: Same logic pattern as SMA_5_10_Cross, later window pair.  
Why: Captures slower trend confirmation (filters whipsaws of 5/10).  
Relevance: GA may choose one crossover granularity depending on stability contribution.  
Example:
```python
cross = (df['SMA_10'] > df['SMA_20']) & (df['SMA_10'].shift(1) <= df['SMA_20'].shift(1))
df['SMA_10_20_Cross'] = cross.astype(int)
```
</details>

<details><summary><strong><code>ADX_{period}</code></strong></summary>
Purpose: Average Directional Index quantifies trend strength (0–100) irrespective of direction over given period (e.g., 14).  
How/When/Where: Built after computing directional movement (+DM, -DM) and True Range; smoothed and averaged.  
Why: Distinguishes ranging (low ADX) from trending (high ADX) environments—affects reliability of momentum vs mean‑reversion features.  
Relevance: GA may retain ADX_14 to modulate interpretation of other indicators (interaction not explicit but implicit in model). Improves cv_stability by explaining variance in feature efficacy across regimes.  
Interpretation: ADX rising above ~20–25 indicates strengthening trend context.  
</details>

<details><summary><strong><code>Plus_DI_{period}</code></strong></summary>
Purpose: Positive Directional Indicator (+DI): 100 * smoothed +DM / ATR; measures bullish directional movement magnitude.  
How/When/Where: Intermediate calculation in ADX block; stored as feature for period (e.g., Plus_DI_14).  
Why: Provides directional bias intensity—not just net price change but directional efficiency.  
Relevance: Helps model distinguish sustained Up (Target=1) probability when +DI dominates -DI.  
Example (concept):
```python
plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
Plus_DI = 100 * smooth(plus_dm, period) / ATR
```
</details>

<details><summary><strong><code>Minus_DI_{period}</code></strong></summary>
Purpose: Negative Directional Indicator (−DI): 100 * smoothed −DM / ATR; bearish directional intensity.  
How/When/Where: Computed alongside Plus_DI.  
Why: Symmetric measure to gauge downside dominance.  
Relevance: Comparative relation (+DI vs −DI) influences DI_Diff and derived DMI signals chosen by GA for class separation (Down vs Sideways).  
</details>

<details><summary><strong><code>DI_Diff_{period}</code></strong></summary>
Purpose: Difference Plus_DI - Minus_DI; signed directional strength.  
How/When/Where: Calculated immediately after DI components.  
Why: Collapses two correlated indicators into a single signed feature (parsimony).  
Relevance: Often retained instead of both DI lines, reducing feature redundancy → better feature reduction percentage without accuracy loss.  
Example:
```python
df[f'DI_Diff_{p}'] = df[f'Plus_DI_{p}'] - df[f'Minus_DI_{p}']
```
Interpretation: Positive & rising ⇒ bullish bias; negative ⇒ bearish pressure.
</details>

<details><summary><strong><code>DMI_{period}</code></strong></summary>
Purpose: Directional Movement Index = |+DI - −DI| / (+DI + −DI) * 100; normalized directional dominance.  
How/When/Where: Added after DI lines to express relative separation.  
Why: Bounds directional separation to interpretable 0–100 scale, improving model calibration versus raw spread.  
Relevance: GA may prefer DMI_{period} over raw DI_Diff when scale normalization improves cross‑fold stability.  
Example:
```python
df[f'DMI_{p}'] = abs(df[f'Plus_DI_{p}'] - df[f'Minus_DI_{p}']) / \
                 (df[f'Plus_DI_{p}'] + df[f'Minus_DI_{p}']) * 100
```
</details>

<details><summary><strong>PSAR / <code>PSAR_Dist</code> / <code>PSAR_Bull</code></strong></summary>
Purpose: Parabolic SAR (Stop And Reverse) trails price to identify trend direction and potential reversal points. PSAR_Dist = % distance (Close - PSAR)/Close * 100; PSAR_Bull = binary (PSAR below Close).  
How/When/Where: Implemented in volume branch when VOLUME_TYPE_TICKER False; iterative loop accumulates SAR with acceleration factor.  
Why: Encodes dynamic trailing structure—useful for capturing transitions between trending and consolidating phases. Distance normalizes for price level; binary flag simplifies regime encoding.  
Relevance: Provides orthogonal trend persistence signal complementary to moving averages; GA might retain PSAR_Bull plus a single distance metric instead of several overlapping MA distances.  
Example (condensed):
```python
df['PSAR'] = psar(df)  # custom function
df['PSAR_Dist'] = (df['Close'] - df['PSAR']) / df['Close'] * 100
df['PSAR_Bull'] = (df['PSAR'] < df['Close']).astype(int)
```
Interpretation: Flip in PSAR_Bull from 0→1 often aligns with early Up classification candidate (future Target=1).
</details>

## MACD & Momentum Indicators

<details><summary><strong>MACD</strong></summary>
Purpose: Moving Average Convergence Divergence; momentum/trend-following indicator derived from the difference between two EMAs (fast 12, slow 26 by convention).  
How/When: Computed after EMAs are available during feature engineering.  
Where: Columns: MACD_Line, MACD_Signal, MACD_Histogram, MACD_CrossAbove, MACD_CrossBelow.  
Why: Captures shifts in momentum and trend acceleration/decay earlier than raw price or single EMAs.  
Relevance (project): GA can select MACD components or crossover flags instead of multiple correlated EMA distances, improving sparsity while retaining directional regime context (benefits cv_stability).  
Example:
```python
fast = close.ewm(span=12, adjust=False).mean()
slow = close.ewm(span=26, adjust=False).mean()
macd_line = fast - slow
signal = macd_line.ewm(span=9, adjust=False).mean()
hist = macd_line - signal
```
Interpretation: MACD crossing above signal with histogram expanding ⇒ strengthening bullish momentum (potential Target=1).
</details>

<details><summary><strong>MACD_Line</strong></summary>
Purpose: Fast EMA − Slow EMA (e.g., EMA_12 − EMA_26).  
How/When: Built after computing the two EMAs.  
Why: Raw momentum differential; removes shared long-term trend, emphasizing velocity change.  
Relevance: Often retained if GA favors continuous momentum slope over binary cross flags.  
Example: `MACD_Line = EMA_12 - EMA_26`.  
Signal: Rising MACD_Line above zero confirms bullish momentum bias.
</details>

<details><summary><strong>MACD_Signal</strong></summary>
Purpose: Smoothed (typically 9‑period) EMA of MACD_Line.  
How/When: Immediately after MACD_Line creation.  
Why: Dampens noise; establishes reference baseline for cross detection.  
Relevance: Enables creation of low‑entropy crossover features the GA may prefer for parsimony.  
Example: `MACD_Signal = MACD_Line.ewm(span=9, adjust=False).mean()`.
</details>

<details><summary><strong>MACD_Histogram</strong></summary>
Purpose: MACD_Line − MACD_Signal; measures distance (momentum acceleration / deceleration).  
How/When: After both MACD_Line and MACD_Signal computed.  
Why: Zero‑centered oscillator; slope + sign shifts can precede price turns.  
Relevance: GA may keep histogram instead of both line + signal to reduce redundant dimensionality.  
Example: Positive and growing histogram ⇒ strengthening bullish impulse.
</details>

<details><summary><strong>MACD_CrosAbove / MACD_CrossAbove</strong></summary>
Purpose: Binary flag = 1 when MACD_Line crosses above MACD_Signal (bullish crossover).  
Note: MACD_CrosAbove is a historical typo; standardized form: MACD_CrossAbove.  
How/When: Created with shift comparison to avoid same-bar bias.  
Why: Compresses multi-valued momentum relationship into a sparse, interpretable event.  
Relevance: GA may select this instead of raw MACD components, aiding feature reduction.  
Example:
```python
cross_above = (macd_line > signal) & (macd_line.shift(1) <= signal.shift(1))
df['MACD_CrossAbove'] = cross_above.astype(int)
```
</details>

<details><summary><strong>MACD_CrossBelow</strong></summary>
Purpose: Binary flag = 1 when MACD_Line crosses below MACD_Signal (bearish crossover).  
How/When: Same logic as CrossAbove with inverted condition.  
Why: Encodes bearish momentum shift without continuous values.  
Relevance: Pair (CrossAbove, CrossBelow) offers low‑cardinality regime change signals benefiting stability.  
Example:
```python
cross_below = (macd_line < signal) & (macd_line.shift(1) >= signal.shift(1))
df['MACD_CrossBelow'] = cross_below.astype(int)
```
</details>

<details><summary><strong>Momentum</strong></summary>
Purpose: General concept describing rate and persistence of directional price movement.  
How: Operationalized via differences, oscillators (MACD, RSI, ROC), distance-to-average features.  
Why: Momentum regimes (persistent drift vs mean reversion) influence class probabilities (Up/Down vs Sideways).  
Relevance: Many engineered indicators are momentum proxies; GA selection curates the minimal subset with stable predictive contribution.  
Example Concept: Sustained positive ROC + MACD_Histogram expansion = bullish momentum cluster.
</details>

<details><summary><strong>Momentum Indicators</strong></summary>
Purpose: Family of technical indicators quantifying speed or persistence of price changes (RSI, Stochastic, ROC, Williams %R, CCI, MACD).  
How/When: Computed after base returns and rolling statistics.  
Why: Provide orthogonal momentum perspectives (overbought/oversold, rate of change, directional efficiency).  
Relevance: Redundant momentum measures inflate correlation; GA prunes overlap, improving feature reduction %.  
Example: RSI detects overextension; ROC captures velocity; pairing both may yield stable cross-validation performance.
</details>

<details><summary><strong>RSI / <code>RSI_{period}</code></strong></summary>
Purpose: Relative Strength Index: 100 − 100 / (1 + RS); RS = avg gain / avg loss over lookback.  
How/When: After price diffs computed; typical periods: 2, 7, 14, 21.  
Why: Normalized (0–100) momentum oscillator highlighting overbought/oversold extremes.  
Relevance: Multiple RSI periods can be collinear; GA may retain a single period or a fast/slow pair for regime adaptability.  
Example:
```python
delta = close.diff()
gain = delta.clip(lower=0).rolling(p).mean()
loss = (-delta.clip(upper=0)).rolling(p).mean()
rs = gain / loss
df[f'RSI_{p}'] = 100 - 100 / (1 + rs)
```
Interpretation: RSI_14 < 30 may precede mean-reversion (Sideways→Up).
</details>

<details><summary><strong>Stochastic Oscillator / <code>Stoch_K_{period}</code> / <code>Stoch_D_{period}</code></strong></summary>
Purpose: %K = (Close − LowestLow) / (HighestHigh − LowestLow) * 100; %D = smoothed %K (usually 3-period SMA).  
How/When: After rolling high/low windows computed (e.g., 14, 21).  
Why: Measures close placement within recent range (momentum + relative positioning).  
Relevance: Provides different normalization vs RSI; GA may choose %K or %D for smoother signal.  
Example:
```python
low_roll = low.rolling(p).min()
high_roll = high.rolling(p).max()
df[f'Stoch_K_{p}'] = 100 * (close - low_roll) / (high_roll - low_roll)
df[f'Stoch_D_{p}'] = df[f'Stoch_K_{p}'].rolling(3).mean()
```
</details>

<details><summary><strong>ROC (Rate of Change) / <code>ROC_{period}</code></strong></summary>
Purpose: Percentage change over N periods: (Close_t / Close_{t-N} − 1) * 100.  
How/When: After basic returns; periods e.g., 5, 10, 20.  
Why: Captures velocity over multi-bar horizon, filtering single-bar noise.  
Relevance: Faster ROC windows may complement slower trend distances; GA balances short vs medium horizons for stability.  
Example:
```python
p = 10
df[f'ROC_{p}'] = (close / close.shift(p) - 1) * 100
```
</details>

<details><summary><strong>Williams %R / <code>Williams_R_{period}</code></strong></summary>
Purpose: Momentum oscillator: -100 * (HighestHigh − Close) / (HighestHigh − LowestLow). Range -100 to 0.  
How/When: After rolling min/max.  
Why: Similar to Stochastic but inverted scale; identifies overbought (> -20) / oversold (< -80) zones.  
Relevance: Provides alternative scaling; GA may keep one of Williams %R vs Stoch_K to avoid redundancy.  
Example:
```python
hh = high.rolling(p).max()
ll = low.rolling(p).min()
df[f'Williams_R_{p}'] = -100 * (hh - close) / (hh - ll)
```
</details>

<details><summary><strong>CCI / <code>CCI_{period}</code></strong></summary>
Purpose: Commodity Channel Index: (TypicalPrice − SMA(TP)) / (0.015 * MeanAbsDeviation).  
How/When: After computing TypicalPrice = (High+Low+Close)/3; typically period = 20.  
Why: Measures deviation from statistical mean; highlights cyclical extremes beyond simple z-scores.  
Relevance: Adds a volatility-adjusted overextension metric; GA may retain CCI instead of multiple overlapping z-score style features.  
Example:
```python
tp = (high + low + close) / 3
sma = tp.rolling(p).mean()
mad = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
df[f'CCI_{p}'] = (tp - sma) / (0.015 * mad)
```
</details>

<details><summary><strong>momentum divergence / convergence</strong></summary>
Purpose: Concept where price makes new highs (or lows) while a momentum indicator (e.g., MACD_Histogram, RSI) fails to confirm (divergence) or realigns (convergence).  
How/When: Detected analytically post feature generation (not explicitly engineered yet).  
Why: Signals potential trend exhaustion (divergence) or confirmation (convergence).  
Relevance: Although not directly encoded, GA-selected momentum + price distance features allow models to implicitly learn divergence patterns (improving minority class recall).  
Example Concept: Price higher high + RSI lower high ⇒ bearish divergence (possible future Down classification).  
Extension: Could add engineered binary Divergence_Flag features in future runs and compare selection frequency.
</details>

## Bollinger Bands & Volatility Indicators

<details><summary><strong>Bollinger Bands</strong></summary>
Purpose: A volatility envelope consisting of a moving average (middle band) and upper/lower bands offset by k standard deviations.  
How/When: For each period window (e.g., 20), compute rolling mean and rolling std of Close; upper = mean + k*std (k=2 typical), lower = mean - k*std. Calculated during baseline feature engineering after sorting data and before target generation.  
Where (project): Columns BB_Middle_{period}, BB_Std_{period}, BB_Upper_{period}, BB_Lower_{period}, BB_Width_{period}, BB_Pct_B_{period}.  
Why: Dynamically scales with volatility—captures compression (squeeze) and expansion phases that often precede directional shifts (Up/Down vs Sideways).  
Relevance: GA may retain band-derived features because they encode regime shifts and volatility-normalized positioning, improving cv_stability.  
Example:
```python
w = 20
mid = close.rolling(w).mean()
std = close.rolling(w).std()
upper = mid + 2 * std
lower = mid - 2 * std
```
Interpretation: Price hugging upper band with widening width ⇒ momentum; contracting width ⇒ potential breakout setup.  
</details>

<details><summary><strong>BB_Middle_{period}</strong></summary>
Purpose: Rolling simple moving average (baseline) of Close for the specified period.  
How: mean = Close.rolling(period).mean().  
When: Created before derived band metrics (width, %B).  
Why: Serves as central tendency; deviations (price - middle) contextualize over/under-extension.  
Relevance: GA might choose BB_Middle over a redundant SMA_{period} if band ecosystem features (width/%B) are also present, aiding dimensionality reduction.  
Example: BB_Middle_20 ≈ equilibrium price; frequent crossings may align with Sideways (Target=2).  
</details>

<details><summary><strong>BB_Std_{period}</strong></summary>
Purpose: Rolling standard deviation of Close used to scale band distance.  
How: Close.rolling(period).std().  
Why: Adaptive volatility gauge—higher std widens bands, reducing false overbought/oversold signals.  
Relevance: Provides raw volatility magnitude; GA may select this instead of redundant volatility features if it stabilizes fitness.  
Example: Spike in BB_Std_20 often precedes band expansion and potential trend continuation filtering noise.  
</details>

<details><summary><strong>BB_Upper_{period}</strong></summary>
Purpose: Upper envelope = BB_Middle_{period} + k * BB_Std_{period} (k=2).  
When: After middle & std computed.  
Why: Defines statistically high boundary; touches during expansion can indicate sustained momentum rather than reversal.  
Relevance: Not always selected directly (derived ratios more compact); still intermediate for other band metrics.  
Example: If Close > BB_Upper_20 and BB_Width rising, probability of sustained Up (Target=1) may increase.  
</details>

<details><summary><strong>BB_Lower_{period}</strong></summary>
Purpose: Lower envelope = BB_Middle_{period} - k * BB_Std_{period}.  
Why: Symmetric downside boundary; touches during contraction may signal mean reversion or impending breakdown.  
Relevance: Similar redundancy considerations as BB_Upper; GA may skip raw boundary in favor of normalized %B.  
Example: Close dipping below BB_Lower_20 with narrowing width can precede volatility expansion reversal.  
</details>

<details><summary><strong>BB_Width_{period}</strong></summary>
Purpose: Relative band width = (BB_Upper - BB_Lower) / BB_Middle.  
How: Captures normalized volatility (scale independent).  
Why: Measures compression/expansion cycles; lower widths (squeezes) often precede breakouts.  
Relevance: Strong candidate for GA selection—compact, volatility-aware, aids class boundary discrimination (Sideways vs upcoming Up/Down).  
Example:
```python
width = (upper - lower) / mid
```
Interpretation: Persistent low BB_Width_20 then sharp increase with directional close improves predictive stability.  
</details>

<details><summary><strong>BB_Pct_B_{period}</strong></summary>
Purpose: Percent bandwidth position: (Close - BB_Lower) / (BB_Upper - BB_Lower); ranges 0–1.  
Why: Scale-free oscillator indicating where price sits within the envelope.  
How: Computed after upper/lower bands.  
Relevance: Encodes overextension without raw price; GA often prefers %B over simultaneous inclusion of upper & lower lines (feature sparsity).  
Example: BB_Pct_B_20 > 0.9 with widening width suggests momentum continuation; <0.1 indicates downward pressure.  
</details>

<details><summary><strong>Volatility</strong></summary>
Purpose: General measure of price variability (dispersion) over time.  
How (project): Rolling std of returns (Percentage_Change), ATR, band std, specialized estimators (Garman-Klass, Parkinson).  
When: Early feature engineering to build downstream ratios (ATR_Ratio, Normalized_Vol).  
Why: Drives dynamic target thresholds (vol * MULTIPLIER) to balance class distribution.  
Relevance: Accurate volatility features enhance Target labeling robustness and GA fitness stability.  
Example: vol_20 = returns.rolling(20).std().  
</details>

<details><summary><strong>Volatility Indicator</strong></summary>
Purpose: Any feature specifically quantifying or proxying variability (ATR, BB_Std, Volatility_{period}, GK_Volatility).  
Why: Different formulations capture gap risk, high–low dispersion, or close-to-close noise.  
Relevance: GA can diversify risk regime encoding selecting complementary volatility indicators improving generalization.  
Example: Combining ATR_14 + BB_Width_20 often outperforms either alone in stability.  
</details>

<details><summary><strong>ATR, ATR_{period}, ATR_Ratio_{period}</strong></summary>
Purpose: Average True Range—smoothed average of True Range capturing gap and intrabar volatility. ATR_Ratio normalizes ATR by price (or Close) * 100 for scale invariance.  
How: True Range = max(High-Low, |High-prevClose|, |Low-prevClose|); ATR = rolling mean over period (e.g., 14).  
When: Mid feature engineering after base OHLC transformations.  
Why: Robust to jumps; informs dynamic thresholding and risk context. ATR_Ratio_{period} enhances comparability across price levels.  
Relevance: Normalized variants often favored by GA (penalizes raw scale redundancy) and stabilize cv_stability.  
Example:
```python
tr = pd.concat([high-low,
               (high-close.shift(1)).abs(),
               (low-close.shift(1)).abs()], axis=1).max(axis=1)
df['ATR_14'] = tr.rolling(14).mean()
df['ATR_Ratio_14'] = df['ATR_14'] / df['Close'] * 100
```
</details>

<details><summary><strong>Volatility_{period}</strong></summary>
Purpose: Rolling (period) standard deviation of percentage returns scaled by sqrt(period) or raw std (project: Close.pct_change().rolling(period).std() * sqrt(period)).  
Why: Captures realized variability with horizon sensitivity; longer periods smooth noise, shorter respond faster.  
Relevance: GA may balance a fast (5) and medium (20/50) volatility window for regime detection without redundancy.  
Example: Volatility_10 spike ⇒ potential expansion; may shift Target class distribution.  
</details>

<details><summary><strong>Normalized_Vol_{period}</strong></summary>
Purpose: Relative volatility = Volatility_{period} / rolling_mean(Volatility_{period}, long_window) (e.g., / rolling(100).mean()).  
Why: Highlights volatility regimes (above or below long-term average) rather than raw magnitude.  
Relevance: Enhances stability by contextualizing current variability; GA often prefers normalized measures to avoid scale drift across runs.  
Example:
```python
vol_p = returns.rolling(p).std()
norm_vol_p = vol_p / vol_p.rolling(100).mean()
```
Interpretation: >1 ⇒ elevated regime; <1 calm regime (more Sideways).  
</details>

<details><summary><strong>GK_Volatility (Garman-Klass)</strong></summary>
Purpose: Estimator using Open, High, Low, Close to capture overnight/ intrabar information with lower variance than close-to-close std.  
Formula (simplified):
σ² ≈ 0.5*(ln(High/Low))² - (2ln2 -1)*(ln(Close/Open))².  
When: After OHLC available before other volatility composites.  
Why: Incorporates range + open/close movement, improving efficiency.  
Relevance: Provides alternative volatility signal; GA can select GK_Volatility if it increases fitness stability vs redundant standard deviation measures.  
Example:
```python
df['GK_Volatility'] = np.sqrt(0.5*np.log(df.High/df.Low)**2 -
                              (2*np.log(2)-1)*np.log(df.Close/df.Open)**2)
```
</details>

<details><summary><strong>Parkinson estimators</strong></summary>
Purpose: Range-based volatility estimator using only High & Low; σ² ≈ (1/(4 ln 2)) * (ln(High/Low))².  
How: Ignores open/close; efficient under continuous trading/no drift assumption.  
Why: More efficient than close-to-close when high/low accurate; sensitive to outliers if jumps present.  
Relevance: Adds diversity to volatility feature pool; GA may keep Parkinson_Vol when its simplicity offers complementary signal to ATR or GK.  
Example:
```python
df['Parkinson_Vol'] = np.sqrt((1/(4*np.log(2))) * (np.log(df.High/df.Low)**2))
```
</details>

<details><summary><strong>implied vs realized volatility</strong></summary>
Purpose: Implied volatility (IV) derives from option prices (forward-looking); realized volatility is historical (statistical) variability.  
Project Context: Only realized volatility proxies (ATR, BB_Std, GK, Parkinson, rolling std) are currently implemented—no option chain data available.  
Why Mentioned: Highlights potential extension: integrating IV could enhance feature richness and regime anticipation.  
Relevance: Documenting distinction clarifies that current GA optimization is constrained to realized measures; future inclusion of IV could improve early detection of volatility regime shifts affecting Target thresholds.  
Example Concept: If IV >> realized vol, market anticipates expansion—could adjust MULTIPLIER adaptively.  
</details>

## Volume & Tick-Based Indicators

<details><summary><strong>Volume Indicators (category)</strong></summary>
Purpose: Group of features quantifying traded (or tick) activity to contextualize price movement (confirm / refute directional bars, detect exhaustion or breakout readiness).  
How/When: Engineered after OHLC data is cleaned; tick‑volume aware branch executes when VOLUME_TYPE_TICKER = True.  
Where: Created in Baseline Feature Engineering (section “Volume Indicators (tick-volume aware)” in Combined_Metaheuristics_Workflow.ipynb).  
Why: Volume (or its proxy) often precedes or confirms directional moves; combining volatility + price + volume improves class separation (Up / Down vs Sideways).  
Relevance (project): GA can selectively retain the minimal subset (e.g., TickVol_Z_20 + OBV_TickDev) that stabilizes cross‑validation performance without redundancy.  
Example: Low TickVol_Z_20 + contracting BB_Width_20 may signal pending volatility expansion; selected in a parsimonious GA subset.  
</details>

<details><summary><strong>volume</strong></summary>
Purpose: Raw per‑bar traded size; here may be “true” exchange volume (commodities/equities) or substituted by tick counts.  
How/When: Loaded directly from raw_data CSV before any derived computations.  
Where: Base column ‘Volume’; copied for transformations.  
Why: Fundamental intensity measure; absolute changes can flag regime shifts or news bursts.  
Relevance: Trees may down‑weight raw scale; derived normalized variants (ratios / z-scores) often chosen by GA for stability.  
Example: Spiking Volume with wide Candle_Range can elevate probability of non‑Sideways (Target ≠ 2).  
</details>

<details><summary><strong>Tick_Volume</strong></summary>
Purpose: Proxy for true traded size when only tick/event counts are available (FX / CFDs).  
How/When: Created when VOLUME_TYPE_TICKER = True: feature_df['Tick_Volume'] = Volume.astype(float).  
Where: Volume indicators block (tick-volume aware branch).  
Why: Still correlates with underlying participation; enables relative / standardized computations.  
Relevance: Underpins all tick‑volume derived rolling stats (mean, std, z-score) feeding GA search space.  
Example: A sudden 3× Tick_Volume spike may precede directional continuation captured by AMA_GA_Optimized sensitivity.  
</details>

<details><summary><strong>TickVol_RollMean_20</strong></summary>
Purpose: 20-bar rolling mean of Tick_Volume (local activity baseline).  
How/When: feature_df['TickVol_RollMean_20'] = Tick_Volume.rolling(20).mean().  
Where: Tick-volume feature engineering section.  
Why: Normalizes instantaneous ticks; supports creation of relative / z-score features.  
Relevance: GA may not select it directly but uses it indirectly through derived columns (TickVol_Z_20, TickVol_Relative_24).  
Example: If Tick_Volume = 120, RollMean = 60 ⇒ activity is 2× typical → potential breakout context.  
</details>

<details><summary><strong>TickVol_RollStd_20</strong></summary>
Purpose: 20-bar rolling standard deviation of Tick_Volume (short-term dispersion).  
How/When: Calculated alongside rolling mean; zeros replaced with NaN to avoid divide-by-zero in z-score.  
Where: Tick-volume block.  
Why: Needed to standardize tick volume (z-score).  
Relevance: Stability: scaling raw spikes reduces variance across CV folds; helps GA evaluating subsets with z-scored volume signals.  
Example: High RollStd implies volatility in activity; a moderate z-score might be less meaningful if dispersion is large.  
</details>

<details><summary><strong>TickVol_Z_20</strong></summary>
Purpose: Standardized tick volume: (Tick_Volume − TickVol_RollMean_20) / TickVol_RollStd_20.  
How/When: After mean & std exist.  
Where: Tick-volume engineering section.  
Why: Produces scale‑free anomaly signal (activity shock indicator).  
Relevance: GA often favors z-scores over raw counts for cross-run comparability; improves cv_stability by tempering scale differences.  
Example: TickVol_Z_20 > 2 may precede strong directional bar improving predictive F1.  
</details>

<details><summary><strong>TickVol_Relative_24</strong></summary>
Purpose: Ratio of current tick volume to 24-bar rolling mean (diurnal / intraday adjustment).  
How/When: feature_df['TickVol_Relative_24'] = Tick_Volume / Tick_Volume.rolling(24).mean().  
Where: Tick-volume block.  
Why: Captures whether current participation is elevated vs a full day cycle smoothing seasonal patterns.  
Relevance: Complements TickVol_Z_20 (short horizon) with a longer contextual measure; GA may keep one of them to reduce redundancy.  
Example: Relative_24 = 1.8 suggests sustained activity regime shift vs isolated spike.  
</details>

<details><summary><strong>TickVol_Log1p</strong></summary>
Purpose: Log-transformed tick volume: log(1 + Tick_Volume).  
How/When: Created after raw tick volume to reduce skew/outlier dominance.  
Where: Tick-volume block.  
Why: Stabilizes variance; mitigates impact of large bursts on tree split thresholds / scaler.  
Relevance: Sometimes substitutes for raw volume in GA-selected subset where z-score volatility is noisy.  
Example: Tick_Volume 300 vs 30 compresses to log1p(301) ≈ 5.71 vs log1p(31) ≈ 3.47 (reduced ratio).  
</details>

<details><summary><strong>TickVol_Clipped</strong></summary>
Purpose: Winsorized tick volume capped at 99th percentile.  
How/When: upper_clip = quantile(0.99); then clip(lower=1, upper=upper_clip).  
Where: Tick-volume block.  
Why: Prevent extreme spikes from distorting scaling or tree impurity computations.  
Relevance: Enhances stability across folds; GA may choose clipped version if raw extremes hurt minority class recall.  
Example: A 20× anomaly becomes capped—preventing that bar from dominating model structure.  
</details>

<details><summary><strong>OBV (On Balance Volume)</strong></summary>
Purpose: Cumulative volume adding volume on up bars, subtracting on down bars (price-volume confirmation).  
How/When: Only in non tick-volume branch (true volume scenario).  
Where: Volume Indicators block (else branch).  
Why: Seeks directional participation confirmation.  
Relevance: In tick-volume branch replaced by OBV_TickDev (volatility-aware variant); GA may still favor simpler OBV if present.  
Example: Rising OBV with EMA_Dist_20 > 0 strengthens Up (Target=1) probability.  
</details>

<details><summary><strong>OBV_Change</strong></summary>
Purpose: Intermediate signed per-bar volume delta used to build cumulative OBV.  
How/When: Defined via np.where price up/down logic before cumulative sum.  
Where: Non tick-volume path.  
Why: Decomposes contribution for optional alternative aggregations.  
Relevance: Rarely selected directly; acts as source for OBV feature.  
Example: OBV_Change = +Volume when Close_t > Close_{t-1}.  
</details>

<details><summary><strong>OBV_TickDev</strong></summary>
Purpose: Cumulative sum of (price direction sign × deviation of tick volume from its rolling mean).  
How/When: sign = sign(Close.diff()); dev_from_mean = Tick_Volume − TickVol_RollMean_20; cumulative.  
Where: Tick-volume branch.  
Why: Emphasizes abnormal activity aligned with direction; filters out routine volume.  
Relevance: Often higher signal-to-noise vs raw OBV under proxy volume; GA may prioritize it due to improved cv_stability.  
Example: Two consecutive bullish bars with +40 and +30 tick deviations produce rising OBV_TickDev trend.  
</details>

<details><summary><strong>CMF (Chaikin Money Flow)</strong></summary>
Purpose: Volume-weighted accumulation/distribution oscillator using (Close positioning within High-Low) × Volume over window / sum(Volume).  
How/When: Computed in non tick-volume mode (CMF_20).  
Where: Volume indicators else branch.  
Why: Measures buying/selling pressure intensity.  
Relevance: Normalized 0-ish centered; can complement momentum (MACD) for regime filtering in GA-selected subsets.  
Example: CMF_20 > 0.2 indicates persistent accumulation bias.  
</details>

<details><summary><strong>CMF_20_tick</strong></summary>
Purpose: Chaikin Money Flow adapted for tick volume (proxy).  
How/When: Uses mf_multiplier * Tick_Volume rolling sums.  
Where: Tick-volume branch.  
Why: Enables accumulation/distribution style signal without true volume.  
Relevance: GA may pick CMF_20_tick when OBV_TickDev not selected to retain one participation-strength proxy.  
Example: Sustained CMF_20_tick > 0 supports retaining bullish class predictions with fewer false positives.  
</details>

<details><summary><strong>Volume_Osc_tick_{short_period}_{long_period}</strong></summary>
Purpose: Normalized difference between short and long rolling means of tick volume: (MA_short − MA_long)/MA_long * 100.  
How/When: Loop over (5,10) and (12,26).  
Where: Tick-volume branch.  
Why: Detects momentum in activity (accelerating participation).  
Relevance: A compact leading indicator for volume regime shifts; GA may retain only one parameterization for sparsity.  
Example: Volume_Osc_tick_5_10 > 50 signals recent surge relative to medium baseline.  
</details>

<details><summary><strong>Volume_ROC_tick_{period}</strong></summary>
Purpose: Percentage rate of change of tick volume over period: (Tick_Volume / Tick_Volume.shift(p) − 1) * 100.  
How/When: Periods 10, 20.  
Where: Tick-volume branch.  
Why: Captures abrupt volume accelerations not smoothed by moving averages.  
Relevance: Complements oscillator (level vs growth). GA may prefer ROC if oscillator redundant with z-score.  
Example: ROC_10 = 120% implies more than doubling relative to 10 bars earlier.  
</details>

<details><summary><strong>VWAP (Volume Weighted Average Price)</strong></summary>
Purpose: Price level weighted by volume within a session: sum(price*volume)/sum(volume). (Not currently implemented; conceptual placeholder.)  
How/When (future): Intraday cumulative or rolling-session basis.  
Where: Would be added to feature engineering pre-GA.  
Why: Reflects average execution price; price above VWAP may indicate intraday strength.  
Relevance: Could enhance regime context; inclusion would allow GA to substitute multiple MA deviations with a single volume-aware anchor.  
Example Concept:
```python
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
```
</details>

<details><summary><strong>order book (bid/ask, spread, depth)</strong></summary>
Purpose: Microstructure data describing available liquidity (best bid/ask quotes and size distribution). (Not in current dataset.)  
How/When (future): Stream from broker/API; aggregate to per-bar spread & depth metrics.  
Where: Potential extension in raw_data ingestion layer before feature engineering.  
Why: Narrowing spread + rising depth supports reliable breakouts; widening spread may foretell volatility bursts.  
Relevance: Additional microstructure features could reduce reliance on proxy tick-volume indicators, potentially improving GA feature reduction by replacing multiple indirect signals.  
Example Concept Metrics: spread = ask − bid; mid-price = (bid+ask)/2; depth imbalance = (bid_size − ask_size)/(bid_size + ask_size).  
</details>

## Technical Indicator Miscellany

<details><summary><strong>Technical indicator (general)</strong></summary>
Purpose: Quantitative transformation of raw OHLCV (and derived) data intended to summarize price/volume behavior (trend, momentum, volatility, participation) in a form more learnable by models.  
How/When/Where: Computed during baseline feature engineering (Combined_Metaheuristics_Workflow.ipynb Section 3) directly from chronologically sorted OHLCV before target generation; stored as columns in engineered_features.csv and passed (after scaling) into GA feature selection.  
Why: Raw prices are non‑stationary and highly autocorrelated; indicators convert them into (often partially standardized) signals encoding regimes (trend vs range, expansion vs contraction) improving model discriminative power and cv_stability.  
Relevance (project): Forms the search space “feature set” for the GeneticAlgorithm. Selection pressure prunes redundant / noisy indicators, improving feature reduction percentage while maintaining or improving weighted F1.  
Example: RSI_14, ATR_14, MACD_Histogram, BB_Width_20, Volatility_10.  
Edge Notes: Indicators can be collinear (e.g., SMA vs EMA vs TEMA). GA mitigates multicollinearity by subset optimization.  
</details>

<details><summary><strong>Advanced Indicators: Adaptive Moving Average (AMA), Fractal Dimension Indicator (FDI)</strong></summary>
Purpose: Higher‑order adaptive constructs capturing market state beyond fixed‑window smoothing: AMA adjusts responsiveness to volatility/efficiency; FDI measures geometric roughness (trend persistence vs choppiness).  
How/When/Where: Engineered in GA Feature Engineering phase (Sections 6–7 of Combined workflow) via MH_Feature_Engineering.py after baseline GA feature selection (so their marginal contribution can be evaluated distinctly). Added as new columns AMA_GA_Optimized & FDI_GA_Optimized.  
Why: Static moving averages lag in regime shifts; fractal/efficiency concepts adapt to structure, potentially stabilizing model performance under non‑stationarity and volatility clustering.  
Relevance: Provide orthogonal adaptive signals often rank moderately to highly in feature importance, improving cv_stability and reducing reliance on multiple redundant traditional indicators.  
Example Intuition: During smooth directional move AMA tracks price closely (low lag); in noisy consolidation it damps responsiveness, reducing false momentum signals. FDI near 1.2–1.3 (trend) vs ~1.5 (random/choppy).  
Edge Notes: Their parameters are themselves GA‑optimized for this dataset window → reproducibility tied to run_* timestamp artifacts.  
</details>

<details><summary><strong>AMA_GA_Optimized</strong></summary>
Purpose: Adaptive Moving Average series produced by a parameter set (smoothing_factor, volatility_window, fast_period, slow_period) evolved via GA to maximize weighted F1 + stability on the training window.  
How/When/Where: Created by engineer_adaptive_moving_average() (MH_Feature_Engineering.py) after baseline selection; inserted into the enhanced feature subset before tuning enhanced models. Stored in run directory (engineered_features_info.pkl).  
Why: Dynamically modulates smoothing constant using an efficiency ratio: accelerates in directional phases (reducing lag) and decelerates in noise (reducing whipsaw), offering a single adaptive alternative to multiple fixed MAs.  
Relevance: Often enables pruning of several SMA_Dist / EMA_Dist features while retaining or increasing predictive power (improved feature reduction percentage and potential F1 uplift).  
Example (conceptual pseudo):
eff_ratio = abs(price_t - price_{t−W}) / sum(abs(diff_i)) over window W  
smooth_const = (ER*(fastSC - slowSC) + slowSC)^2  
AMA_t = AMA_{t−1} + smooth_const * (Close_t - AMA_{t−1})  
Edge Notes: GA discards parameter sets yielding low variance or unstable CV performance (fitness penalizes instability).  
</details>

<details><summary><strong>FDI_GA_Optimized</strong></summary>
Purpose: Fractal Dimension Indicator series optimized via GA parameters (window_size, high_low_factor, scaling_factor) to quantify path roughness and differentiate trending vs ranging regimes.  
How/When/Where: Generated by engineer_fractal_dimension_indicator() after AMA creation; appended to dataset and included in enhanced feature lists.  
Why: Fractal dimension decreases toward 1 in smoother (trend) paths and approaches 1.5 in more random/choppy conditions; using FDI allows the model to condition interpretations of momentum / breakout indicators, aiding stability across volatile regime transitions.  
Relevance: Helps classifier avoid overpredicting directional classes in high‑noise zones (improves precision) and supports selective activation of trend features (improves recall for Up/Down).  
Example Interpretation: If FDI_GA_Optimized < 1.3 and MACD_Histogram positive expansion → higher probability Target=1 vs same MACD in FDI ~1.5 (likely Sideways).  
Edge Notes: Poor parameter sets yielding near-constant series receive low fitness (variance + stability components).  
</details>

<details><summary><strong>feature interactions</strong></summary>
Purpose: Combinations (implicit or explicit) where predictive signal emerges from joint relationship (e.g., low BB_Width_20 AND rising TickVol_Z_20) rather than any single feature independently.  
How/When/Where: Not explicitly engineered (no polynomial expansion by default). Tree models & XGBoost learn splits that approximate interactions; GA selection curates a subset enabling those emergent synergies. Potential future extension: explicit cross features (e.g., DI_Diff_14 * BB_Width_20).  
Why: Many market phenomena are conditional (momentum signals stronger under low volatility contraction → expansion). Capturing interactions improves discriminative structure without overfitting if dimensionality controlled.  
Relevance: GA indirectly optimizes for interaction-supportive sparsity—retaining complementary (non‑redundant) features that together stabilize cross‑validation variance.  
Example: (BB_Width_20 < percentile_25) & (Volume_Osc_tick_5_10 > 0) preceding breakout (Target=1). Model learns decision path when both conditions satisfied.  
Edge Notes: Adding explicit interaction explosion without selection would inflate search space; GA-driven pruning prerequisite for any interaction expansion roadmap.  
</details>

<details><summary><strong>lagged features</strong></summary>
Purpose: Historical shifted versions of base indicators (e.g., RSI_14.shift(1)) capturing temporal persistence / transition dynamics.  
How/When/Where: Currently implicit through rolling constructs (they contain lag info); explicit lag columns not broadly added to limit dimensionality. Could be introduced post initial GA to evaluate marginal gain.  
Why: Financial signals often exhibit short memory; lags can help models distinguish continuation vs reversal patterns.  
Relevance: Avoiding early proliferation keeps GA search tractable; adaptive indicators (AMA, FDI) absorb some temporal context, reducing immediate need for explicit lags while achieving feature reduction.  
Example (future extension):
df['RSI_14_lag1'] = df['RSI_14'].shift(1) (would drop resulting NaNs before splits).  
Edge Notes: Must ensure no negative shifts (future leakage) and maintain chronological alignment—especially critical for Target horizon truncation.  
</details>

<details><summary><strong>lead/lag relationships</strong></summary>
Purpose: Temporal ordering where one feature (leading) changes systematically ahead of another (lagging), supporting predictive inference (e.g., volume surge leading price expansion).  
How/When/Where: Assessed analytically via domain knowledge & exploratory correlation / partial dependence; not explicitly encoded except through existing indicators whose construction inherently lags (moving averages) or responds rapidly (z-scores, AMA).  
Why: Identifying reliable leading signals (e.g., contraction + volume acceleration) improves model anticipation of class transition (Sideways → Up/Down).  
Relevance: GA tends to keep a mix of faster (Volatility_5, TickVol_Z_20) and slower (SMA_50, ADX_14) features enabling the model to synthesize lead/lag structure improving cv_stability across regimes.  
Example: Declining BB_Width_20 (lead) followed by rising MACD_Histogram (lag) = breakout confirmation pattern; model learns conditional probability mapping.  
Edge Notes: Spurious apparent lead/lag can arise from overlapping rolling windows—TimeSeriesSplit evaluation mitigates overfitting to accidental alignments.  
</details>

## Feature Engineering & Preprocessing

<details><summary><strong>feature engineering</strong></summary>
Purpose: Create new informative predictor variables from raw OHLCV to expose regime, momentum, volatility, structure.  
How: Transform (returns, ranges), aggregate (rolling stats), normalize (distances, z‑scores), encode events (crossovers, flags), adapt (AMA, FDI).  
When: After raw data cleaning & chronological filtering; before target generation (so target logic can reuse engineered primitives) and before scaling / GA selection.  
Where: Section 3 (Baseline Feature Engineering) of Combined_Metaheuristics_Workflow.ipynb; persisted to engineered_features.csv / working_data.  
Why: Raw prices are non‑stationary and collinear; engineered signals stabilize variance and offer compact regime descriptors improving model + GA fitness (accuracy + cv_stability − complexity).  
Relevance (project): Defines the initial “feature set” search space the GeneticAlgorithm will prune; richer yet not bloated engineering improves probability GA can find sparse high‑stability subsets.  
Example: Create ATR ratio for scale invariance:  
```python
tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
df['ATR_14'] = tr.rolling(14).mean()
df['ATR_Ratio_14'] = df['ATR_14']/df['Close'] * 100
```
</details>

<details><summary><strong>data preprocessiong (sic) / data preprocessing</strong></summary>
Purpose: Prepare data for modeling by enforcing temporal integrity, handling NaNs, scaling, and splitting.  
How: Sort by datetime, drop or fill NaNs (ffill/bfill then drop residual), remove infinities, create Target then truncate horizon, time‑series split (80/10/10), scale features with StandardScaler fit ONLY on training.  
When: Immediately after feature engineering & target creation; before GA feature selection and baseline model training.  
Where: Sections 2–4 of Combined_Metaheuristics_Workflow.ipynb; artifacts saved as X_*_scaled.pkl, y_*.pkl, scaler.pkl in run_* directory.  
Why: Ensures no leakage (future info in train), consistent distributions across folds, and reproducible GA fitness evaluation.  
Relevance: Incorrect preprocessing (e.g., scaling on full dataset) would inflate cv_stability and misguide feature subset evolution.  
Example:
```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
```
</details>

<details><summary><strong>Normalization / scaling patterns</strong></summary>
Purpose: Standardize numerical feature scales to stabilize model splits and logistic/ANN/linear components inside GA fitness and feature engineering GA.  
Patterns Used: StandardScaler (zero mean/unit std), percentage distance ((Close - SMA)/SMA*100), z‑scores ((x - rolling_mean)/rolling_std), ratio normalization (ATR/Close*100), log1p for skew (TickVol_Log1p), relative volatility (Volatility / long_mean).  
When: After feature construction; distances and ratios created inline during engineering, standard scaling applied post split.  
Why: Mixed scales (ATR in price units vs ratios in %) can bias importance and tree impurity; normalized variants reduce redundancy and foster generalizable subsets.  
Relevance: Dynamic rates and diversity mechanisms in GA rely on consistent fitness signals—scaling reduces variance noise across folds (enhancing cv_stability component).  
Example (distance + z-score):
```python
df['SMA_20_Dist'] = (df['Close'] - df['SMA_20'])/df['SMA_20'] * 100
df['Price_Z_Score_20'] = (df['Close'] - df['Close'].rolling(20).mean())/df['Close'].rolling(20).std()
```
</details>

<details><summary><strong>engineered_features.csv</strong></summary>
Purpose: Persistent snapshot of all baseline engineered indicators + Target for a run; acts as canonical feature set input to GA.  
How: Export post feature engineering & target creation; excludes later GA‑engineered AMA/FDI (those added in enhanced phase).  
When: Before GA feature selection; after dropping NaNs to guarantee alignment across future reloads.  
Where: working_data/engineered_features.csv (or raw_data/ in earlier iterations).  
Why: Reproducibility & audit—ensures that different GA parameter sweeps operate on identical base matrix.  
Relevance: feature_names.pkl index ordering matches this file; chromosome bit positions map 1‑to‑1 to its columns (excluding Target).  
Example:
```python
data.to_csv('working_data/engineered_features.csv')
```
</details>

<details><summary><strong>engineer_adaptive_moving_average</strong></summary>
Purpose: GA function that evolves parameters to produce AMA_GA_Optimized—an adaptive moving average responsive to efficiency ratio & volatility.  
How: Searches parameter vector [smoothing_factor, volatility_window, fast_period, slow_period]; fitness blends weighted F1 + stability + variance.  
When: After baseline GA feature selection (so its marginal benefit can be measured vs existing selected subset) in “Feature Engineering using GA” phase.  
Where: MH_Feature_Engineering.py; invoked in Combined_Metaheuristics_Workflow.ipynb Section 6.  
Why: Replace multiple fixed-lag MAs with one adaptive series to reduce dimensionality while preserving responsiveness.  
Relevance: Frequently improves cv_stability, allowing GA-enhanced models to prune redundant SMA/EMA distance features → higher feature reduction %.  
Example call:
```python
ama_feature, ama_details = engineer_adaptive_moving_average(data_ohlcv, target,
                                                            pop_size=30, max_generations=20)
```
</details>

<details><summary><strong>engineer_fractal_dimension_indicator</strong></summary>
Purpose: GA function producing FDI_GA_Optimized capturing geometric roughness (trend persistence vs choppiness).  
How: Evolves [window_size, high_low_factor, scaling_factor]; computes fractal dimension proxy over rolling window; fitness emphasizes predictive stability.  
When: After AMA optimization; before enhanced model training.  
Where: MH_Feature_Engineering.py; Combined workflow Section 6–7.  
Why: Distinguish trending (lower fractal dimension) from noisy regimes to condition momentum features.  
Relevance: Helps models avoid overfitting in high-noise segments, enhancing weighted F1 and stability without proliferating windowed volatility indicators.  
Example:
```python
fdi_feature, fdi_details = engineer_fractal_dimension_indicator(data_ohlcv, target,
                                                                pop_size=30, max_generations=20)
```
</details>

<details><summary><strong>feature crosses</strong></summary>
Purpose: Explicit interaction terms formed by multiplying or conditionally combining two (or more) features (e.g., DI_Diff_14 * BB_Width_20).  
How: Manual creation or automated enumeration (not currently active); interactions mimic conditional logic (momentum effectiveness under low volatility).  
When (future extension): After baseline engineering but before GA selection OR after initial GA to test incremental gain.  
Where: Potential addition in feature engineering notebooks.  
Why: Some predictive signal emerges only under joint conditions (volatility squeeze + volume surge).  
Relevance: Current reliance on tree/XGBoost implicit interactions; explicit crosses could reduce model depth needs but risk dimensionality bloat—would necessitate GA pruning to preserve sparsity.  
Example:
```python
df['DI_BB_Interaction'] = df['DI_Diff_14'] * df['BB_Width_20']
```
</details>

<details><summary><strong>encoding categorical features (one-hot, target encoding)</strong></summary>
Purpose: Convert categorical/time components (e.g., hour-of-day, day-of-week) into numeric form.  
Methods: One-Hot (binary columns), Target / Mean Encoding (replace category with aggregated target statistic).  
When (future): After raw ingestion if temporal seasonality features are added; before scaling & GA.  
Where: Not yet implemented—project presently numeric OHLCV only.  
Why: Intraday or weekday effects can modulate indicator significance (e.g., reduced volatility in specific sessions).  
Relevance: Adding categorical time features introduces potential leakage if not temporally restricted; GA could select minimal informative encodings to boost stability with few extra columns.  
Example (one-hot hour):
```python
df['Hour'] = df.index.hour
hour_dummies = pd.get_dummies(df['Hour'], prefix='Hour')
df = df.join(hour_dummies)
```
</details>

<details><summary><strong>polynomial features</strong></summary>
Purpose: Nonlinear expansions (squared, interaction powers) enabling linear / shallow models to approximate curvature.  
How: sklearn.preprocessing.PolynomialFeatures( degree=2, include_bias=False ).  
When (optional future): After scaling training set; restrict to a small subset to avoid explosion.  
Where: Not currently used (XGBoost/trees handle nonlinearity).  
Why: Could benefit logistic components in feature engineering GA if added; risk large dimensionality and multicollinearity.  
Relevance: Any introduction must rely on GA to prune redundancy; otherwise cv_stability deteriorates.  
Example:
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2, include_bias=False)
X_poly = poly.fit_transform(X_train[['SMA_20_Dist','RSI_14']])
```
</details>

<details><summary><strong>dimensionality reduction (PCA, ICA)</strong></summary>
Purpose: Project high-dimensional correlated features into a lower-dimensional latent space (PCA: variance directions; ICA: independent components).  
How: Fit on training subset only: pca = PCA(n_components=k).fit(X_train_scaled); transform splits.  
When (possible alternative): Pre-step before GA to shrink search space, or post GA to compress residual redundancy.  
Where: Not currently active—direct feature subset selection preferred for interpretability.  
Why: Reduce overfitting risk and computational cost on very wide engineered sets (e.g., 500+ indicators).  
Relevance: Subset GA retains interpretability; PCA would obscure direct feature names—trade-off may be considered if runtime grows super-linearly.  
Example:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=30, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
```
</details>

<details><summary><strong>embedding</strong></summary>
Purpose: Dense vector representation learned for categorical or discretized entities (e.g., hour, asset ticker) capturing similarity structure.  
How: Typically via neural network embedding layers; each category → trainable low-dim vector.  
When (future): If expanding to multi-asset or adding categorical regimes; before model fitting; included in ANN pipeline.  
Where: Not used—ANN simplified to RF in GA fitness.  
Why: Avoid sparse one-hot explosion; encode relational distance more efficiently.  
Relevance: Could complement GA by providing compact learned features; current tree/XGBoost methods do not directly use embeddings.  
Example (conceptual Keras):
```python
Embedding(input_dim=num_hours, output_dim=4, input_length=1)
```
</details>

<details><summary><strong>feature hashing</strong></summary>
Purpose: Map high-cardinality categories into fixed-size numeric vector via hash function (hashing trick) without explicit dictionary.  
How: hash(category) % n_features determines column index; signed hashing optionally.  
When: Large symbol sets / textual tags (not present currently).  
Where: Would appear in preprocessing pipeline before scaling.  
Why: Memory-efficient encoding when cardinality enormous; sacrifices collision interpretability.  
Relevance: Not required for current purely numeric set; documented for scalability if multi-asset categorical metadata added.  
Example (concept):
```python
col_index = hash(category_value) % 128
vector[col_index] += 1
```
</details>

<details><summary><strong>permutation importance</strong></summary>
Purpose: Post-hoc measure of feature contribution by shuffling a column and observing metric drop.  
How: For each feature, permute its values on validation/test set; recompute accuracy/F1; importance = baseline_metric − permuted_metric.  
When: After model training (baseline & enhanced) to validate GA-selected subset meaningfulness.  
Where: Could be added to analysis section after feature_importance CSV generation.  
Why: Model-agnostic and captures interaction contributions absent in simple impurity importances.  
Relevance: Confirms engineered AMA/FDI or retained subset features materially affect predictive stability; justifies feature reduction claims.  
Example:
```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_test, y_test, n_repeats=10, scoring='f1_weighted')
```
</details>

<details><summary><strong>recursive feature elimination (RFE)</strong></summary>
Purpose: Wrapper selection method that iteratively trains a model, ranks features (e.g., by coefficients/importances), and removes the least important until desired count reached.  
How: sklearn.feature_selection.RFE(estimator, n_features_to_select).  
When: Alternative / baseline comparator to GA subset selection.  
Where: Could be run on training set scaled matrix for benchmarking.  
Why: Deterministic (given model) contrast to stochastic GA; useful for validating GA advantage in stability & sparsity.  
Relevance: GA integrates cross-validation stability and diversity; RFE lacks explicit stability objective—comparison can highlight metaheuristic benefits.  
Example:
```python
from sklearn.feature_selection import RFE
rfe = RFE(RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1), n_features_to_select=40)
rfe.fit(X_train_scaled, y_train)
```
</details>

<details><summary><strong>LASSO-based selection</strong></summary>
Purpose: Use L1-regularized model (e.g., LogisticRegression(penalty='l1')) to drive sparse coefficient vector; non-zero coefficients imply selected features.  
How: Fit on standardized training data; tune C to balance sparsity vs performance (e.g., via TimeSeriesSplit).  
When: Baseline linear sparsity benchmark before / after GA to measure marginal gain in nonlinear contexts.  
Where: Optional notebook cell; not integrated into main workflow.  
Why: Fast convex method giving lower bound on necessary features.  
Relevance: LASSO may identify a minimal linear core; GA can augment with nonlinearly useful indicators (volatility, adaptive signals) beyond linear separability.  
Example:
```python
from sklearn.linear_model import LogisticRegression
lasso = LogisticRegression(penalty='l1', solver='saga', C=0.05, max_iter=2000)
lasso.fit(X_train_scaled, y_train)
selected = [f for f,c in zip(feature_names, lasso.coef_.sum(axis=0)) if c!=0]
```
</details>

<details><summary><strong>mutual information</strong></summary>
Purpose: Nonlinear dependency measure between feature and target capturing shared information (entropy reduction).  
How: sklearn.feature_selection.mutual_info_classif(X, y, discrete_features=False).  
When: Pre-filter step to optionally eliminate near-zero relevance features before GA to shrink search space.  
Where: Could precede GA initialization; record filtered list for reproducibility.  
Why: Reduces dimensionality and speeds GA convergence while retaining potentially informative attributes.  
Relevance: Must be applied carefully—over-filtering can remove synergistic features that only matter in interactions (which GA could exploit).  
Example:
```python
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X_train_scaled, y_train, random_state=42)
mi_df = pd.DataFrame({'feature': feature_names, 'MI': mi}).sort_values('MI', ascending=False)
```
Interpretation: Low MI does not guarantee uselessness (especially in multivariate context), hence GA still valuable.
</details>

## Machine Learning Concepts & Algorithms

<details><summary><strong>machine learning</strong></summary>
Purpose: Field enabling models to learn patterns from data without hard‑coded rules.  
How/When: Train algorithms on historical OHLCV + engineered indicators to predict future trend class (Target).  
Where: Baseline and enhanced model training sections (Random Forest, XGBoost).  
Why: Automates extraction of nonlinear relationships (volatility + momentum + adaptive signals) that manual rules would miss.  
Relevance (project): Core engine whose performance the GA seeks to stabilize and improve via feature subset optimization.  
Example: XGBoost classifier learns probability distribution over classes {0,1,2} using GA-selected features.
</details>

<details><summary><strong>supervised learning</strong></summary>
Purpose: Learn mapping X → y given labeled examples.  
How/When: Features (engineered indicators) paired with Target (0/1/2) after target creation phase.  
Where: Model training (baseline, enhanced) and GA fitness evaluation (cross-validation).  
Why: Explicit labels allow optimization of accuracy/F1 and cv_stability.  
Relevance: Entire GA fitness loop relies on labeled Target to score feature subsets.  
Example: RandomForestClassifier.fit(X_train_scaled, y_train).
</details>

<details><summary><strong>unsupervised learning</strong></summary>
Purpose: Discover structure in unlabeled data (clustering, dimensionality reduction).  
How/When (potential): Could cluster volatility regimes before GA to condition feature ranges.  
Where: Not currently implemented—documented as extension.  
Why: Regime clustering might guide adaptive thresholds or feature subset priors.  
Relevance: Future enhancement to seed GA initialization with regime-specific importance.  
Example (future): KMeans on volatility features to label calm vs volatile clusters.
</details>

<details><summary><strong>Ml Model (typo: ML model)</strong></summary>
Purpose: Computation artifact (estimator) mapping input features to predictions.  
How/When: Instantiated (RF, XGBoost) then fit on scaled chronological splits.  
Where: Baseline & enhanced model cells; inside GA (fitness_function creates model).  
Why: Provides performance signal for GA selection.  
Relevance: GA optimizes which features the ML model receives to maximize stability + accuracy.  
Example: rf = RandomForestClassifier(...).
</details>

<details><summary><strong>hyperparameter</strong></summary>
Purpose: Configuration value set prior to training (e.g., n_estimators, max_depth).  
How/When: Chosen for baseline; tuned (GridSearchCV) for enhanced set after GA.  
Where: Model definition cells.  
Why: Controls model capacity, bias‑variance balance.  
Relevance: After feature reduction, optimal hyperparameters may shift (shallower depth sufficient).  
Example: max_depth=10 in baseline RF.
</details>

<details><summary><strong>hyperparameter tuning</strong></summary>
Purpose: Systematic search for hyperparameter values improving validation metrics.  
How/When: GridSearchCV on combined train+validation (enhanced phase).  
Where: “Hyperparameter Tuning for Enhanced Models” section.  
Why: Align model complexity with reduced feature set; avoid under/overfitting.  
Relevance: Ensures reported uplift due to feature quality not suboptimal hyperparameters.  
Example: Grid search over {'n_estimators':[50,100,200]}.
</details>

<details><summary><strong>model evaluation</strong></summary>
Purpose: Quantify model performance on unseen chronological test segment.  
How/When: After training baseline and enhanced models; also per fold inside GA.  
Where: Evaluation functions (evaluate_model).  
Why: Validate generalization and compare baseline vs GA-enhanced.  
Relevance: Drives improvement claims (accuracy, precision, recall, f1_score).  
Example: accuracy_score(y_test, y_pred).
</details>

<details><summary><strong>evaluation metrics</strong></summary>
Purpose: Numerical measures (accuracy, precision, recall, weighted F1) summarizing predictive quality.  
How/When: Computed post prediction and per CV fold.  
Where: Baseline, enhanced comparison tables, GA fitness composition.  
Why: Weighted F1 mitigates class imbalance; stability term complements average performance.  
Relevance: Fitness = accuracy + stability − complexity_penalty; metrics guide selection.  
Example: f1_score(y_test, y_pred, average='weighted').
</details>

<details><summary><strong>cross validation</strong></summary>
Purpose: Partition training window into sequential folds to estimate out-of-sample robustness.  
How/When: GA fitness uses TimeSeriesSplit; grid search uses standard CV (but still chronological).  
Where: GeneticAlgorithm.fitness_function; GridSearchCV.  
Why: Reduces overfitting risk given non-stationarity.  
Relevance: cv_stability (1 − std(scores)) is a primary fitness component.  
Example: TimeSeriesSplit(n_splits=5).
</details>

<details><summary><strong>cross validation score</strong></summary>
Purpose: Per-fold metric value (e.g., accuracy) produced during CV.  
How/When: Collected inside GA to compute mean_accuracy and std for stability.  
Where: fitness_function.  
Why: Standard deviation across folds reveals sensitivity to temporal drift.  
Relevance: Lower variance boosts cv_stability term encouraging robust subsets.  
Example: scores = cross_val_score(model, X_sub, y, cv=cv).
</details>

<details><summary><strong>TimeSeriesSplit (relevant for CV)</strong></summary>
Purpose: Expanding-window splitter preserving temporal order.  
How/When: Each fold trains on earlier slice, tests on following slice.  
Where: GA fitness; previously defined earlier (cross-reference).  
Why: Prevents leakage inherent in shuffled KFold.  
Relevance: Guarantees unbiased stability measure for feature subsets.  
Example: for tr, te in TimeSeriesSplit(5).split(X): ...
</details>

<details><summary><strong>bias-variance tradeoff</strong></summary>
Purpose: Balance between underfitting (high bias) and overfitting (high variance).  
How/When: Managed via feature reduction, regularization (implicit in tree depth), tuning.  
Where: Rationale for GA (reduce variance by pruning noisy features).  
Why: Smaller, informative subset shifts to lower variance without large bias increase.  
Relevance: cv_stability directly reflects variance component reduction.  
Example: Feature subset shrinking from 120→40 lowers fold score variance.
</details>

<details><summary><strong>regularization (L1, L2, elastic net)</strong></summary>
Purpose: Penalize model complexity via coefficient shrinkage.  
How/When: Not primary (tree/XGBoost internal), but logistic in feature engineering GA implicitly uses L2.  
Where: FeatureEngineeringGA (LogisticRegression).  
Why: Prevent overfitting single engineered feature parameter sets.  
Relevance: Ensures AMA/FDI fitness signals reflect genuine predictive value.  
Example: LogisticRegression(penalty='l2').
</details>

<details><summary><strong>early stopping</strong></summary>
Purpose: Halt training when validation metric ceases improving.  
How/When (potential): Could be applied to XGBoost (early_stopping_rounds).  
Where: Not yet implemented; future runtime optimization.  
Why: Avoid overfitting late boosting rounds.  
Relevance: Shortens experiment loops enabling more GA parameter sweeps.  
Example (future): xgb.fit(..., eval_set=[(X_val,y_val)], early_stopping_rounds=20).
</details>

<details><summary><strong>batch size</strong></summary>
Purpose: Number of samples per gradient update in neural nets.  
How/When: Not active (ANN replaced by RF in GA); recorded for completeness.  
Relevance: If ANN reintroduced, impacts convergence stability affecting cv_stability.  
Example: batch_size=64 in Keras model (future).
</details>

<details><summary><strong>epochs</strong></summary>
Purpose: Full passes over training data in neural networks.  
How/When: Currently unused (ANN simplified); concept retained for extensibility.  
Relevance: Too many epochs increase variance; early stopping would harmonize with stability target.  
Example: model.fit(..., epochs=50).
</details>

<details><summary><strong>learning rate scheduler</strong></summary>
Purpose: Adjust learning rate during training to balance speed and convergence.  
How/When: Potential for future ANN/XGBoost tuning (eta decay).  
Relevance: Stable convergence can reduce variability across folds.  
Example: lr * 0.9 every 10 epochs.
</details>

<details><summary><strong>gradient clipping</strong></summary>
Purpose: Limit gradient magnitude to prevent exploding updates.  
How/When: Future ANN introduction.  
Relevance: Enhances reproducibility and stability, aligning with cv_stability objective.  
Example: clipnorm=1.0 in optimizer.
</details>

<details><summary><strong>ensembling (bagging, boosting, stacking)</strong></summary>
Purpose: Combine multiple learners to improve predictive performance and robustness.  
How/When: Bagging (Random Forest baseline), Boosting (XGBoost), potential stacking future.  
Where: Model training cells.  
Why: Reduce variance or bias vs single model.  
Relevance: GA-selected sparse features often amplify ensemble efficiency.  
Example: Compare RF vs XGBoost improvements after feature selection.
</details>

<details><summary><strong>bagging</strong></summary>
Purpose: Bootstrap aggregation builds parallel trees on resampled data (Random Forest).  
How/When: Baseline RF.  
Why: Lowers variance; supports stability goal.  
Relevance: Feature pruning decreases correlation among trees enhancing bagging gains.  
Example: RandomForestClassifier(n_estimators=100).
</details>

<details><summary><strong>boosting (XGBoost, LightGBM, CatBoost)</strong></summary>
Purpose: Sequential learners correct predecessors’ errors.  
How/When: XGBoost used; LightGBM/CatBoost potential.  
Why: Captures complex nonlinear interactions.  
Relevance: Reduced feature set shortens boosting rounds needed, mitigating overfit risk.  
Example: XGBClassifier(max_depth=6, learning_rate=0.1).
</details>

<details><summary><strong>stacking / blending</strong></summary>
Purpose: Meta-model combines predictions of base models.  
How/When: Not yet implemented; future enhancement.  
Why: Exploit complementary error structures.  
Relevance: GA could run per base model then stack selected outputs.  
Example (future): Meta logistic on RF + XGB probabilities.
</details>

<details><summary><strong>Bayesian optimization (Optuna, hyperopt)</strong></summary>
Purpose: Model-based search of hyperparameter space using surrogate (e.g., TPE).  
How/When: Could replace grid for efficiency on high-dimensional spaces.  
Why: Fewer evaluations for comparable performance.  
Relevance: Frees compute for broader GA experimentation.  
Example (future): Optuna optimizing mutation_rate, crossover_rate jointly with model depth.
</details>

<details><summary><strong>grid search / random search</strong></summary>
Purpose: Exhaustive or sampled hyperparameter exploration.  
How/When: GridSearchCV used for enhanced models.  
Why: Determine best configuration post feature reduction.  
Relevance: Ensures feature gains not confounded by poor hyperparameters.  
Example: Grid over n_estimators × max_depth.
</details>

<details><summary><strong>callbacks</strong></summary>
Purpose: Hooks executed during training (e.g., early stopping, logging).  
How/When: Future ANN or boosting early stopping integration.  
Relevance: Can log per-epoch metrics to external tracking improving reproducibility.  
Example: EarlyStopping(monitor='val_loss').
</details>

<details><summary><strong>neural networks</strong></summary>
Purpose: Layered nonlinear function approximators (ANN, CNN, RNN).  
How/When: Potential future replacement for simplified ANN placeholder.  
Why: Capture complex temporal/feature interactions.  
Relevance: GA feature subset can reduce overfitting risk if deep models added.  
Example: Simple dense network on selected features.
</details>

<details><summary><strong>ANN</strong></summary>
Purpose: Feedforward Artificial Neural Network.  
How/When: Placeholder model type (mapped to RF internally for GA fitness to save runtime).  
Why: Simplifies pipeline while reserving interface extensibility.  
Relevance: Prevents heavy GPU requirements yet keeps consistent API.  
Example: model_type='ann' in GA currently instantiates RandomForest.
</details>

<details><summary><strong>RNN</strong></summary>
Purpose: Recurrent Neural Network handling sequential dependencies.  
How/When: Future extension using sequences of candles.  
Why: Could leverage temporal order beyond static indicators.  
Relevance: Might reduce need for some lagged engineered features.  
Example (future): LSTM over sliding window features.
</details>

<details><summary><strong>LSTM</strong></summary>
Purpose: RNN variant with gating (Long Short-Term Memory) capturing long-range dependencies.  
How/When: Potential future sequential modeling on raw OHLCV.  
Why: Handles regime durations more effectively than plain RNN.  
Relevance: Could absorb some volatility/momentum indicators, shrinking feature engineering burden.  
Example: Keras LSTM layer consuming normalized windows.
</details>

<details><summary><strong>CNN</strong></summary>
Purpose: Convolutional Neural Network performing local pattern extraction.  
How/When: Potential on transformed time-series (e.g., rolling windows as channels).  
Why: Detect repeating micro-patterns (candle formations).  
Relevance: Could automate pattern features (engulfing) currently indirect.  
Example: 1D conv layers over feature sequences.
</details>

<details><summary><strong>reinforcement learning</strong></summary>
Purpose: Learn policy via reward feedback (sequential decision).  
How/When: Not used; future for trading strategy on model outputs.  
Why: Directly optimizes profit-based objective including costs.  
Relevance: Classification probabilities could be state inputs for RL agent.  
Example (future): Agent decides long/flat using class probability vector.
</details>

<details><summary><strong>ML pipeline</strong></summary>
Purpose: Ordered steps (engineering → scaling → selection → modeling → evaluation → persistence).  
How/When: Implemented explicitly in combined workflow notebook.  
Why: Enforces reproducibility and separation of concerns.  
Relevance: run_timestamp directory captures each pipeline artifact enabling audit & reruns.  
Example: Feature engineering → TimeSeriesSplit → GA → tuned model → save run_summary.json.
</details>

<details><summary><strong>structured tabular data</strong></summary>
Purpose: Row/column formatted dataset (each row = time index, columns = indicators).  
How/When: All processing uses pandas DataFrame of engineered features + Target.  
Why: Compatible with tree and boosting algorithms; facilitates mask-based selection.  
Relevance: GA individuals map directly to column indices (feature_names.pkl).  
Example: X_train_scaled.shape = (N, F_subset).
</details>

<details><summary><strong>stochastic nature</strong></summary>
Purpose: Randomness inherent in GA (mutation, crossover sampling) and model training (bootstrap, feature subsampling).  
How/When: Controlled via random_state seeds.  
Why: Ensures comparability across runs; randomness aids exploration.  
Relevance: Diversity strategies rely on stochastic variation; reproducibility archived in run_* directories.  
Example: np.random.seed(42) before GA initialization.
</details>

<details><summary><strong>ANN (note: simplified to RF for GA fitness)</strong></summary>
Purpose: Project-specific alias mapping to Random Forest to avoid heavy NN runtime during GA.  
How/Why: Maintains interface consistency while accelerating fitness evaluation.  
Relevance: Allows future swap to real ANN without refactoring GA code path.  
Example: GeneticAlgorithm(..., model_type='ann') → internally create RF model.
</details>

## Common tools & algorithms:

<details><summary><strong>RandomForestClassifier()</strong></summary>
Purpose: Ensemble (bagging) classifier of many decision trees voting on class labels.  
How/When/Where (project): Used as a baseline and enhanced model in Combined_Metaheuristics_Workflow.ipynb (Sections: Baseline Model Training, Enhanced Model Training) and internally for GA fitness when model_type='random_forest' or 'ann'.  
Why: Robust to noisy, heterogeneous engineered technical indicators; handles nonlinear feature interactions without heavy preprocessing.  
Relevance: Serves as (a) baseline performance benchmark, (b) proxy evaluator inside GA to score feature subsets rapidly, (c) comparison point for XGBoost uplift after feature selection and engineering.  
Key Benefits: Handles high-dimensional sparse subsets after GA reduction; less sensitive to scaling than linear methods.  
Example:
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
pred = rf.predict(X_test_scaled)
```
</details>

<details><summary><strong>n_estimators</strong></summary>
Purpose: Number of trees (estimators) in ensemble models (Random Forest, XGBoost, etc.).  
How/When: Set during model instantiation (e.g., 100 for baseline; tuned via GridSearchCV in enhanced phase).  
Why: More trees reduce variance up to diminishing returns; too many increase runtime.  
Relevance: After GA feature reduction fewer trees may achieve same stability—tuning ensures efficiency.  
Guideline: Start 100; increase if out-of-bag or validation variance remains high.  
Example: `RandomForestClassifier(n_estimators=200, ...)`
</details>

<details><summary><strong>max_depth</strong></summary>
Purpose: Maximum depth of each decision tree (stops uncontrolled growth).  
How/When: Fixed (e.g., 10) in baseline; searched in grid for enhanced models.  
Why: Controls overfitting (deep trees memorize small temporal artifacts).  
Relevance: With cleaner GA-selected subsets, shallower optimal depth often emerges, improving generalization and runtime.  
Example: `max_depth=10` vs `None` (unbounded).
</details>

<details><summary><strong>random_state</strong></summary>
Purpose: Seed controlling pseudorandom behavior (bootstrapping, feature subsampling, GA randomness).  
How/When: Set globally (np.random.seed(42)) and per estimator (random_state=42).  
Why: Ensures reproducibility across run_* directories for fair baseline vs enhanced comparison.  
Relevance: Critical for validating that performance uplift stems from feature subset quality, not chance variation.  
Example: `XGBClassifier(random_state=42, ...)`
</details>

<details><summary><strong>n_job / n_jobs</strong></summary>
Purpose: Parallel CPU core usage parameter (scikit-learn estimators use n_jobs; typo n_job may appear in notes).  
How/When: Set to -1 to utilize all available cores for RandomForest, GridSearchCV, and joblib Parallel in GA evaluation.  
Why: Reduces wall-clock time for feature subset evaluation and model training.  
Relevance: GA iterations evaluate many individuals; parallelization keeps experimentation feasible.  
Example: `RandomForestClassifier(n_jobs=-1)`.  
Note: Always use n_jobs (plural) in sklearn; adjust downward if memory pressure observed.
</details>

<details><summary><strong>learning_rate</strong></summary>
Purpose: Step size shrinkage for boosting algorithms (XGBoost, LightGBM, CatBoost).  
How/When: XGBoost baseline uses 0.1; tuned in grid search for enhanced models.  
Why: Lower values improve generalization but require more trees; higher values risk overfitting.  
Relevance: After GA reduction, informative sparse features allow moderately lower learning_rate without excessive boosting rounds.  
Example: `XGBClassifier(learning_rate=0.05, n_estimators=200, ...)`
</details>

<details><summary><strong>eval_metric</strong></summary>
Purpose: Metric used internally by boosting library (e.g., 'logloss' for XGBoost) during training / early stopping.  
How/When: Set in XGBoost instantiation (`eval_metric='logloss'`) in baseline and enhanced models.  
Why: Guides objective monitoring; classification with class imbalance benefits from logloss visibility.  
Relevance: Consistent eval_metric ensures comparable performance logs across runs while GA focuses on weighted F1 + stability externally.  
Example: `XGBClassifier(eval_metric='logloss', ...)`
</details>

<details><summary><strong>ANN (simplified to RF for GA fitness)</strong></summary>
Purpose: Placeholder model type mapped to RandomForest to avoid heavy neural network training during GA feature selection.  
How/When: Passing model_type='ann' in GeneticAlgorithm creates an RF internally.  
Why: Speeds experimentation while keeping interface extensibility (future swap to true neural network).  
Relevance: Allows consistent GA API without incurring GPU or long epoch cycles; stability metric remains comparable.  
Example:
```python
ga = GeneticAlgorithm(X_train, y_train, feature_names, model_type='ann')
# Internally uses RandomForestClassifier
```
</details>

<details><summary><strong>XGBoost</strong></summary>
Purpose: Gradient boosting library (tree-based) optimizing additive model with second-order gradients.  
How/When: Baseline + enhanced classifier; evaluated after GA feature selection and feature engineering (AMA/FDI).  
Why: Captures nonlinear interactions and conditional feature effects (e.g., volatility + momentum conjunctions).  
Relevance: Often demonstrates larger uplift from GA because removal of redundant features reduces overfitting and speeds boosting convergence.  
Key Params Used: n_estimators=100, max_depth=6, learning_rate=0.1 (baseline).  
Example:
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)
```
</details>

<details><summary><strong>LightGBM</strong></summary>
Purpose: Gradient boosting framework using histogram-based decision tree learning (not yet integrated).  
How/When (potential): Could replace / complement XGBoost in enhanced model stage for faster training on wide engineered feature sets.  
Why: Efficient on large feature spaces; supports categorical handling natively.  
Relevance: Future extension—would allow benchmarking GA feature subset portability across boosting libraries.  
Example (conceptual): `lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=200)`.
</details>

<details><summary><strong>CatBoost</strong></summary>
Purpose: Gradient boosting library optimized for categorical features via ordered boosting (not currently used).  
How/When (future): Could be added when time-derived categorical (hour-of-day) or multi-asset identifiers introduced.  
Why: Reduces need for one-hot encoding; mitigates prediction shift issues.  
Relevance: Once categorical time/asset features appear, GA can assess CatBoost vs existing models for stability gain with fewer engineered transformations.  
Example (conceptual): `CatBoostClassifier(depth=6, learning_rate=0.1, iterations=300)`.
</details>

<details><summary><strong>LogisticRegression</strong></summary>
Purpose: Linear classifier modeling class log-odds; used with regularization.  
How/When: Employed inside FeatureEngineeringGA (MH_Feature_Engineering.py) to score AMA/FDI parameter candidates via time-series CV (single-feature predictive utility).  
Why: Fast, interpretable, low variance baseline to evaluate marginal contribution of a single engineered feature without nonlinear confounding.  
Relevance: Improves reliability of feature engineering fitness—avoids overfitting parameter search with powerful nonlinear learners.  
Example:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_feat, y)
```
</details>

<details><summary><strong>Support Vector Machine (SVM)</strong></summary>
Purpose: Margin-based classifier (linear or kernel) maximizing separation between classes.  
How/When (not currently in pipeline): Could be added as alternative model_type for GA fitness or benchmark after feature reduction.  
Why: With GA-pruned informative subset, SVM (especially linear) may achieve competitive performance efficiently.  
Relevance: Adds diversity to evaluation—helps confirm that GA-selected features generalize across algorithm families (tree vs margin-based).  
Example (conceptual):
```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_selected, y_train)
```
</details>

<details><summary><strong>accuracy</strong></summary>
Purpose: Proportion of correct predictions ( (TP+TN)/Total ).  
How/When/Where: Computed after model predictions on validation/test sets and within GA folds (implicitly via cross_val_score when scoring='accuracy').  
Why: Simple overall correctness metric.  
Relevance (project): Can be misleading under class imbalance (Sideways dominating); thus supplemented with precision/recall/F1 and cv_stability.  
Example:
```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
```
</details>

<details><summary><strong>precision</strong></summary>
Purpose: TP / (TP + FP); correctness of positive (or each class) predictions.  
How/When: Calculated per class then combined via averaging scheme.  
Why: Penalizes false positives (important if predicting Up/Down wrongly could cause trades).  
Relevance: Weighted precision indicates reliability of directional classes vs dominant Sideways.  
Example:
```python
from sklearn.metrics import precision_score
p = precision_score(y_test, y_pred, average='weighted')
```
</details>

<details><summary><strong>recall</strong></summary>
Purpose: TP / (TP + FN); ability to capture actual positives.  
Why: Measures missed signals (false negatives).  
Relevance: Ensures minority direction classes (0 / 1) are detected, preventing a trivial Sideways-heavy model.  
Example:
```python
r = recall_score(y_test, y_pred, average='weighted')
```
</details>

<details><summary><strong>f1_score</strong></summary>
Purpose: Harmonic mean of precision & recall (balances both).  
Why: Stable metric when classes imbalanced.  
Relevance: Primary comparative metric baseline vs GA-enhanced; uplift indicates better directional discrimination without overfitting.  
Example:
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')
```
</details>

<details><summary><strong>macro / micro / weighted averaging</strong></summary>
Purpose: Strategies to aggregate per-class metrics.  
How: macro = unweighted mean; micro = global TP/FP/FN aggregation; weighted = class-frequency-weighted mean.  
Why: Weighted chosen to reflect true distribution while penalizing minority misclassification.  
Relevance: Project mainly reports weighted to avoid Sideways dominance bias.  
Example:
```python
precision_score(y_test, y_pred, average='macro')
```
</details>

<details><summary><strong>R2 Score</strong></summary>
Purpose: Variance explained (regression).  
Why: Not primary (classification task) but relevant if future regression targets (e.g., return magnitude) added.  
Relevance: Currently minimal; included for completeness.  
Example:
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true_reg, y_pred_reg)
```
</details>

<details><summary><strong>MSE</strong></summary>
Purpose: Mean Squared Error (regression loss).  
Why: Sensitive to large errors; used in potential regression extensions.  
Relevance: Appears indirectly in engineered objective discussion (if modeling continuous volatility).  
Example:
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```
</details>

<details><summary><strong>RMSE</strong></summary>
Purpose: sqrt(MSE); same units as target.  
Relevance: Interpretability if forecasting numeric values added later (e.g., volatility).  
Example:
```python
rmse = mean_squared_error(y_true, y_pred, squared=False)
```
</details>

<details><summary><strong>MAE</strong></summary>
Purpose: Mean Absolute Error; robust to outliers vs MSE.  
Relevance: Possible alternative in regression-style feature engineering fitness.  
Example:
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```
</details>

<details><summary><strong>MSE + L2 Reg</strong></summary>
Purpose: Objective MSE + λ * ||w||² (Ridge); penalizes large weights.  
Why: Controls overfitting, improves generalization.  
Relevance: Analogous to stability emphasis; logistic/L2 inside GA engineering mirrors this regularization principle.  
Example (concept):
```python
loss = mse + alpha * np.sum(coef**2)
```
</details>

<details><summary><strong>log loss</strong></summary>
Purpose: Negative log-likelihood for probabilistic classification; penalizes miscalibrated confident errors.  
How: Used internally by XGBoost (eval_metric='logloss').  
Relevance: Lower log loss with fewer features signals better calibrated subset.  
Example:
```python
from sklearn.metrics import log_loss
ll = log_loss(y_test, proba)
```
</details>

<details><summary><strong>cross-entropy</strong></summary>
Purpose: General form of log loss (classification entropy between true and predicted distributions).  
Why: Measures information inefficiency.  
Relevance: Conceptual underpinning of probability calibration and potential focal loss extension.  
Example: Same as log loss in multinomial setting.
</details>

<details><summary><strong>AUC ROC</strong></summary>
Purpose: Area under ROC curve (TPR vs FPR).  
How: One-vs-rest for multiclass.  
Why: Threshold-independent discrimination.  
Relevance: Secondary metric if evaluating ranking quality of probabilities (optional extension).  
Example:
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, proba, multi_class='ovr')
```
</details>

<details><summary><strong>PR AUC (Precision-Recall AUC)</strong></summary>
Purpose: Area under Precision–Recall curve.  
Why: More informative than ROC under class imbalance.  
Relevance: Evaluates model’s ability to retrieve minority Up/Down classes.  
Example:
```python
from sklearn.metrics import average_precision_score
ap = average_precision_score(y_test==1, proba_class1)
```
</details>

<details><summary><strong>Brier score</strong></summary>
Purpose: Mean squared error of probabilistic forecasts (calibration + refinement).  
Why: Lower = better calibrated probabilities.  
Relevance: Useful if probability quality required for risk-adjusted decisions (position sizing).  
Example:
```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_binary, proba_positive)
```
</details>

<details><summary><strong>calibration curve</strong></summary>
Purpose: Plot predicted probability vs actual frequency.  
Why: Detect over/under-confidence.  
Relevance: After GA selection, improved calibration may emerge (less noisy features).  
Example:
```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_binary, proba, n_bins=10)
```
</details>

<details><summary><strong>confusion matrix (TP, TN, FP, FN)</strong></summary>
Purpose: Tabulates prediction outcomes per class.  
Why: Basis for precision/recall/F1.  
Relevance: Inspect misclassification shifts after feature reduction (e.g., fewer false Up signals).  
Example:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
</details>

<details><summary><strong>Matthews correlation coefficient</strong></summary>
Purpose: Balanced quality metric using all confusion matrix terms (−1 to 1).  
Why: Robust under imbalance.  
Relevance: Optional robustness check that GA improvement not superficial.  
Example:
```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)
```
</details>

<details><summary><strong>Cohen's kappa</strong></summary>
Purpose: Agreement vs chance-adjusted baseline.  
Why: Controls for class prevalence.  
Relevance: Confirms gains not due to dominant Sideways prediction.  
Example:
```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test, y_pred)
```
</details>

<details><summary><strong>top-k accuracy</strong></summary>
Purpose: Whether true class is within model’s top k probability ranks.  
Why: Relevant if downstream strategy can consider multiple candidate directions.  
Relevance: Could evaluate probabilistic ranking quality (future extension).  
Example:
```python
import numpy as np
def topk_acc(proba, y, k=2):
    return np.mean([y[i] in np.argsort(proba[i])[-k:] for i in range(len(y))])
```
</details>

<details><summary><strong>class-weighting</strong></summary>
Purpose: Adjust loss to penalize errors on minority classes more.  
How: class_weight='balanced' or manual dict.  
Why: Mitigates imbalance (Sideways majority).  
Relevance: Could enhance recall for Up/Down during GA fitness if integrated.  
Example:
```python
RandomForestClassifier(class_weight='balanced')
```
</details>

<details><summary><strong>focal loss</strong></summary>
Purpose: Down-weights easy examples, focuses on hard/minority instances.  
Why: Improves minority class learning.  
Relevance: Potential XGBoost / custom objective extension to further stabilize directional recall.  
Formula (binary): FL = −α (1−p)^γ log(p).  
</details>

<details><summary><strong>cross-validation stability (cv_stability concept)</strong></summary>
Purpose: 1 − std(CV scores); measures consistency across TimeSeriesSplit folds.  
How: fitness = mean_accuracy + cv_stability − complexity_penalty.  
Why: Prioritizes robust feature subsets resilient to temporal drift.  
Relevance: Core GA innovation guiding selection beyond raw accuracy.  
Example (concept):
```python
scores = cross_val_score(model, X_sub, y, cv=tscv)
cv_stability = max(0.0, 1.0 - scores.std())
```
</details>

<details><summary><strong>bootstrap confidence intervals</strong></summary>
Purpose: Resample test predictions to estimate metric uncertainty (e.g., F1 95% CI).  
Why: Quantify statistical significance of GA improvement.  
Relevance: Future enhancement for run_summary rigor.  
Example:
```python
import numpy as np
def bootstrap_ci(metric_fn, y, yhat, B=1000):
    vals = []
    n = len(y)
    for _ in range(B):
        idx = np.random.randint(0, n, n)
        vals.append(metric_fn(y[idx], yhat[idx]))
    return np.percentile(vals, [2.5, 97.5])
```
</details>

## Statistical / Optimization Foundations

<details><summary><strong>gradient descent</strong></summary>
Purpose: Iterative optimization algorithm that updates parameters in the negative direction of the gradient of a loss function to find (local) minima.  
How/When/Where: Core for training differentiable models (neural nets, logistic regression). Each step: θ ← θ − η ∇L(θ). Not directly used for Random Forest / XGBoost (they use greedy splits / additive boosting), but concept underlies ANN extensions or future differentiable objectives.  
Why: Closed‑form solutions (like OLS) fail for complex/nonlinear high‑dimensional loss landscapes.  
Relevance (project): Explains why ANN (if reintroduced) would need learning rate scheduling & potential adaptive optimizers; contrasts with tree methods (no gradient descent on full parameter vector). In GA feature engineering, logistic regression (used for AMA/FDI fitness) is solved via an internal gradient-based optimizer.  
Example:
```python
# Pseudo single step
theta = theta - lr * grad_loss(theta)
```
</details>

<details><summary><strong>sigmoid</strong></summary>
Purpose: Activation / squashing function σ(x)=1/(1+e^{-x}) mapping ℝ → (0,1).  
How/When/Where: Used in binary logistic regression output layer; potentially in ANN final layer for binary targets.  
Why: Transforms linear score into probability.  
Relevance: Target here is 3-class, so softmax is more appropriate; sigmoid appears conceptually when discussing logistic baseline used in feature engineering GA (one-vs-rest internal).  
Example:
```python
import numpy as np
p = 1/(1+np.exp(-z))
```
</details>

<details><summary><strong>ReLU</strong></summary>
Purpose: Rectified Linear Unit activation: ReLU(x)=max(0,x).  
How/When/Where: Standard hidden-layer activation in deep nets; not used in current RF/XGBoost pipeline but relevant if ANN added.  
Why: Mitigates vanishing gradients vs sigmoid/tanh; computationally cheap.  
Relevance: If ANN replaces RF proxy inside GA, ReLU impacts convergence speed and stability metrics (cv_stability).  
Example:
```python
relu = np.maximum(0, z)
```
</details>

<details><summary><strong>softmax</strong></summary>
Purpose: Converts logits z_k to class probabilities: p_k = exp(z_k)/Σ_j exp(z_j).  
How/When/Where: Multiclass classification output layer (for Target classes 0/1/2 in an ANN).  
Why: Produces normalized probability simplex enabling cross-entropy loss.  
Relevance: Current models (RF, XGBoost) internally approximate class probabilities similar to softmax; understanding softmax justifies evaluating probability-calibrated metrics (log loss, potential calibration curves).  
Example:
```python
import numpy as np
z = np.array([1.2, 0.4, -0.3])
p = np.exp(z)/np.exp(z).sum()
```
</details>

<details><summary><strong>MLE (Maximum Likelihood Estimation)</strong></summary>
Purpose: Parameter estimation maximizing likelihood of observed data under a probabilistic model.  
How/When/Where: Logistic regression coefficients, Gaussian assumptions in volatility modeling, underlying XGBoost objective (approximate additive MLE).  
Why: Provides statistically grounded parameter estimates with asymptotic properties.  
Relevance: Fitness components (accuracy / stability) are empirical; MLE perspective explains why models produce probability estimates used for stability evaluation (variance across folds).  
Example (binary logistic): maximize Π_i p_i^{y_i}(1-p_i)^{1-y_i}.  
</details>

<details><summary><strong>OLS (Ordinary Least Squares)</strong></summary>
Purpose: Linear regression estimator minimizing Σ (y - Xβ)^2; closed-form β=(XᵀX)^{-1}Xᵀy (if invertible).  
How/When/Where: Not directly used (classification focus) but conceptual baseline for understanding feature multicollinearity (XᵀX ill-conditioned → unstable).  
Why: Simplest parametric estimator; frames need for regularization or nonlinear models.  
Relevance: Multicollinearity among engineered indicators motivates GA feature pruning (improves conditioning akin to reducing variance of OLS-style estimators).  
Example:
```python
beta = np.linalg.pinv(X.T @ X) @ X.T @ y
```
</details>

<details><summary><strong>eigen vectors (eigenvectors)</strong></summary>
Purpose: Directions v satisfying A v = λ v for matrix A; λ eigenvalues.  
How/When/Where: Underlies PCA (covariance matrix decomposition), condition number evaluation (ratio of largest/smallest singular values).  
Why: Reveal dominant variance directions / redundancy.  
Relevance: Justifies potential dimensionality reduction (PCA) pre-GA if feature space grows; large eigenvalue spread → instability; GA acts as a discrete alternative to projecting onto eigenvectors.  
Example (NumPy):
```python
w, V = np.linalg.eig(cov_matrix)
```
</details>

<details><summary><strong>entropy</strong></summary>
Purpose: Measure of uncertainty: H(p)=−Σ p_i log p_i.  
How/When/Where: Impurity criterion in decision trees (information gain), basis for log loss.  
Why: Encourages splits that reduce class uncertainty.  
Relevance: Random Forest and XGBoost splits indirectly optimize entropy-based metrics; lower test entropy often aligns with higher cv_stability after feature pruning.  
Example: If class probs = [0.7,0.2,0.1], H≈1.157 bits.  
</details>

<details><summary><strong>KL divergence (kl divergence)</strong></summary>
Purpose: Asymmetric measure of distribution difference: KL(P||Q)=Σ P log(P/Q).  
How/When/Where: Conceptual for model calibration (how predicted distribution diverges from empirical); used in some boosting / deep learning objectives.  
Why: Penalizes misallocation of probability mass.  
Relevance: Stability emphasis aims to reduce fold-to-fold distribution drift; implicitly lowers expected KL between fold predictive distributions and true distribution.  
Example:
```python
KL = np.sum(p * np.log(p / q))
```
</details>

<details><summary><strong>log transform</strong></summary>
Purpose: Apply log/ log1p to compress skew and stabilize variance.  
How/When/Where: Used for TickVol_Log1p, returns (log returns).  
Why: Reduces impact of large outliers; approximates additive structure.  
Relevance: Improves scaling consistency feeding GA fitness (reduces variance of fold scores).  
Example:
```python
df['TickVol_Log1p'] = np.log1p(df['Tick_Volume'])
```
</details>

<details><summary><strong>SVD (Singular Value Decomposition)</strong></summary>
Purpose: Factorization A = U Σ Vᵀ; singular values in Σ indicate rank/energy distribution.  
How/When/Where: Foundation for PCA (on centered data), diagnosing redundancy.  
Why: Identifies low-information directions ripe for pruning.  
Relevance: GA feature selection is a combinatorial analog to truncating small singular value directions (reduces dimensionality while preserving predictive variance).  
Example:
```python
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
```
</details>

<details><summary><strong>Lagrange multiplier (langrange multiplier)</strong></summary>
Purpose: Technique for constrained optimization by augmenting objective: L(x,λ)=f(x)+λ g(x).  
How/When/Where: Conceptual—could frame feature selection with constraint Σ mask_i ≤ K.  
Why: Converts constrained problem to saddle-point search.  
Relevance: Current GA uses penalty-like complexity component (implicit Lagrangian idea) in fitness (accuracy + stability − complexity_penalty).  
Example (constraint g(x)=0): solve ∇_x L=0 and ∂L/∂λ=0.  
</details>

<details><summary><strong>AIC / BIC</strong></summary>
Purpose: Information criteria balancing fit vs complexity:  
AIC = 2k − 2 ln(L); BIC = k ln(n) − 2 ln(L).  
How/When/Where: Model selection among parametric sets.  
Why: Prevent overfitting by penalizing excess parameters.  
Relevance: Analogous to GA’s complexity penalty (min_features/max_features) controlling subset size; conceptually justifies penalizing large feature masks.  
Example: Lower AIC/BIC preferred; feature subset with marginal accuracy gain but large size may be rejected similarly.  
</details>

<details><summary><strong>convex vs non-convex optimization</strong></summary>
Purpose: Convex problems have single global minimum; non-convex have multiple local minima.  
How/When/Where: OLS & logistic (convex); neural nets, GA objective (subset selection) non-convex/discrete.  
Why: Non-convexity motivates metaheuristics (GA, simulated annealing).  
Relevance: Feature subset search is combinatorial non-convex → gradient methods unsuitable, validating GA + dynamic rates + natural disasters.  
Example: Boolean mask search space 2^F not convex; GA explores stochastically.  
</details>

<details><summary><strong>stochastic nature</strong></summary>
Purpose: Randomness in algorithm steps (mutation, bootstrap sampling, parameter initialization).  
How/When/Where: GA mutation/crossover, Random Forest bagging, train/validation fold composition.  
Why: Promotes exploration; averages reduce variance.  
Relevance: Necessitates fixed random_state & run directories for reproducibility; informs interpreting small metric deltas (need multiple runs).  
Example: Two GA runs may choose slightly different but similarly performant subsets.  
</details>

<details><summary><strong>simulated annealing</strong></summary>
Purpose: Metaheuristic mimicking cooling process; probabilistically accepts worse moves early (temperature T) to escape local minima.  
How/When/Where: Not implemented yet; candidate extension for feature selection or engineered parameter tuning.  
Why: Simple single-solution exploration alternative to population-based GA.  
Relevance: Could benchmark against GA dynamic rates; especially for high-dimensional continuous parameter tuning (e.g., AMA variants).  
Example Acceptance:
```python
accept = np.exp(-(E_new - E_old)/T) > np.random.rand()
```
</details>

<details><summary><strong>NFL (No Free Lunch) theorem</strong></summary>
Purpose: States that averaged over all possible problems, all optimization algorithms perform equally—no universally superior method.  
How/When/Where: Theoretical underpinning for selecting algorithm per problem structure.  
Why: Justifies tailoring GA strategies (dynamic mutation/crossover, natural disasters) to financial time-series specifics instead of assuming one generic procedure.  
Relevance: Encourages documenting assumptions (temporal ordering, stability metric) making algorithm effective here; invites comparing with alternative heuristics (simulated annealing, PSO) for robustness.  
Example: GA outperforming RFE here does not imply universal superiority—must validate on new instruments/timeframes.  
</details>

<details><summary><strong>No Free Lunch theorem</strong></summary>
Purpose: Alias of NFL entry above.  
Relevance: See “NFL (No Free Lunch) theorem” toggle; included for completeness / search convenience.  
</details>

## Explainability & Interpretability

<details><summary><strong>SHAP (Shapley Additive Explanations)</strong></summary>
Purpose: Framework using Shapley values from cooperative game theory to attribute each feature's marginal contribution to a specific prediction or overall model output (additive feature attribution).  
How/When/Where: Applied after training baseline or GA‑enhanced models (Random Forest / XGBoost) to (a) validate that GA kept genuinely influential features, (b) compare importance patterns pre/post feature reduction, (c) assess engineered AMA_GA_Optimized / FDI_GA_Optimized impact. Use on validation or test set (never training-only for narrative) once feature subsets finalized.  
Why: Provides consistent, locally accurate attributions even under correlated features (though correlations still complicate interpretation). Explains directional influence (positive/negative push toward a class or log‑odds).  
Relevance (project): Confirms GA’s feature subset retains high-contributing indicators; detects if removal introduced reliance on spurious proxies; helps justify inclusion of adaptive engineered features (they should exhibit non-trivial SHAP magnitude).  
Example (XGBoost multiclass – class 1 probability):  
```python
import shap
explainer = shap.TreeExplainer(xgb_enhanced)
shap_values = explainer.enhance(X_test_selected)  # recent SHAP versions use explainer(X)
# For classic API:
# shap_values = explainer.shap_values(X_test_selected)
shap.summary_plot(shap_values[1], X_test_selected)  # class 1 (Up)
```
Interpretation Tip: High absolute SHAP value ⇒ strong impact that period; clusters reveal feature regimes (e.g., BB_Width_20 low + positive AMA contribution before Up moves).  
</details>

<details><summary><strong>LIME</strong></summary>
Purpose: Local Interpretable Model-agnostic Explanations—fits a simple surrogate (e.g., sparse linear model) around a single instance by perturbing inputs to approximate local decision boundary.  
How/When/Where: Use sparingly on representative or edge-case test samples (misclassifications, high-confidence predictions) after model finalization. Avoid exhaustive use (runtime + perturbation assumptions).  
Why: Provides intuitive per‑observation explanation when stakeholders require a human-readable linear approximation rather than full SHAP distribution.  
Relevance: Validates that GA feature pruning did not create brittle decision regions: if surrogate weights are dominated by a few logical indicators (e.g., DI_Diff_14, AMA_GA_Optimized), sparsity aligns with project goal.  
Example:  
```python
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train_selected.values,
                                 feature_names=selected_features,
                                 class_names=['Down','Up','Sideways'],
                                 discretize_continuous=True)
exp = explainer.explain_instance(X_test_selected.iloc[i].values,
                                 xgb_enhanced.predict_proba,
                                 num_features=8)
exp.show_in_notebook()
```
Caution: Perturbations may break time-series coherence (feature correlations); use for qualitative insight only.  
</details>

<details><summary><strong>partial dependence plot (PDP)</strong></summary>
Purpose: Shows marginal effect of one (or two) features on predicted outcome by averaging model predictions over the joint empirical distribution of all other features.  
How/When/Where: After model training (baseline vs enhanced) on test or validation subset (never training only). Use sklearn.inspection.partial_dependence / PartialDependenceDisplay on selected continuous features (e.g., AMA_GA_Optimized, BB_Width_20).  
Why: Highlights monotonicity, saturation, or threshold behavior; helps detect whether adaptive engineered features introduce clearer, smoother response curves than redundant clusters removed by GA.  
Relevance: If PDP for AMA_GA_Optimized shows stable increasing probability of Up within certain efficiency band while SMA variants were noisy earlier, it supports feature consolidation success.  
Example:  
```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(rf_enhanced, X_test_selected,
                                        ['BB_Width_20','AMA_GA_Optimized'])
``` 
Limitations: Averages can mislead under strong feature interactions—consider ALE as complement.  
</details>

<details><summary><strong>accumulated local effects (ALE)</strong></summary>
Purpose: Alternative to PDP that accumulates local finite differences; less biased by correlated features and preserves locality without extrapolating into sparse regions.  
How/When/Where: Post-training interpretability step when feature correlation (e.g., SMA_Dist_20 vs EMA_Dist_20) could distort PDP. Libraries: alibi, pyALE, scikit-learn-contrib (third-party).  
Why: Produces more faithful marginal effect estimates under multicollinearity—important in engineered technical indicator sets.  
Relevance: Helps validate that retained GA subset features (e.g., DI_Diff_14, FDI_GA_Optimized) have stable localized influence patterns, not artifacts of averaging across unrealistic combinations.  
Example (pyALE):  
```python
from pyALE import ale
ale_res = ale(X=X_test_selected, 
              model=rf_enhanced.predict_proba,
              feature=['FDI_GA_Optimized'], 
              grid_size=40)
```
Interpretation: ALE curve slope indicates local effect direction; flat regions imply diminishing marginal influence.  
</details>

<details><summary><strong>permutation importance</strong></summary>
Purpose: Model-agnostic feature importance: measures performance degradation (e.g., weighted F1) when a single feature’s values are randomly permuted.  
How/When/Where: After final model fit on test set (freeze random seeds); use sklearn.inspection.permutation_importance. Compare baseline vs GA-enhanced difference.  
Why: Captures actual performance dependency (including interactions) unlike raw impurity-based importances alone; robust confirmation that pruned features were indeed low-impact.  
Relevance: Should reveal concentration of importance in a smaller, higher-signal set after GA (feature reduction percentage) and confirm added engineered features contribute non-zero drop.  
Example:  
```python
from sklearn.inspection import permutation_importance
r = permutation_importance(xgb_enhanced, X_test_selected, y_test, 
                           n_repeats=20, scoring='f1_weighted', random_state=42)
imp_df = (pd.DataFrame({'feature': X_test_selected.columns, 'drop': r.importances_mean})
          .sort_values('drop', ascending=False))
``` 
Tip: Aggregate multiple runs if variance high in noisy markets.  
</details>

<details><summary><strong>feature attribution</strong></summary>
Purpose: General term for assigning quantitative contribution scores (local or global) to input features regarding predictions (e.g., SHAP values, permutation drops, LIME weights).  
How/When/Where: Applied post-training for transparency, debugging misclassifications, validating GA subset parsimony, and documenting run_* results (could store attribution summary in run_summary.json).  
Why: Ensures reduced subset still offers explainable rationale; assists in auditing shifts across runs (drift in importance).  
Relevance: Provides evidence that GA’s dynamic strategies + natural disasters did not retain spurious or unstable features; supports governance & reproducibility claims.  
Example: Combine multiple attributions: top 10 by mean |SHAP| vs top 10 by permutation drop—intersect set should include AMA/FDI if they add value.  
</details>

<details><summary><strong>counterfactual explanations</strong></summary>
Purpose: Show minimal feature changes needed to alter a model’s prediction (e.g., from Sideways to Up).  
How/When/Where: Optional advanced interpretability step performed on enhanced model predictions for borderline cases; libraries: alibi, dice-ml. Generate using only actionable or interpretable technical indicators (avoid unrealistic shifts such as retroactively changing past volatility).  
Why: Helps understand decision boundaries; identifies which retained features exert decisive influence near class thresholds.  
Relevance: Confirms GA subset expresses meaningful, controllable drivers (e.g., small shift in DI_Diff_14 + contraction in BB_Width_20 tipping classification). Also surfaces if model over-relies on a single engineered feature (potential risk).  
Example (conceptual using dice-ml):  
```python
import dice_ml
# data_interface & model already wrapped
exp = dice_ml.Dice(model, data_interface, method='random')
cf = exp.generate_counterfactuals(query_instance, total_CFs=2, desired_class=1)
```
Caution: In time series, counterfactual feasibility limited—use for interpretive insight not trade simulation.  
</details>

<details><summary><strong>model-agnostic vs model-specific explanations</strong></summary>
Purpose: Distinction between interpretability tools usable across any estimator (model-agnostic: LIME, permutation importance, PDP, ALE, counterfactuals) vs those leveraging internal structure (model-specific: tree-based SHAP optimizations, gain/cover importances, path-dependent explanations).  
How/When/Where: Select approach based on (a) need to compare RF vs XGBoost uniformly (use agnostic), (b) efficiency and precision (use model-specific SHAP for trees). Document choice in run README or extended interpretability appendix.  
Why: Balances fidelity and computational cost; ensures fair baseline vs enhanced model comparison.  
Relevance: Project uses both: permutation importance (agnostic) to validate cross-model robustness; TreeExplainer SHAP (specific) for efficient granular attributions on XGBoost / RF.  
Example: If a feature ranks high in model-specific importance but low in agnostic permutation drop, investigate potential leakage or correlation masking.  
</details>

<details><summary><strong>interpretability vs explainability</strong></summary>
Purpose: Interpretability: inherent transparency of model mechanics (e.g., linear coefficients, small decision trees). Explainability: post-hoc techniques clarifying opaque model behavior (e.g., SHAP on XGBoost).  
How/When/Where: After adopting ensemble models (less interpretable), apply explainability tools to recover stakeholder insight; record summaries in run_* artifacts (e.g., top SHAP contributors).  
Why: Aligns predictive performance gains (via GA, adaptive indicators, boosting) with auditability and trust—critical for financial model governance.  
Relevance: GA increases sparsity (improved interpretability indirectly by shrinking feature space) while explainability tools provide local/global rationale—dual approach satisfies compliance and model risk management.  
Example: Feature count drop (120 → 38) + SHAP summary showing diversified yet concise contributor set = improved interpretability; per-trade SHAP waterfall = explainability for single prediction.  
</details>

## Metaheuristics & Genetic Algorithm (GA) Concepts

<details><summary><strong>Meta</strong></summary>
Purpose: Prefix meaning “about itself” or “at a higher (strategic) level.”  
How/When/Where: Appears in project naming (Metaheuristics) to signal algorithms that search over other algorithm configurations or solution structures (feature subsets, indicator parameters).  
Why: Distinguishes higher-level search logic (GA) from base learners (RandomForest, XGBoost).  
Relevance: Clarifies that the project optimizes (selects/engineers) features rather than hard‑coding them.  
Example: “Meta layer” = GA evolving binary masks; “base layer” = model trained on selected features.
</details>

<details><summary><strong>Heuristic</strong></summary>
Purpose: Rule-of-thumb or approximate method providing “good enough” solutions faster than exhaustive search.  
How/When/Where: Embedded in GA operators (selection, crossover, mutation) and feature engineering parameter bounds.  
Why: Full enumeration of 2^F feature subsets is infeasible; heuristics guide exploration.  
Relevance: Enables timely convergence on sparse, stable feature sets needed for iterative financial experimentation.  
Example: Using tournament selection instead of evaluating all pairwise parent combinations.
</details>

<details><summary><strong>Metaheuristic</strong></summary>
Purpose: Higher-level strategy (e.g., GA, PSO, ACO, Simulated Annealing) orchestrating heuristic moves to explore complex search spaces.  
How/When/Where: GA used for (1) feature subset selection, (2) adaptive indicator parameter optimization (AMA, FDI).  
Why: Provides balance between exploration (diversity, mutation) and exploitation (elitism, selection).  
Relevance: Central to reducing feature dimensionality while improving cross‑validation stability in volatile financial time series.  
Example: GA + natural disaster mechanism = metaheuristic adaptation to stagnation.
</details>

<details><summary><strong>Genetic Algorithm</strong></summary>
Purpose: Population-based evolutionary metaheuristic inspired by natural selection.  
How: Iterative cycle: initialize_population → evaluate_population (fitness_function) → selection → crossover → mutation → elitism → diversity control → next generation.  
When/Where: Implemented in MH_Feature_Selection.GeneticAlgorithm and indirectly echoed in feature engineering module.  
Why: Efficiently searches combinatorial (binary mask) and constrained parameter spaces.  
Relevance: Produces smaller, more stable feature subsets increasing model generalization and reducing runtime in downstream tuning.  
Example Pseudocode: population = init(); for g in range(G): scores = eval(pop); pop = evolve(pop,scores).
</details>

<details><summary><strong>population</strong></summary>
Purpose: Set/list of candidate solutions (feature masks or parameter vectors) maintained each generation.  
How/When: Initialized via initialize_population(); updated in evolve().  
Where: Stored as list of numpy arrays (binary masks) inside GeneticAlgorithm instance.  
Why: Enables parallel evaluation (joblib) and diversity-driven search.  
Relevance: Larger population increases coverage of correlated financial indicator space; governed by dynamic rates and disaster resets.  
Example: pop_size=50 → 50 binary vectors length = number of features.
</details>

<details><summary><strong>individual</strong></summary>
Purpose: Single candidate solution (binary mask of features).  
How/When: Created during initialization, recombined in crossover(), altered in mutate().  
Where: Numpy array of 0/1 aligned to feature_names order (feature_names.pkl).  
Why: Encodes inclusion/exclusion decision succinctly for evaluation.  
Relevance: Determines which indicators feed model during CV, directly influencing fitness.  
Example: [1,0,0,1,1,...] selects features at indices with 1s.
</details>

<details><summary><strong>initialize_population()</strong></summary>
Purpose: Seed first generation with guided (importance-ranked) and random individuals.  
How/When: Called once in run() before generational loop.  
Where: MH_Feature_Selection.GeneticAlgorithm.initialize_population.  
Why: Blends exploitation (top-ranked features) and exploration (random masks) to accelerate convergence.  
Relevance: Reduces time to a competitive subset on high-dimensional engineered feature matrix.  
Example: 25% guided using preliminary feature importance; rest random within min/max feature constraints.
</details>

<details><summary><strong>fitness_function()</strong></summary>
Purpose: Quantifies quality of an individual (subset) via time-series CV.  
How: Trains model on selected features, collects cross_val_score accuracy, computes cv_stability (1 − std), subtracts complexity_penalty.  
When: During evaluate_population() each generation.  
Where: MH_Feature_Selection.GeneticAlgorithm.fitness_function.  
Why: Encourages both predictive accuracy and temporal robustness while discouraging oversized subsets.  
Relevance: Drives emergence of sparse, stable features appropriate for non-stationary regimes.  
Example: fitness = mean_acc + (1-std_acc) - λ*(num_features/total).
</details>

<details><summary><strong>complexity_penalty</strong></summary>
Purpose: Penalizes large feature subsets to promote sparsity.  
How/When: Added inside fitness_function() as subtractive term proportional to relative feature count.  
Why: Prevents trivial “select all” which may inflate in-sample accuracy but reduce stability/out-of-sample generalization.  
Relevance: Supports project objective: higher feature reduction percentage without performance loss.  
Example: penalty = 0.1 * (k / max_features).
</details>

<details><summary><strong>evaluate_population</strong></summary>
Purpose: Batch computation of fitness for all individuals.  
How/When: Uses joblib.Parallel to call fitness_function() per mask each generation.  
Where: GeneticAlgorithm.evaluate_population().  
Why: Parallelism cuts wall-clock time for expensive CV scoring.  
Relevance: Makes GA feasible on dozens of generations with 50+ candidates and 5-fold TimeSeriesSplit.  
Example: fitness_scores = Parallel(...)(delayed(fitness_function)(ind) for ind in population).
</details>

<details><summary><strong>diversity</strong></summary>
Purpose: Measure of heterogeneity among individuals (often mean pairwise Hamming distance).  
How/When: Calculated each generation (calculate_population_diversity).  
Why: Low diversity risks premature convergence.  
Relevance: Financial indicators are correlated; maintaining diversity avoids local optima dominated by one indicator cluster (e.g., multiple MA distances).  
Example: diversity ≈ 0.25 means average 25% differing bits across population masks.
</details>

<details><summary><strong>inject_diversity()</strong></summary>
Purpose: Actively introduce new random individuals when diversity below threshold.  
How/When: Triggered post evaluation if diversity < diversity_threshold.  
Why: Re-seeds population to escape stagnation.  
Relevance: Complements natural disaster; mitigates overfitting to early volatile fold artifacts.  
Example: Replace lowest-fitness fraction with fresh random masks honoring min/max features.
</details>

<details><summary><strong>dynamic rates</strong></summary>
Purpose: Generation-dependent adaptation of crossover_rate and mutation_rate (ILM/DHC or DHM/ILC strategies).  
How/When: calculate_dynamic_rates(generation) updates rates as progress ratio increases.  
Why: High crossover early (recombination) then rising mutation later (fine search) or inverse for large populations.  
Relevance: Tailors exploration/exploitation schedule to population size, improving convergence speed on feature selection.  
Example: Small pop: mutation grows 0→high; crossover decays high→lower.
</details>

<details><summary><strong>crossover rate</strong></summary>
Purpose: Probability of performing crossover between selected parent pairs (or proportion of offspring created by crossover).  
How/When: Adjusted per generation by dynamic rates; controls recombination intensity.  
Why: Too high late can disrupt good building blocks; too low early slows mixing.  
Relevance: Financial feature blocks (trend, volatility, volume) benefit from early recombination to find synergistic subsets.  
Example: Start 0.8 → decay toward 0.4 by final generations.
</details>

<details><summary><strong>mutation rate</strong></summary>
Purpose: Probability of flipping a bit (feature inclusion) per locus.  
How/When: Adjusts with dynamic strategy (increasing or decreasing).  
Why: Maintains exploratory pressure; avoids getting trapped near local optima.  
Relevance: Allows occasional introduction of underrepresented indicator classes (e.g., fractal/volatility) late in search.  
Example: 0.05 early → 0.25 late (ILM pattern).
</details>

<details><summary><strong>crossover</strong></summary>
Purpose: Combine two parents to produce offspring (feature masks).  
How: Uniform crossover chooses each bit from parent1 or parent2 with 0.5 probability.  
When: During evolve(), after parent selection.  
Why: Reassembles complementary subsets (e.g., momentum bits from one, volatility bits from another).  
Relevance: Accelerates discovery of high-fitness multi-category feature mixes.  
Example: Parent A: 110010, Parent B: 101100 → Child: 111100.
</details>

<details><summary><strong>mutation</strong></summary>
Purpose: Randomly flip bits to introduce novel features or remove redundant ones.  
How/When: After crossover on each offspring; rate dynamic.  
Why: Injects variability not achievable by recombination alone (exploration).  
Relevance: Essential for discovering niche indicators (e.g., FDI or Chaikin variants) overlooked in guided seeds.  
Example: Mask 101001 with mutation rate 0.2 → 111001 (bit 2 flips).
</details>

<details><summary><strong>elitism</strong></summary>
Purpose: Preservation of top-performing individuals across generations.  
How/When: elitism() copies best k (elite_size) masks into new population before filling remainder.  
Why: Prevents loss of current best subset due to stochastic operators.  
Relevance: Stabilizes fitness trajectory and supports reproducibility of best feature list.  
Example: elite_size=5 always retained.
</details>

<details><summary><strong>natural disaster</strong></summary>
Purpose: Radical diversity reset mechanism when stagnation detected (no fitness improvement for calm_before_storm generations).  
How/When: natural_disaster() stratifies population by fitness thirds and retains tiered fractions (top>mid>bottom) then refills with random individuals.  
Why: Escapes deep local minima where minor mutations fail.  
Relevance: Market feature landscape has correlated “plateaus”; disaster jump-starts search beyond them.  
Example: Keep 50% of top third, 30% of middle third, 10% of bottom third, rest random.
</details>

<details><summary><strong>evolve</strong></summary>
Purpose: Execute one generation cycle: rate update → evaluate → diversity checks → selection → reproduction → elitism.  
How/When: evolve(generation) called inside run() loop.  
Why: Encapsulates evolutionary step for clarity and modular instrumentation (logging diversity, feature counts).  
Relevance: Enables adding diagnostics (feature count evolution, diversity history) for financial audit.  
Example: for g in range(max_generations): evolve(g).
</details>

<details><summary><strong>generation</strong></summary>
Purpose: Iteration index of evolutionary loop.  
How/When: 0 … max_generations-1.  
Why: Drives dynamic rate schedules and stagnation counters.  
Relevance: Provides timeline for convergence plots (fitness vs generation) saved per run directory.  
Example: Convergence flattening after generation ~18 suggests diminishing returns.
</details>

<details><summary><strong>parent</strong></summary>
Purpose: Selected individual used to produce offspring via crossover/mutation.  
How/When: Chosen by tournament selection (or other selection policy) every generation.  
Why: Bias reproduction toward fitter subsets while allowing chance presence of weaker (diversity).  
Relevance: Ensures high-signal feature groupings (e.g., ADX + BB_Width + AMA) propagate.  
Example: Parent chosen if wins tournament of size 3.
</details>

<details><summary><strong>diversity_threshold</strong></summary>
Purpose: Minimum acceptable population diversity before triggering injection or disaster logic.  
How/When: Checked each generation post evaluation.  
Why: Guardrail preventing early convergence.  
Relevance: Too low → risk redundant MA distance dominance; tuned to keep volatility, volume, adaptive features in play.  
Example: threshold=0.15 (Hamming) → if actual <0.15, reintroduce random masks.
</details>

<details><summary><strong>inject_diversity() (alias)</strong></summary>
Purpose: Same as earlier entry; repeated in notes; ensure deduplication in documentation.  
Relevance: Avoids misinterpretation when scanning variable/function names.
</details>

<details><summary><strong>calm_before_storm</strong></summary>
Purpose: Stagnation counter threshold (generations without fitness gain) before triggering natural_disaster.  
How/When: Incremented when best_fitness unchanged; reset on improvement.  
Why: Adaptive restart timing rather than fixed schedule.  
Relevance: Helps respond to plateaus in financial indicator search; reduces wasted generations.  
Example: calm_before_storm=8 → disaster on 8th stagnant generation. (Typo “clam_before_storm” corrected in docs.)
</details>

<details><summary><strong>natural disaster mechanism (selective retention + random fill)</strong></summary>
Purpose: Structured partial reset preserving stratified excellence while re-injecting exploration.  
How: Retain tiered slices (top/mid/bottom) then random-generate remaining population.  
Why: Maintains learned building blocks (e.g., AMA + volatility pair) while freeing capacity for novel combinations.  
Relevance: Balances exploitation and radical exploration crucial in correlated feature spaces.  
Example: After stagnation: new_population = elite fractions + random_new.
</details>

<details><summary><strong>tournament selection</strong></summary>
Purpose: Parent selection method: sample k individuals, keep fittest.  
How/When: Each parent pick during selection phase.  
Why: Adjustable selection pressure via tournament_size; simpler than proportional methods.  
Relevance: Stable under noisy fitness variance of financial CV; prevents dominance by outlier fitness spikes.  
Example: k=3 gives moderate pressure; probability weaker survives ≈ (1/3)^rank.
</details>

<details><summary><strong>uniform crossover</strong></summary>
Purpose: Recombination scheme selecting each bit from either parent with equal probability.  
How/When: Applied in crossover() to produce two children.  
Why: Maximizes mixing granularity (vs 1‑point) in unordered feature masks.  
Relevance: Financial indicators have no inherent ordering; uniform preserves neutrality.  
Example: mask = rand(len) < 0.5 → child = mask?parent1:parent2.
</details>

<details><summary><strong>bit-flip mutation</strong></summary>
Purpose: Mutation operator toggling individual bits with probability = mutation_rate.  
How/When: After crossover per offspring.  
Why: Simple, unbiased exploration of inclusion space.  
Relevance: Efficiently activates rarely selected features (e.g., fractal dimension) for fitness evaluation.  
Example: if rand() < 0.2: bit ^= 1.
</details>

<details><summary><strong>particle swarm optimisation (PSO)</strong></summary>
Purpose: Population-based metaheuristic where “particles” move in continuous space guided by personal/global bests.  
How/When (not implemented): Potential alternative for continuous indicator parameter tuning (e.g., AMA parameters).  
Why: Fast convergence in smooth continuous spaces; less suited to sparse discrete masks.  
Relevance: Documented extension pathway for feature engineering parameter search beyond GA.  
Example Concept: velocity ← w*velocity + c1*r1*(pbest - pos) + c2*r2*(gbest - pos).
</details>

<details><summary><strong>ant colony optimisation (ACO)</strong></summary>
Purpose: Probabilistic constructive metaheuristic using pheromone trails to bias future solution construction.  
How/When (future): Could build feature subsets sequentially (feature inclusion decisions) influenced by pheromone = historical fitness contribution.  
Why: Explores combinatorial spaces with adaptive memory.  
Relevance: Alternative benchmark to confirm GA robustness in selecting stable financial indicators.  
Example Concept: P(feature i chosen) ∝ pheromone_i^α * heuristic_i^β.
</details>

<details><summary><strong>simulated annealing</strong></summary>
Purpose: Single-solution metaheuristic accepting worse moves with temperature-controlled probability to escape local minima.  
How/When (future): Could refine a GA-derived subset (post-processing).  
Why: Lightweight exploitation after broad GA exploration.  
Relevance: Hybrid pipeline (GA global + SA local) may yield marginal incremental stability gains.  
Example Acceptance: if Δfitness < 0 accept with prob exp(Δ/T), T → cool_rate*T.
</details>

## GA Parameters & Patterns (project-specific)

<details><summary><strong>pop_size</strong></summary>
Purpose: Number of individuals (feature masks) maintained per GA generation.  
How/When/Where: Set in GeneticAlgorithm(..., pop_size=50) during initialization (Combined workflow Section 5). Used in population initialization, evaluation loops, and strategies selecting dynamic rate schedule (≤100 = small, ≥200 = large).  
Why: Balances exploration (diversity coverage of feature space) vs runtime (CV cost scales ~ pop_size × cv_folds).  
Relevance (project): Too small risks premature convergence on correlated indicators (e.g., multiple MA distances); too large inflates compute given 5-fold TimeSeriesSplit. Default 50 exploits dynamic mutation ramp (ILM/DHC) efficiently.  
Example:
```python
ga = GeneticAlgorithm(X_train, y_train, feature_names, pop_size=50)
```
Tip: Increase only if feature count grows substantially (e.g., >300 engineered columns).
</details>

<details><summary><strong>crossover_rate</strong></summary>
Purpose: Base probability (or proportion) governing how much offspring derive from recombining parent masks.  
How/When/Where: Passed at GA init; dynamically adjusted by strategy (ILM/DHC or DHM/ILC) inside calculate_dynamic_rates().  
Why: High early crossover mixes indicator “building blocks” (trend + volatility + volume) accelerating convergence toward synergistic subsets.  
Relevance: In small-pop strategy, starts high then tapers, preventing disruption of refined later‑generation masks while mutation increases to fine tune sparsity.  
Example: crossover_rate=0.8 → ~80% of new offspring initially produced via uniform crossover.
</details>

<details><summary><strong>mutation_rate</strong></summary>
Purpose: Probability of flipping each bit (feature inclusion) in an offspring mask.  
How/When/Where: Base value set at init (e.g., 0.2) then dynamically evolved (increasing or decreasing) per generation. Applied in mutate() after crossover.  
Why: Injects novel features or removes redundant ones; guards against local optima in correlated technical indicator space.  
Relevance: ILM phase starts lower to preserve early assembling, then ramps to explore late sparse refinements (dropping extra MA/volatility duplicates).  
Example: Late generation mutation_rate may rise from 0.2 → ~0.35 producing incremental swaps among remaining mid-importance indicators.
</details>

<details><summary><strong>max_generations</strong></summary>
Purpose: Upper bound on evolutionary iterations.  
How/When/Where: Loop in run(); each generation executes evaluation → selection → reproduction → (possible) disaster.  
Why: Controls runtime; marginal gains often plateau before maximum if stagnation triggers a natural disaster earlier.  
Relevance: 30 generations with pop_size=50 and 5-fold CV strikes balance between stability discovery (cv_stability) and notebook wall-clock constraints.  
Example:
```python
ga = GeneticAlgorithm(..., max_generations=30)
```
Guideline: Increase only if fitness curve still trending upward near final generation.
</details>

<details><summary><strong>min_features</strong></summary>
Purpose: Lower constraint on number of selected features in any individual.  
How/When/Where: Enforced in _ensure_constraints() and fitness_function() (invalid if count below).  
Why: Prevents overly sparse masks that omit essential regime descriptors (trend, volatility, volume) causing unstable CV variance.  
Relevance: Set (e.g., 15) higher than raw minimum (5) when feature space large to maintain baseline diversity during early search.  
Example: With 140 total features, min_features=15 avoids degenerate masks with only momentum indicators.
</details>

<details><summary><strong>max_features</strong></summary>
Purpose: Upper constraint on features per individual.  
How/When/Where: Checked in fitness_function(); enforced post mutation/crossover.  
Why: Penalizes bloated subsets that inflate variance and reduce interpretability (complexity_penalty).  
Relevance: Confines search to sparse regimes (e.g., max_features=50) encouraging feature reduction percentage metrics (e.g., 65% pruning).  
Example: If mask selects 72 features but max_features=50, excess bits are randomly turned off until within limit.
</details>

<details><summary><strong>cv_folds</strong></summary>
Purpose: Number of TimeSeriesSplit folds for fitness evaluation.  
How/When/Where: Passed to TimeSeriesSplit in __init__ and used inside fitness_function() to get per‑fold accuracy scores.  
Why: More folds improve robustness of cv_stability signal; too many raise runtime.  
Relevance: 5 folds adequate for hourly 6‑month window while capturing drift segments (early, mid, late).  
Example:
```python
cv = TimeSeriesSplit(n_splits=cv_folds)
```
Tip: Reduce to 3 only for rapid prototyping; stability metric becomes noisier.
</details>

<details><summary><strong>adaptive_rates</strong></summary>
Purpose: Optional toggle for diversity-responsive adjustments layered atop dynamic strategy (OFF by default in combined workflow).  
How/When/Where: If True, update_adaptive_rates() further tweaks mutation/crossover based on measured diversity vs threshold.  
Why: Fine-grained reactive control preventing premature convergence when dynamic schedule alone insufficient.  
Relevance: Disabled to simplify reproducibility; dynamic ILM/DHC already supplies staged exploration. Could be enabled if diversity_history shows early collapse.  
Example: Set adaptive_rates=True when adding many new engineered indicators (e.g., seasonal features).
</details>

<details><summary><strong>diversity_threshold</strong></summary>
Purpose: Minimum acceptable average pairwise Hamming diversity before injection logic triggers.  
How/When/Where: Calculated each generation (calculate_population_diversity); compared in inject_diversity().  
Why: Low diversity (masks nearly identical) increases stagnation and overfitting risk to transient temporal pattern.  
Relevance: Threshold often relaxed (e.g., 0.10–0.15) in financial feature spaces with inherently correlated indicators; too high causes excessive random reseeding.  
Example: diversity ≈0.08 < 0.10 → replace a fraction of lowest-fitness individuals with fresh random masks.
</details>

<details><summary><strong>ILM / DHC (Increasing Low Mutation / Dynamic High Crossover)</strong></summary>
Purpose: Small-population dynamic rate schedule: start with high crossover, low mutation; gradually shift to higher mutation as generations progress.  
How/When/Where: Selected automatically when use_dynamic_rates=True and pop_size ≤ 100. Implemented in calculate_dynamic_rates().  
Why: Early recombination assembles broad indicator mixes; later rising mutation prunes and fine-tunes sparsity.  
Relevance: Matches project’s moderate pop_size=50, accelerating convergence to stable sparse subsets while retaining late exploration for niche signals (e.g., adaptive AMA + single volatility estimator).  
Example (concept): generation_progress=0.7 ⇒ crossover_rate decayed (e.g., 0.55), mutation_rate elevated (e.g., 0.30).
</details>

<details><summary><strong>DHM / ILC (Dynamic High Mutation / Increasing Low Crossover)</strong></summary>
Purpose: Large-population alternative: begin with higher mutation and lower crossover, then gradually increase crossover later.  
How/When/Where: Chosen when pop_size ≥ 200 enabling broad random exploration early.  
Why: Large populations already cover recombination space; early mutation diversifies quickly; later increased crossover exploits discovered building blocks.  
Relevance: Not default here (pop_size=50) but documented for scalability if expanding feature set (e.g., multi-asset, lag stacks).  
Example: Early generation mutation_rate ≈0.35, crossover_rate ≈0.4; late generation mutation_rate declines, crossover_rate rises to recombine refined partial masks.
</details>

<details><summary><strong>population strategies for small vs large populations</strong></summary>
Purpose: Conditional selection of dynamic rate pattern (ILM/DHC for small; DHM/ILC for large).  
How/When/Where: Evaluated at GA init to set self.dynamic_strategy; influences per-generation rates.  
Why: Tailors exploration/exploitation balance to coverage capacity: small populations need recombination early; large ones need mutation to diversify before recombining.  
Relevance: Ensures efficiency: avoids overusing mutation in small pop (destroys good masks) or excessive crossover in large pop (wastes already diverse pool).  
Example:
```python
if pop_size <= 100: strategy='ILM_DHC'
elif pop_size >= 200: strategy='DHM_ILC'
```
</details>

<details><summary><strong>natural disaster retention ratios (top/mid/bottom thirds)</strong></summary>
Purpose: Structured partial reset after prolonged stagnation (stagnation_count ≥ calm_before_storm).  
How/When/Where: natural_disaster(): sort fitness → split into thirds → retain fractional slices (e.g., 50% top, 30% middle, 10% bottom) → refill remainder randomly.  
Why: Retains elite building blocks while re-injecting exploration to escape local plateau. Bottom slice small retention preserves occasional “innovative” outlier genes.  
Relevance: Captures abrupt regime shifts (e.g., volatility regime change) by rapidly diversifying masks when cv_stability no longer improves. Supports resilience of selected subsets across temporal change.  
Example:
```python
# After stagnation
new_pop = top[:int(.5*len(top))] + mid[:int(.3*len(mid))] + bot[:int(.1*len(bot))]
while len(new_pop) < pop_size:
    new_pop.append(random_mask())
```
</details>

## Feature Selection Outputs & Comparison Concepts

<details><summary><strong>feature importance</strong></summary>
Purpose: Quantitative estimate of how much each feature contributes to model predictive performance.
How/When/Where: Computed after model training (baseline and enhanced) using tree-based impurity importances (rf_enhanced.feature_importances_), XGBoost gain/weight/cover, or model-agnostic permutation importance (sklearn.inspection.permutation_importance) on the held‑out test set. Can also be inspected pre/post GA to validate that high-signal indicators are retained.
Why: Validates that GA-selected (and GA‑engineered AMA_GA_Optimized / FDI_GA_Optimized) features carry substantive signal rather than random noise; supports interpretability and audit.
Relevance (project): Confirms success of feature selection—importance mass should concentrate on a smaller subset after GA. Helps document which indicator categories (volatility, momentum, adaptive) drive improvements in cv_stability and weighted F1.
Example:
```python
rf_importance = pd.DataFrame({
    'feature': enhanced_features['rf'],
    'importance': rf_enhanced.feature_importances_
}).sort_values('importance', ascending=False)
print(rf_importance.head(10))
```
Tip: Pair with permutation drop to mitigate bias from correlated indicators (e.g., multiple moving average distances).
</details>

<details><summary><strong>importance ranking</strong></summary>
Purpose: Ordered list of features sorted by an importance metric (gain, impurity, permutation drop, SHAP magnitude).
How/When/Where: Generated immediately after computing feature importance; used in documentation, guided GA population seeding (initialize_population may use preliminary ranking if implemented), or post‑hoc selection analysis.
Why: Provides priority ordering for interpretability, potential manual pruning, or targeted ablation tests.
Relevance: Helps show consolidation—many mid/low-ranked redundant moving average / volatility variants should disappear in GA-enhanced models while adaptive or normalized features rise in relative position.
Example:
```python
ranking = rf_importance.sort_values('importance', ascending=False)['feature'].tolist()
top5 = ranking[:5]
```
Use: Can seed future heuristic initial individuals by turning on top-k bits and randomly sampling the rest (accelerates convergence).
</details>

<details><summary><strong>baseline vs GA-enhanced models</strong></summary>
Purpose: Comparative frame between models trained on full engineered feature set (baseline) and those trained on GA-selected + GA-engineered (AMA, FDI) features (enhanced).
How/When/Where: Baseline models trained before GA step; enhanced models trained after (a) GA feature selection and (b) addition of optimized AMA/FDI; comparison reported in performance_comparison.csv and improvements_summary.csv.
Why: Isolates incremental value of metaheuristic feature pruning + adaptive feature engineering from raw modeling capacity.
Relevance: Central evidence that the metaheuristic pipeline improves weighted F1 / accuracy while reducing dimensionality and potentially model complexity (e.g., shallower optimal max_depth).
Example (interpretation):
Baseline RF: F1 = 0.692 with 140 features  
Enhanced RF: F1 = 0.709 with 42 features (−70% features, +0.017 F1)
</details>

<details><summary><strong>feature reduction percentage</strong></summary>
Purpose: Quantifies sparsity improvement: (1 − selected_count / total_original_features) * 100.
How/When/Where: Calculated after GA run using length of selected feature list (selected_features_*.pkl) vs total feature_names; logged in run_summary.json and final summary printout.
Why: Measures efficiency gain (less overfitting risk, faster training, improved interpretability).
Relevance: A headline metric demonstrating GA success—high reduction with neutral or improved F1 indicates better bias‑variance tradeoff and stable cv_stability.
Example:
```python
total = len(feature_names)
selected = len(selected_features_rf[2])
reduction_pct = (1 - selected / total) * 100
print(f"RF reduction: {reduction_pct:.1f}%")
```
Target: Meaningful uplift typically when >40–50% reduction without metric degradation.
</details>

<details><summary><strong>selected_features_*.pkl</strong></summary>
Purpose: Serialized artifact capturing GA outcome per model type; usually a tuple (best_mask, best_fitness, selected_feature_names).
How/When/Where: Written immediately after ga.run(); stored in run_YYYYMMDD_HHMMSS/ (e.g., selected_features_rf.pkl, selected_features_xgb.pkl).
Why: Decouples selection from model hyperparameter tuning—enables retraining or alternative model comparisons without rerunning GA (reproducibility & audit).
Relevance: Canonical reference aligning feature ordering with feature_names.pkl so subsequent enhanced model training stays consistent; essential for backtracking exact subset that produced reported metrics.
Example:
```python
with open('run_xxx/selected_features_rf.pkl','rb') as f:
    mask, fitness, names = pickle.load(f)
print(len(names), fitness)
```
Tip: Always archive alongside run_summary.json to trace GA parameters (mutation_rate, strategy).
</details>

<details><summary><strong>pareto front (performance vs complexity)</strong></summary>
Purpose: Set of non-dominated solutions where improving one objective (e.g., accuracy/F1) would worsen another (complexity: feature count).
How/When/Where: (Optional) Derived by logging intermediate GA individuals’ (fitness_metric, feature_count) pairs and filtering points not strictly dominated; can be visualized post-run.
Why: Illustrates trade-offs—users may prefer slightly lower F1 with substantially fewer features (operational simplicity).
Relevance: Validates that chosen final subset lies near the efficient frontier rather than being dominated by a leaner equally accurate alternative; supports decision rationale in run README.
Example (concept extraction):
```python
points = [(fit[i], counts[i]) for i in range(len(fit))]
pareto = [p for p in points if not any((q[0] >= p[0] and q[1] <= p[1] and q != p) for q in points)]
```
Interpretation: Plot F1 (y) vs feature count (x); frontier bends downward—choose elbow balancing marginal gains vs added complexity.
</details>

<details><summary><strong>sparsity</strong></summary>
Purpose: Degree to which the selected feature vector contains zeros (excluded features); formally 1 − (selected_features / total_features).
How/When/Where: Emerges after GA selection; tracked via feature reduction percentage; can also be monitored generation-by-generation (num_features_history).
Why: Sparse representations reduce overfitting, training time, memory usage, and enhance interpretability.
Relevance: High sparsity with stable cv_stability indicates GA effectively pruned redundant correlated technical indicators (e.g., overlapping moving average distances) while preserving signal-rich adaptive and regime descriptors.
Example:
```python
sparsity = 1 - selected / total
print(f"Sparsity: {sparsity:.2%}")
```
Note: Excessive sparsity below min_features can harm regime coverage—balance against performance via Pareto analysis.
</details>

## Finance & Risk Metrics

<details><summary><strong>ticker</strong></summary>
Purpose: Short symbol identifying a tradable asset (e.g., EURUSD, BRENTCMDUSD, TSLA).  
How/When/Where: Used in data fetching (Polygon API) and file naming (raw_data/{TICKER}_H1.csv).  
Why: Standardized identifier enables automated retrieval, storage, and reproducibility across run_* directories.  
Relevance (project): Selecting TICKER (or DATASET override) defines the OHLCV universe from which engineered features and GA-selected subsets are derived.  
Example: TICKER='C:GBPUSD' → fetch hourly candles, then engineer indicators and run GA.
</details>

<details><summary><strong>technical indicator</strong></summary>
Purpose: Quantitative transformation of OHLCV capturing trend, momentum, volatility, volume, or structure (e.g., RSI, ATR, BB_Width).  
How/When/Where: Engineered in baseline feature block (Combined workflow Section 3); stored in engineered_features.csv.  
Why: Converts non‑stationary raw prices into more stationarity‑adjacent signals the models can exploit.  
Relevance: Defines the feature search space for GA feature selection; adaptive indicators (AMA / FDI) extend this set.  
Example: RSI_14 < 30 used (implicitly) by model as oversold context for potential Up (Target=1).
</details>

<details><summary><strong>open</strong></summary>
Purpose: First traded (or aggregated) price of the bar (hour).  
How/When: Ingested from raw CSV / API; part of OHLCV base schema.  
Why: Serves in gap, range and ratio calculations (Gap, Open_Close_Ratio).  
Relevance: Inputs multiple engineered features feeding GA masks.  
Example: Gap = Open_t − Close_{t−1}.
</details>

<details><summary><strong>high</strong></summary>
Purpose: Maximum price in the bar interval.  
Use: Range (High−Low), ATR True Range components, Bollinger, volatility estimators (Garman‑Klass, Parkinson).  
Relevance: Core for volatility & candle anatomy features selected (or pruned) by GA.  
Example: Candle_Range = High − Low.
</details>

<details><summary><strong>low</strong></summary>
Purpose: Minimum price in the bar interval.  
Use: Same contexts as High; part of wick / shadow features and volatility estimators.  
Relevance: Supports regime indicators (BB_Width, ATR_Ratio) influencing Target thresholds.  
Example: Lower_Shadow = min(Open,Close) − Low.
</details>

<details><summary><strong>close</strong></summary>
Purpose: Final (or last known) price of the bar.  
Use: Basis for returns, moving averages, momentum oscillators, price distance/z-score features.  
Relevance: Most engineered indicators reference Close; GA evaluates subsets of derived transformations.  
Example: Percentage_Change = Close.pct_change().
</details>

<details><summary><strong>volume</strong></summary>
Purpose: Number of units (or tick count proxy) traded during the bar.  
Use: Volume ratios, OBV/OBV_TickDev, CMF, activity z-scores.  
Relevance: Activity context helps discriminate Sideways vs directional classes for stability.  
Example: TickVol_Z_20 > 2 implies elevated participation.
</details>

<details><summary><strong>Sharpe Ratio</strong></summary>
Purpose: Risk‑adjusted return = (Mean(Return) − Risk‑Free)/Std(Return).  
How/When: Post-model strategy evaluation (not yet integrated); computed on hypothetical strategy equity curve.  
Why: Normalizes returns by volatility to assess efficiency.  
Relevance: Future performance lens if model outputs drive trading simulation; complements cv_stability by adding economic significance.  
Example: If mean hourly strategy return = 0.04%, std = 0.30%, risk‑free≈0 → Sharpe ≈ 0.04 / 0.30 ≈ 0.133.
</details>

<details><summary><strong>Sortino Ratio</strong></summary>
Purpose: Downside risk-adjusted return = (Mean(Return) − Risk‑Free)/Std(Negative Returns).  
Why: Penalizes harmful volatility only; distinguishes benign upside variance.  
Relevance: Future enhancement to evaluate predictive model’s directional filters in asymmetric risk contexts.  
Example: If downside std = 0.2% vs total 0.3%, Sortino > Sharpe when drawdowns limited.
</details>

<details><summary><strong>Maximum Drawdown (MaxDD)</strong></summary>
Purpose: Largest peak-to-trough decline of cumulative return.  
How/When: Derived from equity = (1+r).cumprod() after a backtest.  
Why: Captures tail risk and capital impairment magnitude.  
Relevance: Could validate if GA-selected sparse features reduce extreme adverse sequences in a deployed strategy layer.  
Example: Equity peak 1.20 → trough 0.95 ⇒ MaxDD = (0.95/1.20 − 1) = −20.8%.
</details>

<details><summary><strong>CAGR (Compound Annual Growth Rate)</strong></summary>
Purpose: Annualized geometric growth rate of equity over test horizon.  
How: ((Ending Equity / Starting)^(1/Years)) − 1.  
Why: Normalize performance vs varying test lengths.  
Relevance: When converting classification to trading signals; helps compare across different dataset spans.  
Example: Start 1.0 → End 1.35 over 0.5 years ⇒ CAGR ≈ 1.35^(1/0.5)−1 ≈ 82%.
</details>

<details><summary><strong>VaR (Value at Risk)</strong></summary>
Purpose: Quantile loss threshold (e.g., 95%) not expected to be exceeded under normal conditions over a horizon.  
How: Historical / parametric (μ−zσ) / simulation.  
Why: Regulatory & risk sizing context.  
Relevance: Future risk overlay calibrating position sizing using model probability outputs.  
Example: 95% 1-hour VaR = −0.7% means worse loss exceeds 0.7% only 5% of hours (historically).
</details>

<details><summary><strong>CVaR / Expected Shortfall</strong></summary>
Purpose: Mean loss conditional on exceeding VaR (tail severity).  
Why: Captures tail magnitude vs mere threshold.  
Relevance: Helps assess whether GA-driven features indirectly moderate tail events by improving class discrimination.  
Example: If losses beyond 95% VaR average −1.2%, CVaR_95 = −1.2%.
</details>

<details><summary><strong>profit factor</strong></summary>
Purpose: Gross Profits / Gross Losses in a backtest.  
Why: Simple efficiency ratio ( >1 favorable).  
Relevance: Future trading layer metric to judge if feature reduction improves signal quality (fewer noisy trades).  
Example: Profits 1500, Losses 1000 ⇒ Profit Factor = 1.5.
</details>

<details><summary><strong>win rate</strong></summary>
Purpose: % of trades with positive P&L.  
Why: Indicates consistency but ignores payoff magnitude.  
Relevance: Model could lower win rate while increasing average payoff—pair with profit factor.  
Example: 40 wins / 100 trades = 40%.
</details>

<details><summary><strong>average trade return</strong></summary>
Purpose: Mean per-trade percentage or monetary return.  
Why: Complements win rate (captures payoff asymmetry).  
Relevance: Evaluate whether GA-pruned features raise average edge per signal.  
Example: Sum returns 12% / 100 trades = 0.12% per trade.
</details>

<details><summary><strong>slippage</strong></summary>
Purpose: Execution price deviation from intended (due to latency/liquidity).  
How: Modeled as spread/partial fill adjustments.  
Relevance: Not modeled yet; would reduce realized performance vs predictive metrics—important for future backtest validity.  
Example: Predicted fill 1.1000, actual 1.1002 (−0.2 pip slippage).
</details>

<details><summary><strong>spread</strong></summary>
Purpose: Bid-ask difference (transaction cost component).  
Why: Affects net returns; high spread degrades low edge signals.  
Relevance: Feature subset may enable stricter filters to overcome spreads.  
Example: EURUSD spread 0.8 pips; requires strategy expectancy >0.8 pips per trade.
</details>

<details><summary><strong>bid / ask</strong></summary>
Purpose: Best available selling (bid) and buying (ask) prices.  
How: Mid=(bid+ask)/2 base for some microstructure features (future).  
Relevance: Current dataset lacks; potential expansion for spread-aware indicators (microstructure predictive signals).  
Example: Bid 1.0998 / Ask 1.1000 ⇒ Spread 0.0002 (2 pips).
</details>

<details><summary><strong>pip / lot size</strong></summary>
Purpose: Pip: minimum standardized FX price increment; lot size: trade unit quantity (e.g., 100k base units).  
Why: Convert percentage/price predictions into monetary risk & position sizing.  
Relevance: Needed for translating model class outputs into standardized trade sizing and risk metrics.  
Example: EURUSD pip = 0.0001; 50 pips gain on 1 lot ≈ $500.
</details>

<details><summary><strong>leverage</strong></summary>
Purpose: Use of borrowed capital to amplify exposure relative to equity.  
Why: Increases return & loss magnitude.  
Relevance: Risk layer must adjust leverage based on model confidence & volatility features (e.g., ATR_Ratio).  
Example: 10× leverage on 1% move = ±10% equity change.
</details>

<details><summary><strong>margin</strong></summary>
Purpose: Collateral required to open leveraged position.  
Why: Prevents default; margin calls on adverse moves.  
Relevance: Volatility features could dynamically modulate margin utilization in future strategy module.  
Example: Required margin 5% → Notional 100k needs 5k equity.
</details>

<details><summary><strong>position sizing</strong></summary>
Purpose: Determining trade quantity based on risk (e.g., % equity per ATR stop).  
Why: Core risk control affecting drawdown & Sharpe.  
Relevance: Model confidence (probability Up vs Down) + volatility (ATR) + CVaR estimates can drive adaptive sizing.  
Example: Risk 0.5% equity per trade with stop = 1.5 * ATR.
</details>

<details><summary><strong>transaction cost model</strong></summary>
Purpose: Framework approximating slippage, spread, commissions.  
Why: Adjust raw predictive edge to net performance.  
Relevance: Essential for validating that GA feature reduction doesn’t just overfit gross metrics.  
Example: Cost per trade = spread(pips)*pip_value + commission + modeled slippage.
</details>

<details><summary><strong>backtest engine</strong></summary>
Purpose: Simulator applying strategy rules to historical data (chronological, no lookahead) producing equity curve & metrics.  
Relevance: Next layer after current predictive classification; would consume model signals + thresholds derived from features.  
Example: Walk-forward evaluation aligning with TimeSeriesSplit logic.
</details>

<details><summary><strong>execution risk</strong></summary>
Purpose: Risk of adverse price movement between signal and fill.  
Why: Reduces realized vs modeled edge.  
Relevance: Latency-sensitive signals (e.g., short horizon momentum) may underperform unless accounted for in cost model.  
Example: Sudden widening of spread during news.
</details>

<details><summary><strong>market impact</strong></summary>
Purpose: Price movement caused by one’s own order flow.  
Why: Large orders erode expected edge.  
Relevance: Probably minimal in small-scale prototype; important if scaling or aggregating multi-asset models.  
Example: Impact cost modeled ~ k * (size / ADV)^α.
</details>

<details><summary><strong>financial instrument</strong></summary>
Purpose: Tradable contract (currency pair, CFD, commodity, equity).  
Why: Defines market microstructure & feature relevance (e.g., tick volume reliability).  
Relevance: Feature engineering must adapt if switching from FX to equity (true volume vs tick volume logic branch).  
Example: BRENTCMDUSD vs EURUSD difference in volume semantics.
</details>

<details><summary><strong>trading strategy</strong></summary>
Purpose: Rule set converting predictions & features into position, size, and exit actions.  
Relevance: Current project supplies predictive input component; strategy layer evaluates economic viability (Sharpe, MaxDD).  
Example: Go long when model prob(Up) − prob(Down) > threshold & ADX_14 rising.
</details>

<details><summary><strong>backtesting</strong></summary>
Purpose: Historical simulation of strategy logic to estimate performance metrics.  
Why: Validate generalization beyond classification metrics.  
Relevance: Extends cv_stability concept into risk-adjusted return evaluation.  
Example: Walk-forward application of probabilistic thresholding over test segment.
</details>

<details><summary><strong>portfolio optimisation</strong></summary>
Purpose: Allocation of capital across multiple instruments/strategies to maximize risk-adjusted return (e.g., mean‑variance, risk parity).  
Relevance: Future multi-asset extension; GA-selected sparse feature sets could feed per-asset alpha models aggregated via optimizer.  
Example: Use predicted class probabilities to derive expected return inputs μ in Markowitz optimization.
</details>

<details><summary><strong>algorithmic trading</strong></summary>
Purpose: Automated trade execution based on pre-defined logic & model outputs.  
Relevance: Downstream deployment target for the predictive model once validated with risk metrics.  
Example: Script polls latest engineered features, runs classifier, sends order if criteria met.
</details>

<details><summary><strong>high-frequency trading</strong></summary>
Purpose: Ultra-low latency strategies on sub-second horizons.  
Relevance: Out of scope (hourly bars); indicates limits of current feature granularity (cannot evaluate microsecond microstructure).  
Example: Not appropriate to extrapolate current GA performance to HFT domain.
</details>

<details><summary><strong>returns</strong></summary>
Purpose: Percentage or log change in asset value over interval; foundational performance and risk input.  
How: Percentage_Change, Log_Return already engineered.  
Relevance: Drives target labeling, volatility, Sharpe-like future metrics.  
Example: r_t = Close_t / Close_{t-1} − 1.
</details>

<details><summary><strong>Sharpe ratio (duplicate)</strong></summary>
Purpose: Duplicate entry of Sharpe Ratio; see earlier “Sharpe Ratio” definition.  
Relevance: Maintained for search completeness—avoid double counting in documentation.
</details>

<details><summary><strong>model monitoring</strong></summary>
Purpose: Continuous tracking of model predictive performance, input feature stats, and operational health after deployment.  
How/When/Where: Implemented post-release (future extension) as scheduled batch job or streaming service comparing current predictions vs later realized outcomes (once horizon elapses). Store metrics (accuracy, weighted F1, class distribution, cv_stability proxy) in a monitoring dashboard / log store.  
Why: Detect silent degradation due to regime shifts (volatility spikes, structural changes) before large performance losses accumulate.  
Relevance (project): GA-selected sparse subset + adaptive features (AMA/FDI) expected to slow degradation; monitoring validates if sparsity correlates with robustness and triggers retraining criteria.  
Example (pseudo):
```python
log_metric('f1_weighted_live', f1_live)
log_distribution('feature_BB_Width_20', current_batch['BB_Width_20'])
```
</details>

<details><summary><strong>data drift</strong></summary>
Purpose: Change in statistical properties of input features vs training baseline (e.g., mean, variance, distribution shape).  
How/When/Where: Periodically compute PSI (Population Stability Index), KL divergence, or rolling z-score deviations for key indicators (volatility, TickVol_Z_20).  
Why: Upstream distribution shift can invalidate learned decision boundaries even if labeling logic unchanged.  
Relevance: Many engineered indicators (distances, ratios) assume relatively stable scaling; drift flags the need to re-run GA selection or re-optimize AMA/FDI parameters.  
Example:
```python
PSI = sum((p - q) * np.log(p / q) for p,q in zip(train_bins, live_bins))
if PSI > 0.25: trigger_alert('data_drift')
```
</details>

<details><summary><strong>concept drift</strong></summary>
Purpose: Change in the relationship between features and target (P(y|X)) even if feature distributions look similar.  
How/When/Where: Monitor live prediction error vs backtest baseline (e.g., rolling F1 delta) or use Page-Hinkley / DDM on error stream after the prediction horizon resolves.  
Why: Market microstructure or regime dynamics may alter feature-target mappings (e.g., volatility squeeze patterns no longer precede trend).  
Relevance: GA-chosen subset optimized for past cv_stability; drift suggests need to re-run GA to discover new stable subset under updated dynamics.  
Example (Page-Hinkley pseudo):
```python
ph.update(current_loss)
if ph.detected_change: schedule_retraining()
```
</details>

<details><summary><strong>retraining schedule</strong></summary>
Purpose: Policy specifying when to retrain (time-based, performance-based, drift-based).  
How/When/Where: Define cron (e.g., monthly) + conditional triggers (PSI > threshold OR F1 drop > X%). Retraining pipeline rebuilds engineered_features.csv, reruns GA feature selection & AMA/FDI optimization, then deploys new artifacts with version increment.  
Why: Balances freshness vs operational cost.  
Relevance: Frequent full GA runs are expensive; schedule ensures only meaningful drift prompts recomputation.  
Example:
```text
Time-based: 1st of month
Performance-based: 7-day F1 < (baseline_F1 - 0.03)
```
</details>

<details><summary><strong>model governance</strong></summary>
Purpose: Structured controls ensuring models are approved, documented, reproducible, and auditable.  
How/When/Where: Maintain run_summary.json, README, feature subset (selected_features_*.pkl), GA parameters, git commit hash, and approval checklist before promotion.  
Why: Financial context requires traceability & risk oversight.  
Relevance: GA introduces stochasticity; governance artifacts prove decisions (mutation_rate, dynamic strategy) and selected adaptive features were legitimate at deployment time.  
Example: Governance record includes {commit, run_timestamp, feature_count, validation_F1, drift_metrics}.
</details>

<details><summary><strong>reproducibility</strong></summary>
Purpose: Ability to recreate identical results (feature subset, metrics) from saved code, data, and parameters.  
How/When/Where: Enforced via fixed seeds, pinned dependencies (requirements.txt or lockfile), archived engineered_features.csv, and run_* directories.  
Why: Supports audit, comparison, debugging anomalies.  
Relevance: Core to validating GA performance claims (feature reduction % + F1 uplift).  
Example: Reload run_X/run_summary.pkl + feature_names.pkl + selected_features_rf.pkl → retrain and confirm metrics within tolerance.
</details>

<details><summary><strong>containerization (Docker)</strong></summary>
Purpose: Package code + dependencies + system libraries into immutable runtime image.  
How/When/Where: Build Docker image (Dockerfile) pre-deployment; run GA or inference inside container for consistency across dev/staging/prod.  
Why: Eliminates “works on my machine” discrepancies (e.g., different numpy/sklearn versions altering model hashing or pickle compatibility).  
Relevance: Guarantees that GA convergence behaviors and engineered feature calculations match training environment.  
Example (Dockerfile snippet):
```docker
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app
WORKDIR /app
```
</details>

<details><summary><strong>environment reproducibility (conda, pip-compile)</strong></summary>
Purpose: Deterministic dependency resolution (exact versions) via conda env.yaml or pip-compile generated requirements.lock.  
How/When/Where: Prior to GA runs, create locked file; CI verifies hash before executing notebooks.  
Why: Library drift (e.g., XGBoost minor changes) can shift CV scores → unstable feature subset selection.  
Relevance: Ensures comparability of cv_stability across historical run_* directories.  
Example:
```bash
pip-compile requirements.in > requirements.txt
pip install -r requirements.txt
```
</details>

<details><summary><strong>random seed management</strong></summary>
Purpose: Centralized setting of seeds (numpy, python.random, xgboost, tensorflow) for controlled stochastic operations (initial population, mutation).  
How/When/Where: Set early in notebook / module __init__ and store value in run_summary.json.  
Why: Allows re-evaluation of GA path; eases debugging unexpected subset differences.  
Relevance: Mutation + crossover outcomes influence selected feature subset; seed reproducibility is critical for governance.  
Example:
```python
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```
</details>

<details><summary><strong>git commit hash</strong></summary>
Purpose: Unique identifier of code state used for a run.  
How/When/Where: Captured at run start (git rev-parse HEAD) and stored in run_summary.json / README.  
Why: Trace which exact implementation produced artifacts; facilitates diffing when performance shifts.  
Relevance: Confirms that fitness function or GA operator tweaks explain metric changes, not data randomness.  
Example:
```bash
git rev-parse HEAD  # -> a1b2c3d...
```
</details>

<details><summary><strong>artifact versioning</strong></summary>
Purpose: Systematic labeling of saved models, feature lists, engineered features, and metrics (semantic or incremental).  
How/When/Where: Include version in filenames or metadata (model_v3_rf.pkl) plus run timestamp; optionally push to registry (e.g., MLflow, S3).  
Why: Supports rollback and comparison; prevents accidental overwrite.  
Relevance: Multiple GA experiments (different mutation schedules) can coexist; versioning clarifies lineage.  
Example metadata:
```json
{"artifact":"rf_model_ga","version":"1.2.0","commit":"a1b2c3d"}
```
</details>

<details><summary><strong>latency / throughput</strong></summary>
Purpose: Latency = time per inference; throughput = predictions per unit time.  
How/When/Where: Measured during staging using timed batches or load tests; optimize feature preprocessing + model size.  
Why: Ensures feasibility if moving from batch to near-real-time (hourly) inference.  
Relevance: GA reduces feature count → lowers preprocessing & model depth, improving latency and throughput margins.  
Example:
```python
start = time.time()
proba = model.predict_proba(X_batch)
lat_ms = (time.time()-start)*1000/len(X_batch)
```
</details>

<details><summary><strong>profiling and resource usage</strong></summary>
Purpose: Measure CPU, memory, and time hotspots to optimize GA fitness loop and feature engineering.  
How/When/Where: Use cProfile, line_profiler, memory_profiler around GA run; log resource stats.  
Why: Identify bottlenecks (e.g., redundant DataFrame slicing) that slow iterative experimentation.  
Relevance: Faster GA iterations enable broader parameter sweeps (pop_size, dynamic strategies) improving final subset quality.  
Example:
```bash
python -m cProfile -o ga.prof run_ga.py
```
</details>

<details><summary><strong>alerts and drift detection</strong></summary>
Purpose: Automated notifications (email, Slack, webhook) when monitored metrics breach thresholds (data drift, concept drift, latency spikes).  
How/When/Where: Scheduled job compares live metrics vs baselines; triggers alert pipeline.  
Why: Rapid awareness reduces time operating with degraded model.  
Relevance: Sparse GA subset may delay drift effects; alerts confirm when re-optimization becomes necessary (e.g., re-run AMA/FDI GA).  
Example (pseudo):
```python
if f1_live < f1_baseline - 0.04:
    send_alert('Performance degradation: retrain suggested.')
```
</details>

## Computational & Engineering Concerns

<details><summary><strong>parallelization</strong></summary>
Purpose: Execute independent computations concurrently across CPU cores to reduce wall‑clock time (e.g., evaluating many GA individuals).  
How/When/Where: Implemented via joblib.Parallel(n_jobs=-1) inside GeneticAlgorithm.evaluate_population() and potentially in GridSearchCV (n_jobs=-1). Triggered every GA generation during fitness evaluation (TimeSeriesSplit across feature subsets).  
Why: GA fitness cost scales ≈ pop_size × cv_folds; parallelizing masks amortizes expensive model fits.  
Relevance (project): Keeps GA (50 pop × 5 folds) feasible for iterative experimentation; without parallelism runtime would hinder parameter sweeps (mutation schedules, min/max feature ranges).  
Example:
```python
from joblib import Parallel, delayed
scores = Parallel(n_jobs=-1)(
    delayed(self.fitness_function)(ind) for ind in self.population
)
```
Edge: Balance n_jobs vs memory (many model instances in RAM simultaneously).
</details>

<details><summary><strong>GPU acceleration</strong></summary>
Purpose: Use GPU for massively parallel numeric kernels (matrix ops, gradient boosting, deep nets).  
How/When/Where: Not currently enabled (RF + CPU XGBoost). Could be activated by installing GPU-enabled XGBoost or future ANN reintroduction (TensorFlow/Keras).  
Why: Speeds training for large feature spaces or deeper boosting rounds, enabling larger pop_size or more generations under same time budget.  
Relevance (project): Optional acceleration path if feature count or dataset (multi‑asset expansion) increases and CPU parallelization saturates.  
Example (future):
```python
xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
```
Constraint: GPU VRAM must hold feature subsets; excessive mask variation could increase data transfer overhead.
</details>

<details><summary><strong>memory constraints / OOM</strong></summary>
Purpose: Manage finite RAM to avoid Out‑Of‑Memory errors when holding multiple DataFrames, model copies, and parallel processes.  
How/When/Where: Risk points: (a) joblib parallel GA evaluation spawning many processes, (b) storing large intermediate arrays (scaled copies), (c) GridSearchCV with wide parameter grids. Monitored during GA generations and tuning phase.  
Why: OOM kills processes → lost progress (if not checkpointed) and inconsistent run artifacts.  
Relevance (project): Ensuring sparsity (feature reduction) indirectly lowers memory footprint; careful n_jobs selection avoids overcommitting RAM.  
Mitigations:
- Reduce n_jobs or pop_size.
- Delete temporary objects (del X_sub) in fitness_function.
- Use dtype downcasting (float64 → float32) if acceptable.  
Example:
```python
if psutil.virtual_memory().available < safety_bytes:
    reduce_jobs()  # custom safeguard
```
</details>

<details><summary><strong>profiling (cProfile, line_profiler)</strong></summary>
Purpose: Identify runtime hotspots (functions consuming most CPU time) for optimization.  
How/When/Where: Run cProfile over ga.run() or feature engineering GA; apply line_profiler (@profile) to fitness_function or _create_* feature creators. Used after noticing slow GA convergence or long generations.  
Why: Focus optimization effort (e.g., avoid repeated DataFrame slicing) rather than premature micro‑tuning.  
Relevance (project): Shorter generations → ability to test more GA configurations (diversity thresholds, natural disaster timing) improving overall feature subset quality.  
Example:
```bash
python -m cProfile -o ga.prof run_ga.py
snakeviz ga.prof
```
Insight: If fitness_function dominates, cache model templates or pre-slice numpy arrays.
</details>

<details><summary><strong>scaling to clusters</strong></summary>
Purpose: Distribute workload across multiple machines (cluster) for larger datasets / wider search (higher pop_size, more folds).  
How/When/Where: Future step using Dask (distributed scheduler) or Ray to parallelize GA evaluation and feature engineering parameter searches beyond a single node.  
Why: Horizontal scaling overcomes single-host CPU/memory limits, enabling exploration of richer metaheuristic variants (multi-population islands).  
Relevance (project): Facilitates scaling from single FX pair to multi‑asset or higher frequency bars without sacrificing GA depth or stability metrics.  
Example (concept):
```python
from dask.distributed import Client
client = Client()
# wrap fitness evaluations in dask.delayed
```
Caution: Serialization overhead—prefer sending numpy arrays vs large DataFrames each task.
</details>

<details><summary><strong>batching and streaming</strong></summary>
Purpose: Process data in manageable chunks (batching) or continuous increments (streaming) instead of full in‑memory loads.  
How/When/Where: Future enhancement for ingesting high-frequency or multi‑asset OHLCV; streaming pipeline could append latest bars, re-engineer incremental features, and trigger lightweight retraining.  
Why: Reduces peak memory, lowers latency to incorporate new data, enables near real‑time drift monitoring.  
Relevance (project): Hourly data currently fits memory; streaming architecture prepares for production deployment (model inference each hour + periodic GA re-run).  
Example (incremental feature update concept):
```python
new_rows = fetch_latest()
data = pd.concat([data, new_rows]).tail(lookback_window)
recompute_recent_indicators(data)
```
</details>

<details><summary><strong>checkpointing</strong></summary>
Purpose: Periodically persist intermediate GA state (population, best individual, fitness history) to allow resume after interruption.  
How/When/Where-+-+-+-+-+
<details><summary><strong>parallelization</strong></summary>
Purpose: Execute multiple independent computations at the same time to shorten total runtime (e.g., scoring many GA individuals).  
How/When/Where: Implemented via joblib.Parallel(n_jobs=-1) inside GeneticAlgorithm.evaluate_population(); also used by GridSearchCV (n_jobs). Runs every GA generation during fitness evaluation (each individual → TimeSeriesSplit CV).  
Why: GA cost ≈ population_size × cv_folds × model_fit_time; parallel workers decrease wall-clock delay.  
Relevance (project): Enables practical 50×5 GA loops and rapid experimentation with mutation / crossover strategies and feature-engineering sweeps.  
Example:
```python
from joblib import Parallel, delayed
fitness_scores = Parallel(n_jobs=-1)(
    delayed(self.fitness_function)(ind) for ind in self.population
)
```
Tip: Adjust n_jobs if memory pressure appears (see memory constraints).
</details>

<details><summary><strong>GPU acceleration</strong></summary>
Purpose: Use graphics processors for massively parallel numeric kernels (tree histogram building, matrix ops, deep learning).  
How/When/Where: Not active by default (CPU RandomForest + CPU XGBoost). Future: XGBoost with tree_method='gpu_hist' or reintroduce ANN (TensorFlow/Keras with GPU).  
Why: Reduces per‑model training time, permitting larger populations, more generations, or broader hyperparameter grids under same time budget.  
Relevance (project): Scaling path if expanding to multi‑asset datasets or adding deeper boosted trees; keeps GA turnaround acceptable.  
Example (future):
```python
xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor',
                    n_estimators=200, max_depth=6)
```
Constraint: VRAM must hold feature batch; excessive host↔device transfers reduce gains.
</details>

<details><summary><strong>memory constraints / OOM</strong></summary>
Purpose: Manage limited RAM so parallel model fits and large DataFrames do not trigger out‑of‑memory termination.  
How/When/Where: Risk points: (a) many parallel GA fitness tasks, (b) multiple scaled copies (train/val/test), (c) large GridSearchCV parameter grids. Monitor with psutil or OS tools during GA generations and tuning.  
Why: Memory saturation causes failures and lost progress (if no checkpoint).  
Relevance (project): Feature reduction (GA objective) also lowers memory footprint (fewer columns per model). Proper n_jobs/pop_size balance keeps workflow stable.  
Mitigations: lower n_jobs, reduce pop_size, cast float64→float32, delete temporary slices inside fitness_function, avoid unnecessary DataFrame copies.  
Example:
```python
X_sub = self.X_train.loc[:, mask_features].values.astype('float32')
del X_sub  # after scoring
```
</details>

<details><summary><strong>profiling (cProfile, line_profiler)</strong></summary>
Purpose: Locate runtime hotspots to focus optimization (avoid premature micro‑tuning).  
How/When/Where: Run cProfile over ga.run(); use snakeviz for visualization; apply line_profiler (@profile decorator) to fitness_function, feature construction loops, or engineering GA parameter evaluation. Trigger after noticing slow generations or long tuning cycles.  
Why: Targeted refactors (e.g., precomputing numpy arrays, avoiding repeated .loc slicing) yield largest time savings.  
Relevance (project): Faster generations → more experimental iterations (adjust diversity_threshold, calm_before_storm) → potentially better feature subsets & stability.  
Example:
```bash
python -m cProfile -o ga.prof run_ga.py
snakeviz ga.prof
```
Insight: If fitness_function dominates >70% time, cache model templates or reuse TimeSeriesSplit indices.
</details>

<details><summary><strong>scaling to clusters</strong></summary>
Purpose: Distribute GA or feature‑engineering evaluations across multiple machines for larger datasets or wider metaheuristic searches.  
How/When/Where: Future enhancement using Dask (distributed scheduler) or Ray; wrap fitness calls in delayed / remote functions; central scheduler coordinates workers.  
Why: Horizontal scaling overcomes single-host CPU + memory limits; enables larger pop_size, more cv folds, or island-model GAs.  
Relevance (project): Supports multi‑asset / higher frequency expansion without inflating per‑run latency.  
Example (concept):
```python
from dask.distributed import Client
client = Client()
# fitness_function wrapped via dask.delayed
```
Consider: Minimize serialization—work with contiguous numpy arrays, not large nested objects.
</details>

<details><summary><strong>batching and streaming</strong></summary>
Purpose: Process data incrementally (batches) or continuously (stream) instead of full in-memory recomputation.  
How/When/Where: Future ingestion loop: fetch latest hour, append, recompute only rolling windows affected, update engineered_features.csv incrementally. Streaming inference each bar; periodic batch GA re-run (e.g., weekly).  
Why: Reduces latency to updated predictions and memory usage on growing histories.  
Relevance (project): Positions pipeline for production monitoring (drift detection using newest slices) without full rebuild each run.  
Example (incremental update):
```python
new = fetch_latest()
data = pd.concat([data, new]).tail(max_lookback)
update_recent_indicators(data)
```
</details>

<details><summary><strong>checkpointing</strong></summary>
Purpose: Persist intermediate GA or feature‑engineering state (population, best individual, fitness history) so work can resume after interruption.  
How/When/Where: After every N generations (e.g., every 3–5) or when a new global best appears: save pickle in run_*/checkpoints/. On resume: load state and continue generation loop.  
Why: Protects progress against kernel resets, time limits, or memory issues; enables long‑running larger searches.  
Relevance (project): Supports experimentation with larger max_generations without risking total restart cost.  
Example:
```python
state = {
  'generation': g,
  'population': self.population,
  'best_mask': self.best_individual,
  'fitness_history': self.fitness_history
}
pickle.dump(state, open(f"{run_dir}/checkpoints/gen_{g}.pkl","wb"))
```
</details>

<details><summary><strong>serialization formats (pickle, joblib, ONNX)</strong></summary>
Purpose: Persist models, feature masks, metadata in portable form.  
How/When/Where: pickle/joblib used now (selected_features_*.pkl, rf_baseline.pkl, scaler.pkl). ONNX (future) for standardized inference deployment (language / platform agnostic).  
Why: Enables reproducibility, audit, later comparative retraining without rerunning GA; ONNX adds interoperability (microservice / cloud inference).  
Relevance (project): Accurate restoration of feature ordering & scaler essential for consistent cv_stability replication; potential ONNX export streamlines deployment.  
Example (current):
```python
with open(path/'selected_features_rf.pkl','wb') as f:
    pickle.dump((mask, fitness, names), f)
```
Caution: Pickle not secure against untrusted input; only load from trusted run directories.
</details>

<details><summary><strong>deterministic runs</strong></summary>
Purpose: Guarantee repeatable outcomes (same feature subset, metrics) given identical code, data, and seeds.  
How/When/Where: Set np.random.seed, random.seed, model random_state, (optionally) os.environ['PYTHONHASHSEED']; log seeds + dependency versions in run_summary.json. Avoid nondeterministic GPU kernels unless necessary (set deterministic flags when using TensorFlow/PyTorch).  
Why: Enables fair baseline vs enhanced comparisons and facilitates debugging small performance deltas.  
Relevance (project): Critical for validating that improvements stem from feature selection / engineering changes, not random drift; supports governance and auditability.  
Example:
```python
SEED = 42
import random, numpy as np
random.seed(SEED); np.random.seed(SEED)
rf = RandomForestClassifier(random_state=SEED)
```
Note: Some algorithms (multi-threaded tree building) may introduce minor nondeterminism; single-thread or stable versions if exact bitwise repeat needed.
</details>

## Notes & Common Pitfalls (short)

> Never break time series order in splits
> Use TimeSeriesSplit or walk-forward validation for GA fitness evaluation
> Keep min/max features balanced to maintain diversity
> Save fitted scaler and ensure pickle compatibility
> Parallel fitness evaluation can be memory heavy
> Watch for lookahead bias and data leakage
> Monitor model performance over time and detect drift early
