# Advanced Air Quality Analysis - Automated Version
import statistics
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from scipy.stats import normaltest, skew
import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime
import time
import json
import sys
import os

# Selected CSV Folder
combined_df = pl.scan_csv("atmospheric_data/*.csv")

# OS Environment Thread Limit to Prevent too much laptop overhead
# (Not Necessary, can comment out)
os.environ["POLARS_MAX_THREADS"] = "1"

# Basic Analysis
analysis = combined_df.select([
    "State Name",
    "Parameter Name",
    "Arithmetic Mean",
    "AQI",
    "Observation Count"
]).group_by(["State Name", "Parameter Name"]).agg([
    pl.col("Arithmetic Mean").mean().alias("Avg_Mean"),
    pl.col("AQI").max().alias("Max_AQI"),
    pl.col("Observation Count").sum().alias("Total_Observations")
])

# Data Collect
#results = analysis.collect()

# Previously-used DataFrames (Less Columns)
# Can be used again properly if separated by
# cache system at the minimum
"""
prepared_df = (
    combined_df
    .with_columns(
        pl.col("Date Local").str.to_date().dt.month().alias("Month")
    )
    .select(["State Name", "Date Local", "Month", "Parameter Name", "AQI", "Arithmetic Mean"])
    .collect()
)

wide_df = (
    prepared_df
    .pivot(
        on="Parameter Name",
        index=["State Name", "Date Local", "Month", "AQI"],
        values="Arithmetic Mean",
        aggregate_function="mean"
    )
    .drop_nulls()
)
"""

# Filter and Select
prepared_df = (
    combined_df
    .filter(pl.col("Observation Percent") >= 75) # Optional: Quality filter
    .with_columns(
        pl.col("Date Local").str.to_date().dt.month().alias("Month")
    )
    .select([
        "State Name", "Date Local", "Month", "Parameter Name", "AQI", 
        "Arithmetic Mean", "Latitude", "Longitude", "Observation Count"
    ])
    .collect()
)

# The "Granular" Pivot (Previous Version)
# Note: Adding Lat/Long to the index keeps individual sensor sites separate!
"""
wide_df = (
    prepared_df
    .pivot(
        on="Parameter Name",
        index=["State Name", "Date Local", "Month", "Latitude", "Longitude", "Observation Count", "AQI"],
        values="Arithmetic Mean",
        aggregate_function="mean"
    )
    .drop_nulls()
)
"""

# Current Wide DataFrame (final Polars manipulation)
wide_df = (
    prepared_df
    .pivot(
        on="Parameter Name",
        index=["State Name", "Date Local", "Month", "Latitude", "Longitude", "Observation Count", "AQI"],
        values="Arithmetic Mean",
        aggregate_function="mean"
    )
)

# Fill Nulls with the mean
wide_df = wide_df.fill_null(strategy="mean")

# Nulls don't need to be dropped if
# The mean replaces them
#wide_df = wide_df.drop_nulls()

# This cleans the data before we start selecting features
wide_df = wide_df.with_columns(
    pl.col("AQI").cast(pl.Float64, strict=False)
)
# AQI nulls are dropped.
# Essentially, no proper y-value cast isn't worth
# keeping.
wide_df = wide_df.drop_nulls(subset=["AQI"])

# Load System Configuration
def load_system_config(config_path="model_control.json"):
    """
    Sets the 'Rules of the Road' for the entire script.
    """
    default_config = {
        "min_r2_threshold": 0.5,
        "include_models": ["Polynomial", "Ridge", "Linear"],
        "scout_features": True,
        "force_refresh": False,
        "history": {}
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return {**default_config, **json.load(f)}
    return default_config

# Configuration Score Sync
def sync_config_with_scores(model_results, config, smape_score = 0, config_path="model_control.json"):
    """
    Updates the JSON cache with the latest performance data for a specific model.
    No models are removed; only performance history is updated.
    """
    model_name = model_results["name"]
    current_r2 = model_results["score"]

    #if model_name not in config.get("include_models", []):
    #    config.setdefault("include_models", []).append(model_name)

    if "completed_models" not in config:
        config["completed_models"] = []

    if model_name not in config["completed_models"]:
        config["completed_models"].append(model_name)

    # Initialize a 'history' section in your JSON if it doesn't exist
    if "model_history" not in config:
        config["model_history"] = {}

    # 2. Update or Create the entry for this specific model
    # We store the 'Best Seen' R2 and the 'Latest' SMAPE
    if model_name not in config["model_history"]:
        config["model_history"][model_name] = {
            "best_r2": current_r2,
            "latest_smape": smape_score,
            "runs_completed": 1
        }
    else:
        # Update the best score if this run was superior
        prev_best = config["model_history"][model_name]["best_r2"]
        config["model_history"][model_name]["best_r2"] = max(prev_best, current_r2)
        config["model_history"][model_name]["latest_smape"] = smape_score
        config["model_history"][model_name]["runs_completed"] += 1

    # Update global metadata
    config["last_run_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Keep the include_models list intact (No Pruning)
    if model_name not in config.get("include_models", []):
        config.setdefault("include_models", []).append(model_name)

    # Save to JSON
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Basic Features
poly_categoricals = ["State Name"]
exclude = ["State Name", "Date Local", "AQI"]
poly_numericals = [col for col in wide_df.columns if col not in exclude]

# Feature Concatenation
feature_cols = poly_categoricals + poly_numericals

# Filter 
#wide_df = wide_df.filter(pl.col("AQI").is_not_null())

# Assigning an X and y variable to prepare train and test data
# Using already-filtered "wide_df"
X = wide_df.select(feature_cols).to_pandas()
y = wide_df["AQI"].to_pandas()

# Final safety check (X and y rows should be equal)
#print(f"📊 Features: {len(X)} rows | Target: {len(y)} rows")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""
AQI Skewness is printed. Such a statistic would be more relevant
if an automated system required it. However, we already have
a magnitude of skewness that isn't too high, so for now, the
data is not going to be analyzed based on the skewness.

The AQI Skewness will be important, though, even in a single
dataset, as the AQI Skewness for the small number of columns
is only -0.07, whereas for the larger amount of variables,
it comes out to 0.88.
"""
print(f"AQI Skewness: {skew(y_train):.2f}")

# Coefficient DataFrame Function not yet used
def coefficient_df(model, model_type, feature_names, coefficients):
    input_to_model = model.named_steps['prep'].get_feature_names_out()
    features = model.named_steps[model_type].get_feature_names_out(input_to_model)
    coefficients = model.named_steps['regressor'].coef_

    coeff_df = pd.DataFrame({'Feature': feature_names, 'Weight': coefficients})
    coeff_df['Abs_Weight'] = coeff_df['Weight'].abs()
    coeff_df = coeff_df.sort_values(by='Abs_Weight', ascending=False)
    return coeff_df


# Function to find top features
def feature_find(fitted_model, numeric_features, data_to_match = None):
    """
    Transforms raw data and returns a labeled DataFrame.
    Guarantees a 1D index to prevent ValueError.
    """
    
    # Extract and Flatten Names
    try:
        # get_feature_names_out() usually returns a 1D numpy array
        raw_names = fitted_model.named_steps['prep'].get_feature_names_in()
        
    except Exception as e:
        feature_names = list(numeric_features) + ["State Name"]
    
    # AUTOMATED SAFETY CHECK
    # If we have the data, ensure the list length matches the column count
    if data_to_match is not None:
        actual_col_count = data_to_match.shape[1]
        if len(feature_names) != actual_col_count:
            # If we are missing one, it's almost always the 'State Name' 
            # that was appended to the data but not the name list
            if actual_col_count == len(feature_names) + 1:
                feature_names.append("State Name")
            # If we have too many, truncate (or handle as needed)
            else:
                feature_names = feature_names[:actual_col_count]
                
    return feature_names


# Class Factory, as oppose to the original analysis,
# which had its tools, such as encoders and pipelines,
# manually-encoded.
class AQIModelFactory:
    def __init__(self, cat_features, default_num):
        self.cat_features = cat_features
        self.default_num = default_num
    
    def _get_encoder(self, encoder_type):
        """Internal helper to select the categorical shield."""
        if encoder_type == "onehot":
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif encoder_type == "ordinal":
            return OrdinalEncoder()
        else:
            raise ValueError(f"Invalid encoder: {encoder_type}")

    def build_pipeline(self, model_type="Polynomial", encoder_type="onehot", custom_num=None, use_scaler=True):
        """Assembles the full pipeline blueprint."""
        active_num = custom_num if custom_num is not None else self.default_num

        # Define Numeric Transformer
        num_trans = StandardScaler() if use_scaler else 'passthrough'
        
        # Build Preprocessor
        preprocessor = ColumnTransformer(transformers=[
            ('cat', self._get_encoder(encoder_type), self.cat_features),
            ('num', num_trans, active_num)
        ])

        # Assemble Steps
        steps = [('prep', preprocessor)]
        
        if model_type == "Polynomial":
            steps.append(('poly', PolynomialFeatures(degree=2)))
            steps.append(('regressor', Ridge(alpha=1.0)))
        else:
            steps.append(('regressor', LinearRegression()))

        return Pipeline(steps)
    

    # Run permutation importance for a given model
    # Limited to "Polynomial" as of now
    def scout_features(self, X_train, y_train, X_test, y_test):
        # Build and fit a baseline model for scouting
        scout_pipe = self.build_pipeline(model_type="Polynomial", encoder_type="ordinal")
        scout_pipe.fit(X_train, y_train)
        # Run Permutation Importance (fan-friendly with n_jobs=1)
        perm_importance = permutation_importance(
            scout_pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
        )
        # Filter for features with importance > 0
        importance_df = pd.DataFrame({
            'Feature': X_test.columns, 
            'Importance': perm_importance.importances_mean
        })
        good_features = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()
        # Return only the numerical ones (the ones we can toggle in ColumnTransformer)
        return [f for f in good_features if f in self.default_num]
    
    # Best-model finder    
    def find_best_model(self, X_train, y_train, X_test, y_test, candidates):
        results = {}
        winner_info = {"name": "", "score": -float('inf'), "model": None}

        for name, params in candidates.items():

            custom_feats = params[2] if len(params) > 2 else None

            # Unpack params (model_type, encoder_type)
            model = self.build_pipeline(model_type=params[0], encoder_type=params[1], custom_num=custom_feats)
            current_X_train = X_train[custom_feats + self.cat_features] if custom_feats else X_train
            current_X_test = X_test[custom_feats + self.cat_features] if custom_feats else X_test
            model.fit(current_X_train, y_train)
            score = model.score(current_X_test, y_test)
            results.update({len(results) + 1 : {"name" : name, "score": score, "model": model}})

            if score > winner_info["score"]:
                winner_info.update({"name": name, "score": score, "model": model})
        
        return winner_info, results

# Class factory instantiation
factory = AQIModelFactory(poly_categoricals, poly_numericals)

# Features are scouted early to allow permutation features
# into a later list of models with parameters
perm_numeric_features = factory.scout_features(X_train, y_train, X_test, y_test)

# Previous coefficient output
#coeff_df = pd.DataFrame({'Feature': feature_names, 'Weight': coefficients})
#coeff_df['Abs_Weight'] = coeff_df['Weight'].abs()
#coeff_df = coeff_df.sort_values(by='Abs_Weight', ascending=False)

#print("\n--- TOP COEFFICIENT WEIGHTS ---")
#print(coeff_df[['Feature', 'Weight']].head(10))

# List of models to be run
model_candidates = {
    "Polynomial_OneHot": ("Polynomial", "onehot"),
    "Polynomial_Ordinal": ("Polynomial", "ordinal"),
    "Linear_Ordinal": ("Linear", "ordinal"),
    "Poly_Pinpointed_Ordinal": ("Polynomial", "ordinal", perm_numeric_features)
}

# Load the "Rules"
config = load_system_config("model_control.json")
allowed_list = config.get("include_models", [])
completed = config.get("completed_models", [])

# Filter the dictionary: Only include models that are in the JSON 'allowed' list
active_candidates = {
    name: params for name, params in model_candidates.items()
    #if any(m in name for m in allowed_list)
    if name not in completed
}

# Not fully ready: No active candidates ends the program.
# Ideally, it would run a comparison of which model is
# best, if not a display of the metrics of the best model.
if not active_candidates:
    print("📋 Status: No new models to train.")
    
    # Check if we have at least one successful model to work with
    if config.get("model_history"):
        pass
        #print("💡 Transitioning to Preparation Mode: Loading best cached model...")
        # Add logic here to load your best model from disk
        # (e.g., joblib.load(config["best_model_path"]))
    else:
        print("❌ No training candidates and no history found. Please check config.")
    
    sys.exit(0) # Stop here so we don't try to run the Factory



# Output of the list of models to run
#print(active_candidates)

# Finding best model
winner, all_scores = factory.find_best_model(X_train, y_train, X_test, y_test, active_candidates)

# Output of winner and all scores
#print(winner)
#print(all_scores)

# Loop variable - only marks when the first score will be calculated
# to have 
spot = 0

for scored_model in all_scores:
    # Break to prevent too much overhead
    time.sleep(2)
    # Conditional if score beats threshold. The SMAPE will simply default to 0 if below threshold.
    if all_scores[scored_model]["score"] >= config.get("min_r2_threshold", 0.5):
        if spot == 0:
            # Instatiate the "true y" and "y prediction"
            y_true = np.array(y_test)
            y_pred = all_scores[scored_model]["model"].predict(X_test)
            # Instantiate the file path and verify it exists
            file_path = "model_performance_ledger.csv"
            file_exists = os.path.isfile(file_path)
        
        # Calculation for SMAPE (returns a percentage 0-100)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        # Avoid true zero division by adding a tiny epsilon
        diff = np.abs(y_true - y_pred) / (denominator + 1e-10)
        smape_score = 100 * np.mean(diff)
        
        # Create the Ledger Entry
        # This captures the 'Health' of the model rather than raw data points
        performance_entry = pd.DataFrame([{
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Model_Name': all_scores[scored_model]["name"],
            'R2_Score': all_scores[scored_model]["score"],
            'SMAPE_Score': round(smape_score, 4),
            'Sample_Size': len(y_test)
        }])
        performance_entry.to_csv(file_path, mode='a', index=False, header=not file_exists)
        print(f"📈 Performance Logged: {all_scores[scored_model]['name']} | SMAPE: {smape_score:.2f}%")
        sync_config_with_scores(all_scores[scored_model], config, smape_score)
    else:
        # 0 SMAPE default
        sync_config_with_scores(all_scores[scored_model], config)

    

