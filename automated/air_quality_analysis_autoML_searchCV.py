# Advanced Air Quality Analysis - Automated
# V4
import statistics
from datetime import datetime
import time
import json
import sys
import os

start = time.time()

USE_CUSTOM_SKLEARN = True
record_model = True
record_time = True

# 1. Setup path
custom_path = os.getenv("CUSTOM_SKLEARN")


# 2. Force the environment variable so CHILD PROCESSES (n_jobs) see it
if USE_CUSTOM_SKLEARN and os.path.exists(custom_path):
    # Aggressive cache clearing for the whole "data stack"
    for mod in list(sys.modules.keys()):
        if mod.startswith(('numpy', 'sklearn', 'scipy', 'pandas')):
            del sys.modules[mod]

    # This ensures RandomizedSearchCV workers look here first
    os.environ["PYTHONPATH"] = custom_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    
    # Update current session path
    if custom_path not in sys.path:
        sys.path.insert(0, custom_path)

"""
# Add to front
if custom_path not in sys.path:
    sys.path.insert(0, custom_path)

# Clear the cache aggressively
for mod in list(sys.modules.keys()):
    if mod.startswith(('numpy', 'sklearn', 'scipy')):
        del sys.modules[mod]

"""
# df_sample = df.sample(fraction=0.1, seed=42)

# Now import
import numpy as np
from scipy.stats import normaltest, skew
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn

print(f"LOG: Success! NumPy {np.__version__} loaded from {np.__file__}")
print(f"LOG: Pandas {pd.__version__} from {pd.__file__}")
print(f"LOG: Success! Sklearn {sklearn.__version__} loaded from {sklearn.__file__}")


if USE_CUSTOM_SKLEARN:
    # We point to the folder you created
    custom_path = r"C:\Users\jp21j\AppData\Local\Programs\Python\Python311\Lib\site-packages\scikit_learn_2_0"
    
    # Insert it at index 0 so it's the very first place Python looks
    sys.path.insert(0, custom_path)
    print("LOG: Using Custom Scikit-Learn (v1.3.0) from scikit_learn_2_0")
else:
    print("LOG: Using Default Scikit-Learn")


import polars as pl
import polars.selectors as cs
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, PolynomialFeatures, StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

"""
This is different from traditional CSV Loading.
As oppose to the usual Pandas read, the
"scan_csv" option puts all data from the same
folder and makes it into one dataset.

After that, methods such as selecting and
grouping put the data into proper form to
analyze.
"""

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


"""
DataFrame with:
- Datetime added with Month
- Duration Hours to Account for Sample Duration
(may be edited as Sample Duration may simply be a factor)
"""
prepared_df = (
    combined_df
    .filter(pl.col("Observation Percent") >= 75) # Optional: Quality filter
    .with_columns([
        pl.col("Sample Duration")
          .str.extract(r"(\d+)")
          .cast(pl.Float64)
          .alias("Duration_Hours"),
        pl.col("Date Local").str.to_date().dt.month().alias("Month")]
    )
    .select([
        "State Name", "Date Local", "Month", "Parameter Name", "AQI", 
        "Duration_Hours", "Arithmetic Mean", "Observation Count",
        "Latitude", "Longitude"
    ])
    .collect()
)

"""
state_lookup = (
    prepared_df
    .select(["State Name", "AQI"])
    .with_columns(pl.col("AQI").cast(pl.Float64, strict=False))
    .drop_nulls() # Remove rows where AQI is missing
    .group_by("State Name")
    .agg(pl.col("AQI").mean().alias("State_AQI_Mean"))
)

print(state_lookup)
"""

# Current Wide DataFrame (final Polars manipulation)
wide_df = (
    prepared_df
    .pivot(
        on="Parameter Name",
        index=["State Name", "Date Local", "Month", "Latitude", "Longitude", "Observation Count", 
               "Duration_Hours", "AQI"],
        values="Arithmetic Mean",
        aggregate_function="mean"
    )
)

"""
wide_df = (
    wide_df
    .join(state_lookup, on="State Name", how="left")
)
"""

# This cleans the data before we start selecting features
wide_df = wide_df.with_columns(
    pl.col("AQI").cast(pl.Float64, strict=False)
)

wide_df = wide_df.with_columns(
    pl.col("AQI").fill_null(strategy="mean")
)

# This cleans the data before we start selecting features
wide_df = wide_df.with_columns(
    pl.col("AQI").cast(pl.Float64, strict=False)
)

wide_df = wide_df.with_columns(
    pl.col("AQI").fill_null(strategy="mean")
)

print(wide_df.columns)

"""
wide_df = wide_df.with_columns(
    (pl.col("Ozone") * pl.col("State_AQI_Mean")).alias("Ozone_State_Interaction"),
    (pl.col("Sulfur dioxide") * pl.col("State_AQI_Mean")).alias("SO2_State_Interaction"),
    (pl.col("Nitrogen dioxide (NO2)") * pl.col("State_AQI_Mean")).alias("NO2_State_Interaction"),
    (pl.col("Carbon monoxide") * pl.col("State_AQI_Mean")).alias("CO_State_Interaction")
)
"""

# Filling nulls (particularly useful with top pollutants connected to AQI)
wide_df = wide_df.fill_null(0)

wide_df = wide_df.sample(fraction=0.1, seed=42)

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

def route_predictions(models, X_test):
    all_predictions = np.zeros(len(X_test))

    # Get the indices for each group
    ca_idx = X_test[X_test['State Name'] == 'CA'].index
    hi_idx = X_test[X_test['State Name'] == 'HI'].index
    others_idx = X_test[~X_test['State Name'].isin(['CA', 'HI'])].index

    # Fill the placeholder at the EXACT locations
    if not ca_idx.empty:
        all_predictions[X_test.index.get_indexer(ca_idx)] = models["California"].predict(X_test.loc[ca_idx])

    if not hi_idx.empty:
        all_predictions[X_test.index.get_indexer(hi_idx)] = models["Hawaii"].predict(X_test.loc[hi_idx])

    if not others_idx.empty:
        all_predictions[X_test.index.get_indexer(others_idx)] = models["Others"].predict(X_test.loc[others_idx])
    
    return all_predictions
    """

    all_preds = np.zeros(len(X_test))
    col = 'State Name'
    
    # 1. Identify rows
    ca_mask = (X_test[col] == 'California').values
    hi_mask = (X_test[col] == 'Hawaii').values
    # 2. "Others" is just the logical "NOT" of the two specialists
    others_mask = ~(ca_mask | hi_mask)

    # 3. Predict using the specific specialists or the generalist
    if ca_mask.any():
        all_preds[ca_mask] = models["California"].predict(X_test[ca_mask])
        
    if hi_mask.any():
        all_preds[hi_mask] = models["Hawaii"].predict(X_test[hi_mask])
        
    if others_mask.any():
        # This will now work because "Others" is a key in your dict
        all_preds[others_mask] = models["Others"].predict(X_test[others_mask])

    return all_preds
    """
    

# Configuration Score Sync
def sync_config_with_scores(model_results, config, smape_score = 100.00, config_path="model_control.json"):
    """
    Updates the JSON cache with the latest performance data for a specific model.
    No models are removed; only performance history is updated.
    """
    model_name = model_results["name"]
    current_r2 = model_results["score"]
    if "completed_models" not in config:
        config["completed_models"] = []

    if model_name not in config["completed_models"]:
        config["completed_models"].append(model_name)

    # Initialize a 'history' section in your JSON if it doesn't exist
    if "model_history" not in config:
        config["model_history"] = {}

    # Update or Create the entry for this specific model
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

# Assigning an X and y variable to prepare train and test data
# Using already-filtered "wide_df"
X = wide_df.select(feature_cols).to_pandas()
y = wide_df["AQI"].to_pandas()

# Coefficient DataFrame Function not used but ready
def coefficient_df(model, model_type):
    input_to_model = model.named_steps['prep'].get_feature_names_out()
    feature_names = model.named_steps[model_type].get_feature_names_out(input_to_model)
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
        elif encoder_type == "target":
            # 'auto' smooths the means to prevent overfitting on small states
            return TargetEncoder(target_type="continuous")
        else:
            raise ValueError(f"Invalid encoder: {encoder_type}")
        
    # Transforms the Variance Inflation Factor (VIF)
    def vif_fit_transform(self, X_train):
        # Build the pipeline using the factory method in a wrapper
        wrapper = factory.build_pipeline(model_type="Polynomial", use_scaler=True)
        
        # Make the pipeline the regressor in the wrapper
        internal_pipeline = wrapper.regressor

        if hasattr(internal_pipeline, 'estimator'):
            # RandomizedSearchCV stores the pipeline in '.estimator' before it is fit
            internal_pipeline = internal_pipeline.estimator
        else:
            # It's already a standard Pipeline
            pass

        # Grab just the preprocessor part
        preprocessor = internal_pipeline.named_steps['prep']

        # Fit and Transform the training data
        # This handles the IterativeImputer (RF) and the StandardScaler automatically
        X_transformed = preprocessor.fit_transform(X_train)

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Get the feature names (important for One-Hot encoded columns)
        features = preprocessor.get_feature_names_out()
        vif_results = []

        for i in range(len(features)):
            vif = variance_inflation_factor(X_transformed, i)
            vif_results.append({"feature": features[i], "VIF": vif})


        # return pd.DataFrame(vif_results).sort_values("VIF", ascending=False)
        return (
            pl.DataFrame(vif_results)
            .sort("VIF", descending=True)
        )

    def build_pipeline(self, model_type="Polynomial", encoder_type="onehot", custom_num=None, use_scaler=True, param_dist="no_search"):
        """Assembles the full pipeline blueprint."""
        active_num = custom_num if custom_num is not None else self.default_num

        # Define Numeric Transformer
        #num_trans = StandardScaler() if use_scaler else 'passthrough'

        numeric_processor = Pipeline([
        ('impute', IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=5),
            max_iter=3
        )),
        ('scaler', StandardScaler()),
        ])
        
        """
        spatial_transformer = Pipeline([
            # Encode the State and keep Lat/Long
            ('encoder', ColumnTransformer([
            ('state_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['State Name']),
            ('coords_pass', 'passthrough', ['Latitude', 'Longitude'])
            ])),
            # Scale everything (PCA requires this!)
            ('scaler', StandardScaler(with_mean=False)), # with_mean=False for sparse matrices
            # Reduce everything into a few components
            ('pca', PCA(n_components=3)) 
        ])
        
        # Build Preprocessor
        preprocessor = ColumnTransformer(transformers=[
            ('cat', self._get_encoder(encoder_type), self.cat_features),
            ('spatial_pca', spatial_transformer, ['Latitude', 'Longitude', 'State Name']),
            ('num', numeric_processor, active_num)
        ])
        """

        # Simplified Preprocessor (Removing the spatial_transformer block)
        preprocessor = ColumnTransformer(transformers=[
            # Handle State Name directly via OneHot
            ('cat', self._get_encoder(encoder_type), self.cat_features + ['State Name']), 
            # Pass Latitude and Longitude directly (No PCA/Scaling)
            ('coords', 'passthrough', ['Latitude', 'Longitude']),
            # Handle other numeric features
            ('num', numeric_processor, active_num)
        ])

        # Assemble Steps
        steps = [('prep', preprocessor)]
        
        
        if model_type == "Polynomial":
            steps.append(('poly', PolynomialFeatures(degree=2)))
            steps.append(('regressor', Ridge(alpha=1.0)))
        elif model_type == "RandomForest":
            # RF doesn't require scaling, but it doesn't hurt if your pipeline uses it
            regressor = RandomForestRegressor(
                n_estimators=100,      # Balance between accuracy and laptop RAM
                max_depth=15,          # Prevents the model from 'memorizing' exact rows
                min_samples_leaf=5,    # Ensures clusters have enough data to generalize
                n_jobs=-1,             # Respects your POLARS_MAX_THREADS/Local Cores
                random_state=42
            )
            steps.append(('regressor', regressor))
        else:
            steps.append(('regressor', LinearRegression()))
        
        full_pipeline = Pipeline(steps)

        param_grids = {
            "Linear": {
                "regressor__fit_intercept": [True, False],
                #"prep__spatial_pca__pca__n_components": [2, 3, 5]
            },
            "Polynomial": {
                "regressor__fit_intercept": [True, False],
                "poly__degree": [2, 3], # Removed 'prep__' because it's a top-level step
                #"prep__spatial_pca__pca__n_components": [2, 3]
            },
            "RandomForest": {
                "regressor__n_estimators": [100, 200, 500],
                "regressor__max_depth": [10, 20, None],
                "regressor__min_samples_leaf": [1, 5, 10],
                #"prep__spatial_pca__pca__n_components": [5, 10]
            },
            "no_search": {
                'regressor__copy_X': [True] # A harmless parameter for Linear/Ridge
            }
        }
        
        if param_dist == "no_search":
            active_param_dist = param_grids["no_search"]
        else:
            active_param_dist = param_grids.get(model_type, param_grids["no_search"])

        search_cv = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=active_param_dist,
            n_iter=4, # Edited to match possible Linear/Polynomial combinations
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        return TransformedTargetRegressor(
            regressor=search_cv,
            transformer=PowerTransformer(method='yeo-johnson')
        )

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
            current_param_dist = params[2] if len(params) > 2 else "no_search"

            custom_feats = params[3] if len(params) > 3 else None

            # Unpack params (model_type, encoder_type)
            model = self.build_pipeline(model_type=params[0], encoder_type=params[1], custom_num=custom_feats, param_dist=current_param_dist)
            current_X_train = X_train[custom_feats + self.cat_features] if custom_feats else X_train
            current_X_test = X_test[custom_feats + self.cat_features] if custom_feats else X_test
            #model.fit(current_X_train, y_train)
            if "RandomForest" in name:
                max_gens = 2
                for gen in range(max_gens):
                    # ALWAYS fit the 'model' (the wrapper), not the internal regressor
                    model.fit(current_X_train, y_train)
            
                    # Now reach in ONLY to check results and update the grid for NEXT time
                    search_cv = model.regressor_ 
                    best_params = search_cv.best_params_
                    current_grid = search_cv.param_distributions
            
                    if best_params['regressor__max_depth'] == max(current_grid['regressor__max_depth']):
                        new_depths = [d + 10 for d in current_grid['regressor__max_depth']]
                        # Update the grid INSIDE the fitted object for the next loop iteration
                        model.regressor.param_distributions['regressor__max_depth'] = new_depths
                        print(f"-> Gen {gen} Success. Shifting Up...")
                    else:
                        break 
            else:
                model.fit(current_X_train, y_train)
            score = model.score(current_X_test, y_test)
            results.update({len(results) + 1 : {"name" : name, "score": score, "model": model}})

            if score > winner_info["score"]:
                winner_info.update({"name": name, "score": score, "model": model})
        
        return winner_info, results

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Class factory instantiation
factory = AQIModelFactory(poly_categoricals, poly_numericals)


# Vif Report
vif_df = factory.vif_fit_transform(X_train)

print("VIF Diagnostic Report:")
print(vif_df.head(10))

def generate_vif_cache(df, output_path="vif.json"):
    """
    Consumes the sorted VIF DataFrame and exports a Top Ranked VIF diagnostic report.
    Format: {"VIF Diagnostic Report": {"feature_name": value, ...}}
    """
    # 1. Take the top 10 rows from your VIF result
    top_vif_df = df.head(50)
    
    # 2. Convert to the specific dictionary format
    # Assuming your vif_df has columns 'feature' and 'VIF'
    report_data = {
        row["feature"]: round(float(row["VIF"]), 6) 
        for row in top_vif_df.to_dicts()
    }
    
    # 3. Wrap in the requested header
    final_output = {"VIF Diagnostic Report": report_data}
    
    # 4. Save to JSON
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"✅ VIF Diagnostic Report cached to {output_path}")


generate_vif_cache(vif_df)


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
print(f"AQI Skewness: {skew(y_train, nan_policy='omit'):.2f}")

pt_y = PowerTransformer(method='yeo-johnson')
y_train_transformed = pt_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_transformed = pt_y.transform(y_test.values.reshape(-1, 1))
print(f"The Lambda calculated from y_train is: {pt_y.lambdas_[0]}")
print(skew(y_train_transformed))

# Features are scouted early to allow permutation features
# into a later list of models with parameters
perm_numeric_features = factory.scout_features(X_train, y_train, X_test, y_test)
print(perm_numeric_features)

# List of models to be run
model_candidates = {
    "Polynomial_OneHot": ("Polynomial", "onehot"),
    "Polynomial_Ordinal": ("Polynomial", "ordinal"),
    "Linear_Ordinal": ("Linear", "ordinal"),
    #"Poly_Pinpointed_Ordinal": ("Polynomial", "ordinal", perm_numeric_features),
    "Polynomial_OneHot_Math": ("Polynomial", "onehot"),
    "Polynomial_Ordinal_Math": ("Polynomial", "ordinal"),
    "Linear_Ordinal_Math": ("Linear", "ordinal"),
    "Polynomial_OneHot_Final": ("Polynomial", "onehot"),
    "Polynomial_OneHot_Filly0s": ("Polynomial", "onehot"),
    "Polynomial_OneHot_MeanFill": ("Polynomial", "onehot"),
    "Polynomial_OneHot_0_Pollutant": ("Polynomial", "onehot"),
    "Polynomial_OneHot_All0": ("Polynomial", "onehot"),
    "Polynomial_OneHot_Duration_Based": ("Polynomial", "onehot"),
    "Polynomial_OneHot_Duration_Features": ("Polynomial", "onehot", perm_numeric_features),
    "Polynomial_OneHot_Duration_Correction": ("Polynomial", "onehot"),
    "Polynomial_OneHot_State_Dist_Added": ("Polynomial", "onehot"),
    "Polynomial_OneHot_Dist_Lat_Long": ("Polynomial", "onehot"),
    "Polynomial_OneHot_PCA_Added": ("Polynomial", "onehot"),
    "Polynomial_OneHot_Spatial_PCA": ("Polynomial", "onehot"),
    "Polynomial_OneHot_PCA_NoDist": ("Polynomial", "onehot"),
    "Polynomial_OneHot_PCA_NoDist_State": ("Polynomial", "onehot"),
    "Polynomial_OneHot_sklearn2_grouped": ("Polynomial", "onehot"),
    "Polynomial_Target_sklearn2_grouped": ("Polynomial", "target"),
    "Polynomial_Target_sklearn2_ungrouped": ("Polynomial", "target"),
    "Polynomial_Target_sklearn2_ungrouped_nointeraction": ("Polynomial", "target"),
    "Polynomial_Target_sklearn2_ungrouped_RandomForest": ("RandomForest", "target"),
    #"Polynomial_OneHot_sklearn2_ungrouped_RandomForest": ("RandomForest", "target"),
    "Polynomial_OneHot_sklearn2_Polynomial_post_forest": ("Polynomial", "target"),
    "Polynomial_OneHotredo_sklearn2_ungrouped_RandomForest": ("RandomForest", "onehot"),
    #"Polynomial_Target_sklearn2_RandomForest_RandomCV": ("RandomForest", "target"),
    "RandomCV_Test2": ("RandomForest", "target"),
    "RandomCV_No_State_AQI_Mean2": ("RandomForest", "target"),
    "RandomCV_No_SpatialTrf": ("RandomForest", "target"),
    "RandomCV_NoStateMeanNoSpatial": ("RandomForest", "target"),
    "RandomCV_DirectionalLoopTest1": ("RandomForest", "target"),
    "Directional_Loop_OneHot": ("RandomForest", "onehot"),
    "Directional_Loop_Target": ("RandomForest", "target"),
    "RF_Directional2": ("RandomForest", "target", "RandomForest"),
    #"Poly_Pinpointed_Ordinal_Math": ("Polynomial", "ordinal", perm_numeric_features),
    #"Poly_Pinpointed_OneHot_Math": ("Polynomial", "onehot", perm_numeric_features)
}

# Load the "Rules"
config = load_system_config("model_control.json")
allowed_list = config.get("include_models", [])
completed = config.get("completed_models", [])

# Filter the dictionary: Only include models that are in the JSON 'allowed' list
active_candidates = {
    name: params for name, params in model_candidates.items()
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
    
    # Time tester - No active end
    print(f"Time taken to no candidate end: {time.time() - start:.4f} seconds")
    sys.exit(0) # Stop here so we don't try to run the Factory


# Finding best model
winner, all_scores = factory.find_best_model(X_train, y_train, X_test, y_test, active_candidates)

for scored_model in all_scores:
    # Break to prevent too much overhead
    time.sleep(2)
    # Conditional if score beats threshold. The SMAPE will simply default to 100.00 if below threshold.
    if all_scores[scored_model]["score"] >= config.get("min_r2_threshold", 0.5):
        # Instatiate the "true y" and "y prediction"
        y_true = np.array(y_test)
        #y_pred = route_predictions(all_scores[scored_model]["model"], X_test)
        y_pred = all_scores[scored_model]["model"].predict(X_test)
        # Instantiate the file path and verify it exists
        file_path = "model_performance_ledger.csv"
        file_exists = os.path.isfile(file_path)
        
        # Calculation for SMAPE (returns a percentage 0-100)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        # Avoid true zero division by adding a tiny epsilon
        diff = np.abs(y_true - y_pred) / (denominator + 1e-10)
        smape_score = 100 * np.mean(diff)

        run_id = f"{all_scores[scored_model]['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create the Ledger Entry
        # This captures the 'Health' of the model rather than raw data points
        performance_entry = pd.DataFrame([{
            'Run_ID': run_id,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Model_Name': all_scores[scored_model]["name"],
            'R2_Score': all_scores[scored_model]["score"],
            'SMAPE_Score': round(smape_score, 4),
            'Sample_Size': len(y_test)
        }])
        print("Test")
        if record_model == True:
            performance_entry.to_csv(file_path, mode='a', index=False, header=not file_exists)
            print(f"📈 Performance Logged: {all_scores[scored_model]['name']} | SMAPE: {smape_score:.2f}%")
            model_obj = all_scores[scored_model]["model"]
            # Condition: Is it wrapped in a Target Transformer?
            if hasattr(model_obj, 'regressor_'):
                inner_layer = model_obj.regressor_
            else:
                inner_layer = model_obj
            # Condition: Is the inner layer a SearchCV (Random/Grid)?
            if hasattr(inner_layer, 'best_estimator_'):
                fitted_pipeline = inner_layer.best_estimator_
            else:
                fitted_pipeline = inner_layer
            # Now call named_steps
            steps = fitted_pipeline.named_steps
            regressor_step = steps.get('regressor')
            #regressor_step = model_obj.named_steps.get('regressor', None)
            # Extract the Progression
            progression = []
            if hasattr(inner_layer, 'cv_results_'):
                results = inner_layer.cv_results_
                # Sort results by the best score (Rank 1 first)
                for i in range(len(results['params'])):
                    progression.append({
                        "rank": int(results['rank_test_score'][i]),
                        "params": results['params'][i],
                        "mean_mae": abs(float(results['mean_test_score'][i])) # MAE is negative in sklearn
                    })
                # Sort the list so Rank 1 is at the top
                progression = sorted(progression, key=lambda x: x['rank'])

            metadata = {
                "run_id": run_id,
                "model_class": str(type(regressor_step)),
                #"is_search_cv": hasattr(regressor_step, 'best_params_'),
                "is_search_cv": hasattr(inner_layer, 'cv_results_'),
                "best_hyperparameters": regressor_step.get_params() if not hasattr(regressor_step, 'best_params_') else regressor_step.best_params_,
                "pipeline_steps": list(list(steps.keys())),
                "search_progression": progression,
                "metrics": {
                    "smape": smape_score,
                    "r2": all_scores[scored_model]["score"]
                }
            }
            # Append to JSON Ledger
            json_ledger_path = "model_config_ledger.json"
            all_metadata = {}
            if os.path.exists(json_ledger_path):
                with open(json_ledger_path, 'r') as f:
                    all_metadata = json.load(f)
            
            all_metadata[run_id] = metadata
            
            with open(json_ledger_path, 'w') as f:
                json.dump(all_metadata, f, indent=4)
            sync_config_with_scores(all_scores[scored_model], config, smape_score)
        else:
            print(f"📈 Result Output with no Performance Log: {all_scores[scored_model]['name']} | SMAPE: {smape_score:.2f}%")
    else:
        print(f"R Squared score doesn't beat minimum.")
        print(all_scores[scored_model]["score"])
        # 0 SMAPE default
        if record_model == True:
            sync_config_with_scores(all_scores[scored_model], config)


# Time tester - Total end
total_time = f"Time taken: {time.time() - start:.4f} seconds"
print(total_time)

if record_time == True:
    time_path = "time_log.csv"
    time_exists = os.path.isfile(time_path)
    time_entry = pd.DataFrame([{
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Entry': "Directional_Test2",
                'Duration': total_time
            }])
    time_entry.to_csv(time_path, mode='a', index=False, header=not time_exists)
