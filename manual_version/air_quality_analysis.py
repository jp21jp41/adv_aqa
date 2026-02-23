# Advanced Air Quality Analysis - Manual Version

# Import Packages
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from scipy.stats import normaltest, skew
import polars as pl
import numpy as np
import pandas as pd

# Selected CSV Folder
combined_df = pl.scan_csv("atmospheric_data/*.csv")

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

# Collect and Display
results = analysis.collect()
print(results)

# Data Frame Selection (Prepared DF) and then Pivot (Wide DF) to have the proper data.
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
# 1. Pivot the data so each Parameter becomes its own column
# This turns your 'narrow' data into 'wide' data suitable for regression
wide_df = (
    combined_df
    .select(["State Name", "Date Local", "Parameter Name", "AQI", "Arithmetic Mean"])
    .collect() # Bring into memory first
    .pivot(
        on="Parameter Name",
        index=["State Name", "Date Local", "AQI"],
        values="Arithmetic Mean",
        aggregate_function="mean"  # <--- This solves the "multiple values" conflict
    )
    .drop_nulls()
)
"""

# Wide DataFrame Display (this is the final of the Polars Data Manipulation)
#print(wide_df.columns)


# Basic Features
categorical_features = ["State Name"]
exclude = ["State Name", "Date Local", "AQI"]
numeric_features = [col for col in wide_df.columns if col not in exclude]
#numeric_features = [col for col in wide_df.columns if col not in categorical_features + ["AQI"]]

# Feature Concatenation
feature_cols = categorical_features + numeric_features

# Assigning an X and y variable to prepare train and test data
X = wide_df.select(feature_cols)
y = wide_df["AQI"]

# Force conversion to numeric, turning errors (like empty strings) into 'NaN'
y = pd.to_numeric(y, errors='coerce')

# Drop any rows that couldn't be converted
y = y[~np.isnan(y)]

# Previous form
#y = wide_df.get_column("AQI").to_numpy()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""
AQI Skewness is printed. Such a statistic would be more relevant
if an automated system required it. However, we already have
a low magnitude of skewness, so for now, it's implicit that
the data used is not going to have any need to be analyzed
based on the skewness.
"""
print(f"AQI Skewness: {skew(y_train):.2f}")

"""
# Previous train test split
X_train, X_test, y_train, y_test = train_test_split(
    wide_df.select(feature_cols), # <--- Clean and simple
    wide_df.get_column("AQI").to_numpy(),
    test_size=0.2, 
    random_state=42
)
"""

# Preprocessor Included
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])


"""
# Previous Polynomial Pipeline
poly_model = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])
"""

# Polynomial Pipeline
poly_model = Pipeline([
    ('prep', preprocessor),
    ('poly', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])

# Original Polynomial Pipeline
#poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Polynomial Fit over train data
poly_model.fit(X_train, y_train)

# Put preprocessor feature names into a variable
input_to_poly = poly_model.named_steps['prep'].get_feature_names_out()

# Another variable for after the polynomial expansion
feature_names = poly_model.named_steps['poly'].get_feature_names_out(input_to_poly)

# Regressor Coefficients
coefficients = poly_model.named_steps['regressor'].coef_


# Create a clean summary
"""
coeff_df = pd.DataFrame({'Feature': feature_names, 'Weight': coefficients})
coeff_df['Abs_Weight'] = coeff_df['Weight'].abs()
coeff_df = coeff_df.sort_values(by='Abs_Weight', ascending=False)
"""

# Top features isn't used, but the previous code block signal doesn't
# have any particular relevance
#print("\n--- TOP COEFFICIENT WEIGHTS ---")
#print(coeff_df[['Feature', 'Weight']].head(10))

# Try-except for passed features
try:
    feature_names = poly_model.named_steps['prep'].feature_names_in_
except:
    # Fallback: if the model doesn't have names, the application uses the ones from the user's numeric list
    feature_names = list(numeric_features) + ["State Name"]

# Convert X_test back into a DataFrame
X_test_fixed = pd.DataFrame(X_test, columns=feature_names)

# Run the importance using the fixed parameter
perm_importance = permutation_importance(
    poly_model, 
    X_test_fixed,  # <--- Labeled version here
    y_test, 
    n_repeats=5, 
    random_state=42
)

# Other Permutation Importance Code (used when X_test already comes labeled)
#perm_importance = permutation_importance(poly_model, X_test, y_test, n_repeats=5, random_state=42)

# Map Permutation Importance to names
importance_df = pd.DataFrame({
    'Feature': X_test.columns, 
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

# Importance DataFrame Printed
#print(importance_df)

# Keep Important Features that are found from the DataFrame in a list
good_features = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()

# List conversion to exclude State Name
perm_numeric_features = [f for f in good_features if f != "State Name"]

# Permutation Numeric Features Printed
#print(perm_numeric_features)

# Permutation Preprocessor
perm_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(), categorical_features),
        ('num', 'passthrough', perm_numeric_features)
    ])

# Permutation Polynomial Model with Permutation Preprocessor
perm_poly_model = Pipeline([
    ('prep', perm_preprocessor),
    ('poly', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])

# --- Filter the data ---
X_train_final = X_train[good_features]
X_test_final = X_test[good_features]

# --- Re-fit and get the NEW Score ---
perm_poly_model.fit(X_train_final, y_train)
final_score = perm_poly_model.score(X_test_final, y_test)

# --- Score Output ---
print(f"New Score after Automated Pinpointing: {final_score:.4f}")

# Linear Pipeline
linear_pipe = Pipeline([
    ('prep', preprocessor),
    ('regressor', LinearRegression())
])

"""
# Previous Regression Code
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

score_model = model.score(X_test, y_test)
print(f"Linear Model Score: {score_model:.4f}")

print(f"Regression Intercept: {model.intercept_}")
print(f"Coefficients for {feature_cols}: {model.coef_}")
"""

# Linear Pipeline Fit
linear_pipe.fit(X_train, y_train)

# Score, Output, and Prediction for Linear Pipeline
score_linear = linear_pipe.score(X_test, y_test)
print(f"Linear Model Score: {score_linear:.4f}")
linear_predictions = linear_pipe.predict(X_test)

# Polynomial Prediction
poly_predictions = poly_model.predict(X_test)

# Score, Output, and Prediction for Permutation Polynomial Model
score_poly = poly_model.score(X_test, y_test)
print(f"Poly Score: {score_poly:.4f}")
perm_poly_predictions = perm_poly_model.predict(X_test)

# Linear Pipeline Coefficients and Intercept
coeffs = linear_pipe.named_steps['regressor'].coef_
intercept = linear_pipe.named_steps['regressor'].intercept_

# Regression Intercept, Features, and Coefficients
print(f"Regression Intercept: {intercept}")
print(f"Coefficients for {feature_cols}: {coeffs}")

# Basic Conditional to Determine the Winning Model
if score_poly > score_linear:
    best_model = poly_model
    winner_name = "Polynomial"
else:
    best_model = linear_pipe
    winner_name = "Linear"

"""
R^2 winner output.
Originally, the highest-performing model was
the model to be used for final predictions.
The code had been adjusted, but we are still
left with this output.
"""
#print(f"The {winner_name} model performed best. Using it for final predictions.")
print(f"The {winner_name} model performed best.")

"""
The original idea was to be able to have automated
prediction of a statistic such as an RMSE. However,
that was before there had been an entire automation
application made.
"""
#rmse = np.sqrt(mean_squared_error(y_test, poly_predictions))

# The AQI Range was going to have a JSON Cache.
# As shown, it was previously used in calculations.
#aqi_range = float(y.max()) - float(y.min())

"""
nrmse = (rmse / aqi_range) * 100
print(f"Standard RMSE: {rmse:.2f} AQI points")
print(f"Normalized RMSE (Scale of 100): {nrmse:.2f}%")

rmse1 = np.sqrt(mean_squared_error(y_test, poly_predictions))
nrmse1 = (rmse1 / aqi_range) * 100
print(f"Standard RMSE: {rmse1:.2f} AQI points")
print(f"Normalized RMSE (Scale of 100): {nrmse1:.2f}%")
"""

# Function that displays the "leaderboard".
# The scores give a list of metrics per
# model, but in search for a score more
# akin to Machine Learning Competitions.
def competition_leaderboard(y_true, y_pred, model_name):
    # Ensure no negative predictions (common in linear models, but breaks log metrics)
    y_pred_clipped = np.clip(y_pred, 0, None)
    
    # Competition Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred_clipped))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*30}")
    print(f" LEADERBOARD: {model_name}")
    print(f"{'='*30}")
    print(f" PRIMARY METRIC (RMSLE): {rmsle:.5f}") # Lower is better
    print(f" RMSE:                   {rmse:.5f}")
    print(f" MAE:                    {mae:.5f}")
    print(f" R2 SCORE:               {r2:.5f}")
    print(f"{'='*30}\n")

# These can still be used to compare models.
competition_leaderboard(y_test, poly_predictions, "POLYNOMIAL_REGRESSION_V1")
competition_leaderboard(y_test, linear_predictions, "LINEAR_BASE_MODEL")
competition_leaderboard(y_test, perm_poly_predictions, "POLYNOMIAL_REGRESSION_V2")

# The SMAPE score function
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid true zero division by adding a tiny epsilon
    diff = np.abs(y_true - y_pred) / (denominator + 1e-10)
    return 100 * np.mean(diff)


# The y variable is further transformed into a float to run properly
# in the SMAPE, as well as the predictions.
y_test_numeric = y_test.astype(float)
linear_preds_numeric = linear_predictions.astype(float)
poly_preds_numeric = poly_predictions.astype(float)
perm_poly_preds_numeric = perm_poly_predictions.astype(float)

print("SMAPE scores displayed below:")

# Linear SMAPE
value = smape(y_test_numeric, linear_preds_numeric)
print("Linear:")
print(value)
competition_score1 = max(0, 100 * (1 - value))

# Polynomial SMAPE
value = smape(y_test_numeric, poly_preds_numeric)
print("Polynomial:")
print(value)
competition_score2 = max(0, 100 * (1 - value))

# Permutation Polynomial SMAPE
value = smape(y_test_numeric, perm_poly_preds_numeric)
print("Permutation Polynomial:")
print(value)
competition_score3 = max(0, 100 * (1 - value))
