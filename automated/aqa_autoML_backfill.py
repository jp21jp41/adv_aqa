import polars as pl
import json
import os
from datetime import datetime
import uuid

def backfill_run_ids(csv_path="model_performance_ledger.csv", json_path="model_config_ledger.json"):
    if not os.path.exists(csv_path):
        print(f"❌ File {csv_path} not found.")
        return

    # 1. Load the existing CSV
    df = pl.read_csv(csv_path)

    # 2. Identify rows missing Run_ID
    # We check if the column exists; if not, we create it.
    if "Run_ID" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("Run_ID"))

    # Filter for rows where Run_ID is null or empty
    missing_mask = df["Run_ID"].is_null()
    
    if not missing_mask.any():
        print("✅ All entries already have a Run_ID.")
        return

    print(f"found {missing_mask.sum()} entries needing a Run_ID. Starting backfill...")

    # 3. Load or initialize the JSON Ledger
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}

    # 4. Generate IDs and Metadata for missing rows
    # We use a list to update the Run_ID column efficiently
    updated_run_ids = list(df["Run_ID"])
    
    for i in range(len(df)):
        if updated_run_ids[i] is None:
            # Create a synthetic Run_ID: [ModelName]_[ShortUUID]_[Date]
            model_name = df[i, "Model_Name"].replace(" ", "_")
            short_id = str(uuid.uuid4())[:8]
            timestamp = df[i, "Timestamp"].replace("-", "").replace(" ", "_").replace(":", "")
            new_id = f"BACKFILL_{model_name}_{timestamp}_{short_id}"
            
            updated_run_ids[i] = new_id

            # Create a "placeholder" metadata entry in the JSON
            # Since we don't have the actual model object anymore, 
            # we store what we know from the CSV.
            all_metadata[new_id] = {
                "run_id": new_id,
                "model_class": "Legacy_Backfill",
                "is_search_cv": "Unknown",
                "hyperparameters": "Metadata not captured in legacy run",
                "metrics": {
                    "smape": df[i, "SMAPE_Score"],
                    "r2": df[i, "R2_Score"]
                }
            }

    # 5. Update the DataFrame and Write back
    df = df.with_columns(pl.Series("Run_ID", updated_run_ids))
    
    # Save CSV
    df.write_csv(csv_path)
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(all_metadata, f, indent=4)

    print(f"🎉 Backfill complete. {missing_mask.sum()} entries synchronized.")

if __name__ == "__main__":
    backfill_run_ids()