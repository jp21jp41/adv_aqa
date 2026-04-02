import polars as pl
import os

def fix_column_order(file_path="model_performance_ledger.csv"):
    if not os.path.exists(file_path):
        return

    # 1. Load the current data
    df = pl.read_csv(file_path)

    # 2. Check if Run_ID exists and is not already first
    cols = df.columns
    if "Run_ID" in cols and cols[0] != "Run_ID":
        print(f"🔄 Reordering columns for {file_path}...")
        
        # Move Run_ID to the front
        remaining_cols = [c for c in cols if c != "Run_ID"]
        new_order = ["Run_ID"] + remaining_cols
        
        # Apply the new order
        df = df.select(new_order)
        
        # 3. Overwrite the CSV with the correct format
        df.write_csv(file_path)
        print("✅ Columns reordered: 'Run_ID' is now the first column.")
    else:
        print("ℹ️ Column order is already correct or Run_ID is missing.")

if __name__ == "__main__":
    fix_column_order()