# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:06:13 2025
@author: bigal
This is the main entry point for the Subsystem 2 workflow.
It calculates health scores based on pre-processed data from Subsystem 1.
"""
import pandas as pd
from databaseEJ import Database, SUBSYSTEM1_COLUMN_MAP, WEIGHTS, COLOR_SCORES

def calculate_health_score(transformer_name, db_manager):
    """
    Performs the health assessment calculations for a single transformer.
    """
    print(f"\nRunning Health Analysis for '{transformer_name}'...")
    
    # 1. Fetch data from the database
    rated_specs = db_manager.get_rated_specs(transformer_name)
    if not rated_specs:
        print(f"[Warning] No rated specs found for '{transformer_name}'. Skipping.")
        return

    latest_averages = db_manager.get_latest_averages(transformer_name)
    if not latest_averages:
        print(f"No average data found from Subsystem 1 for '{transformer_name}'. Skipping.")
        return

    # 2. Perform the calculations
    results = {}
    weighted_sum, weight_total = 0, 0

    for col_name, avg_value in zip(latest_averages.keys(), latest_averages):
        if col_name in SUBSYSTEM1_COLUMN_MAP:
            variable_name = SUBSYSTEM1_COLUMN_MAP[col_name]
            if variable_name in rated_specs and avg_value is not None:
                rated = rated_specs[variable_name]
                if rated == 0: continue
                
                diff_ratio = abs(avg_value - rated) / rated
                status = "Green" if diff_ratio <= 0.05 else "Yellow" if diff_ratio <= 0.10 else "Red"
                
                weight = WEIGHTS.get(variable_name, 1)
                score = COLOR_SCORES[status]
                weighted_sum += score * weight
                weight_total += weight
                
                results[variable_name] = {"Average": avg_value, "Rated": rated, "Status": status}

    if not results:
        print("No matching variables found. Cannot calculate score.")
        return

    overall_score = weighted_sum / weight_total if weight_total > 0 else 0
    overall_color = "Green" if overall_score >= 0.79 else "Yellow" if overall_score >= 0.49 else "Red"

    # 3. Save the results
    db_manager.save_health_results(transformer_name, results, overall_score, overall_color)
    
    # 4. Display the results
    print(f"'{transformer_name}' -> Health Score: {overall_score:.2f} ({overall_color})")
    print(pd.DataFrame(results).T)

def main_workflow():
    """
    Executes the Subsystem 2 workflow: health analysis.
    """
    DB_PATH = "C:/Users/bigal/Capstone-alex/data/my_database.db"
    db_manager = Database(db_path=DB_PATH)
    
    print("--- Step 1: Initializing & Seeding Subsystem 2 ---")
    db_manager.initialize_schema()
    db_manager.seed_transformer_specs()
    
    print("\n--- Step 2: Running Health Analysis for All Transformers ---")
    transformer_names = db_manager.get_transformer_names()
    
    if not transformer_names:
        print("No transformer data from Subsystem 1 found. Exiting.")
    else:
        print(f"Found transformer data for: {transformer_names}")
        for name in transformer_names:
            calculate_health_score(name, db_manager)
    
    print("\n--- Subsystem 2 Workflow Complete ---")
    db_manager.close()

if __name__ == "__main__":
    main_workflow()  