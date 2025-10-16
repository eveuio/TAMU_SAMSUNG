# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:04:11 2025
@author: bigal
This script runs the second part of Subsystem 2: forecasting remaining transformer life.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from databaseEJ import Database

# Configuration
WEEKS_TO_FORECAST = 52 * 5 # 5 years
WARNING_THRESHOLD = 20

def simulate_decay(starting_life, health_score):
    """A simplified decay model based on health score."""
    decay_curve = np.linspace(0, 1, WEEKS_TO_FORECAST)
    decay_rate = 1 + (0.9 - health_score) * 5
    curve = starting_life * np.exp(-decay_rate * decay_curve)
    noise = np.random.normal(0, 0.4, size=curve.shape)
    return np.clip(curve + noise, 0, 100)

def train_and_forecast(transformer_name, db_manager):
    """
    Fetches lifetime data, gets a health score, and forecasts the remaining life.
    """
    try:
        # 1. Fetch historical lifetime data from Subsystem 1's table
        df = db_manager.get_transformer_lifetime_data(transformer_name)
        
        if df.empty or "Lifetime_Percentage" not in df.columns or df["Lifetime_Percentage"].isna().all():
            print(f"[{transformer_name}] No valid lifetime data found. Skipping forecast.")
            return

        df.dropna(subset=["Lifetime_Percentage", "DATETIME"], inplace=True)
        
        # 2. Get the latest health score calculated by this subsystem
        health_score = db_manager.get_latest_health_score(transformer_name)

        # 3. Forecasting Logic
        latest_life = df["Lifetime_Percentage"].iloc[-1]
        start_date = df["DATETIME"].max()
        forecast_dates = [start_date + timedelta(weeks=i) for i in range(WEEKS_TO_FORECAST)]
        forecasted_life = simulate_decay(latest_life, health_score)

        # 4. Save the forecast results to the database
        forecast_df = pd.DataFrame({
            'transformer_name': transformer_name,
            'forecast_date': forecast_dates,
            'predicted_lifetime': forecasted_life
        })
        db_manager.save_forecast_results(transformer_name, forecast_df)

        # 5. Find End-of-Life Date for plotting
        eol_date = next((d for d, l in zip(forecast_dates, forecasted_life) if l <= WARNING_THRESHOLD), None)

        # 6. Plot the results
        plt.figure(figsize=(14, 6))
        plt.plot(df["DATETIME"], df["Lifetime_Percentage"], label="Historical Data", color="blue", linewidth=2)
        plt.scatter(forecast_dates, forecasted_life, s=5, color="purple", label=f"Forecast (Health Score = {health_score:.2f})")
        plt.axhline(y=WARNING_THRESHOLD, color='orange', linestyle=':', label=f"{WARNING_THRESHOLD}% Lifetime Warning")
        if eol_date:
            plt.axvline(eol_date, linestyle="--", color="black", label=f"Forecasted End-of-Life: {eol_date.date()}")

        plt.title(f"{transformer_name} — Remaining Life Forecast")
        plt.xlabel("Date")
        plt.ylabel("Lifetime %")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[ Error processing forecast for {transformer_name}]: {e}")

if __name__ == "__main__":
    DB_PATH = r"C:/Users/bigal/Capstone-alex/data/my_database.db"
    db = Database(db_path=DB_PATH)

    # Get transformers that have calculated health data
    transformer_names = db.get_transformer_names()
    print(f"Running forecasts for transformers: {transformer_names}")
    
    for name in transformer_names:
        train_and_forecast(name, db)
        
    db.close()
