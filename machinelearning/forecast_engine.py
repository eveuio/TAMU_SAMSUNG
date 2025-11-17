# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

class TransformerForecastEngine:
    """
    Advanced forecasting engine for transformer lifetime prediction.
    Fully aligned with the DataProcessing Database class.
    """

    def __init__(self, database=None):
        """
        Args:
            database: Database instance from DataProcessing.programFiles.database
        """
        self.db = database
        self.models = {}
        self.forecast_results = {}

    # ======================================================================
    #                         DATA PREPARATION
    # ======================================================================

    def prepare_data(self, lifetime_data):
        """Clean and prepare transient lifetime data for forecasting."""
        if lifetime_data.empty:
            print("No lifetime data provided to prepare_data()")
            return None

        required = {"DATETIME", "Lifetime_Percentage"}
        if not required.issubset(lifetime_data.columns):
            print("ERROR: Lifetime data missing required columns:", required)
            return None

        df = lifetime_data.copy()
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
        df = df.dropna(subset=["DATETIME", "Lifetime_Percentage"])

        # Convert date → numeric axis (days since start)
        start_date = df["DATETIME"].min()
        df["days_since_start"] = (df["DATETIME"] - start_date).dt.days

        # Remove bad rows
        df = df[df["Lifetime_Percentage"] > 0]
        if len(df) < 3:
            print("Insufficient lifetime data (<3 points).")
            return None

        return df

    # ======================================================================
    #                 HEALTH SCORE ADJUSTMENT FOR DEGRADATION
    # ======================================================================

    def apply_health_score_adjustment(self, df, health_score):
        """Adjust degradation rate based on health score."""
        if health_score is None:
            return df

        if health_score >= 0.8:
            factor = 0.001  # very slow degradation
        elif health_score >= 0.5:
            factor = 1.0    # normal degradation
        else:
            factor = 50.0   # extremely fast degradation

        df = df.copy()
        df["days_since_start"] *= factor

        print(f"Applied health degradation factor: {factor:.3f} for health={health_score:.2f}")
        return df

    # ======================================================================
    #                             FORECAST MODELS
    # ======================================================================

    def linear_regression_forecast(self, df, forecast_days=365*50):
        X = df[["days_since_start"]].values
        y = df["Lifetime_Percentage"].values

        model = LinearRegression()
        model.fit(X, y)

        last_day = df["days_since_start"].max()
        future_days = np.arange(last_day, last_day + forecast_days, 30).reshape(-1, 1)
        preds = model.predict(future_days)

        cutoff_idx = np.where(preds <= 20)[0]
        cutoff = future_days[cutoff_idx[0]][0] if len(cutoff_idx) > 0 else None

        return {
            "model": model,
            "forecast_days": future_days.flatten(),
            "forecast_values": preds,
            "cutoff_day": cutoff,
            "r2_score": model.score(X, y),
            "model_name": "Linear Regression"
        }

    def exponential_decay_forecast(self, df, forecast_days=365*50):
        X = df[["days_since_start"]].values
        y = df["Lifetime_Percentage"].values

        y_log = np.log(np.maximum(y, 0.1))

        model = LinearRegression()
        model.fit(X, y_log)

        last_day = df["days_since_start"].max()
        future_days = np.arange(last_day, last_day + forecast_days, 30).reshape(-1, 1)
        preds = np.exp(model.predict(future_days))

        cutoff_idx = np.where(preds <= 20)[0]
        cutoff = future_days[cutoff_idx[0]][0] if len(cutoff_idx) > 0 else None

        return {
            "model": model,
            "forecast_days": future_days.flatten(),
            "forecast_values": preds,
            "cutoff_day": cutoff,
            "r2_score": model.score(X, y_log),
            "model_name": "Exponential Decay"
        }

    def polynomial_forecast(self, df, degree=2, forecast_days=365*50):
        X = df[["days_since_start"]].values
        y = df["Lifetime_Percentage"].values

        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        last_day = df["days_since_start"].max()
        future_days = np.arange(last_day, last_day + forecast_days, 30).reshape(-1, 1)
        preds = model.predict(poly.transform(future_days))

        preds = np.maximum(preds, 0)

        cutoff_idx = np.where(preds <= 20)[0]
        cutoff = future_days[cutoff_idx[0]][0] if len(cutoff_idx) > 0 else None

        return {
            "model": model,
            "poly_features": poly,
            "forecast_days": future_days.flatten(),
            "forecast_values": preds,
            "cutoff_day": cutoff,
            "r2_score": model.score(X_poly, y),
            "model_name": f"Polynomial (degree {degree})"
        }

    def ensemble_forecast(self, df, forecast_days=365*30):
        linear = self.linear_regression_forecast(df, forecast_days)
        expdecay = self.exponential_decay_forecast(df, forecast_days)
        poly = self.polynomial_forecast(df, degree=2, forecast_days=forecast_days)

        models = [linear, expdecay, poly]

        # Weighted by R²
        r2s = np.array([m["r2_score"] for m in models])
        weights = r2s / r2s.sum() if r2s.sum() > 0 else np.ones_like(r2s) / len(r2s)

        all_days = linear["forecast_days"]
        ensemble_vals = (
            weights[0] * linear["forecast_values"]
            + weights[1] * expdecay["forecast_values"]
            + weights[2] * poly["forecast_values"]
        )

        cutoff_idx = np.where(ensemble_vals <= 20)[0]
        cutoff = all_days[cutoff_idx[0]] if len(cutoff_idx) > 0 else None

        return {
            "forecast_days": all_days,
            "forecast_values": ensemble_vals,
            "cutoff_day": cutoff,
            "r2_score": np.mean(r2s),
            "individual_results": {
                "linear": linear,
                "exponential": expdecay,
                "polynomial": poly
            },
            "model_name": "Ensemble"
        }

    # ======================================================================
    #                         DATABASE ACCESS (FIXED)
    # ======================================================================

    def get_latest_health_score(self, transformer_name):
        """Pull health score from your partner's HealthScores table."""
        try:
            q = """SELECT overall_score 
                   FROM HealthScores 
                   WHERE transformer_name = ?
                   ORDER BY date DESC LIMIT 1"""
            row = self.db.cursor.execute(q, (transformer_name,)).fetchone()
            if row:
                return float(row[0])
        except:
            pass
        return 0.5

    # ======================================================================
    #                     MAIN FORECAST PIPELINE
    # ======================================================================

    def forecast_transformer_lifetime(self, transformer_name, lifetime_data=None, health_score=None, method="ensemble"):
        print(f"\n--- Running {method} forecast for {transformer_name} ---")

        # Fetch lifetime data from your partner's DB (IMPORTANT FIX)
        if lifetime_data is None:
            lifetime_data = self.db.get_transformer_lifetime_data(transformer_name)
            if lifetime_data.empty:
                print("ERROR: No lifetime data found.")
                return None

        if health_score is None:
            health_score = self.get_latest_health_score(transformer_name)

        original = lifetime_data.copy()
        original["DATETIME"] = pd.to_datetime(original["DATETIME"])

        df = self.prepare_data(lifetime_data)
        if df is None:
            return None

        df = self.apply_health_score_adjustment(df, health_score)

        # Select forecast model
        if method == "linear":
            result = self.linear_regression_forecast(df)
        elif method == "exponential":
            result = self.exponential_decay_forecast(df)
        elif method == "polynomial":
            result = self.polynomial_forecast(df)
        else:
            result = self.ensemble_forecast(df)

        # Calculate remaining lifetime
        start_date = original["DATETIME"].min()
        last_day = df["days_since_start"].max()

        if result["cutoff_day"] is not None:
            remaining_days = result["cutoff_day"] - last_day
            result["remaining_life_years"] = remaining_days / 365.25
        else:
            result["remaining_life_years"] = None

        # Package metadata
        result["transformer_name"] = transformer_name
        result["forecast_date"] = datetime.now().strftime("%Y-%m-%d")
        result["start_date"] = start_date
        result["last_data_day"] = last_day

        # Save forecast to DB
        df_out = self.create_forecast_dataframe(result)
        if not df_out.empty:
            self.save_forecast_results(transformer_name, df_out)

        return result

    # ======================================================================
    #                        RESULT → DATAFRAME → DATABASE
    # ======================================================================

    def create_forecast_dataframe(self, result):
        """Convert forecast result into DB-ready DataFrame."""
        try:
            days = result["forecast_days"]
            vals = result["forecast_values"]
            transformer = result["transformer_name"]
            start_date = pd.to_datetime(result["start_date"])

            dates = [start_date + timedelta(days=int(d)) for d in days]

            df = pd.DataFrame({
                "transformer_name": [transformer] * len(dates),
                "forecast_date": [d.strftime("%Y-%m-%d") for d in dates],
                "predicted_lifetime": vals
            })

            print(f"Forecast DataFrame created with {len(df)} rows")
            return df

        except Exception as e:
            print("ERROR creating forecast dataframe:", e)
            return pd.DataFrame()

    def save_forecast_results(self, transformer_name, df):
        """Write forecast dataframe to ForecastData table."""
        try:
            self.db.cursor.execute(
                "DELETE FROM ForecastData WHERE transformer_name = ?",
                (transformer_name,)
            )
            df.to_sql("ForecastData", self.db.conn, if_exists="append", index=False)
            self.db.conn.commit()
            print(f"Saved forecast results for {transformer_name}")
        except Exception as e:
            print("DB SAVE ERROR:", e)
            self.db.conn.rollback()
