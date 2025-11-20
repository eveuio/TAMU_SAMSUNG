# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class TransformerForecastEngine:
    """
    Advanced forecasting engine for transformer lifetime prediction.
    Implements multiple models and ensemble forecasting.
    """

    def __init__(self, database=None):
        """
        Initialize the forecast engine.

        Args:
            database: Optional database object (from DataProcessing.programFiles or database_wrapper)
        """
        self.models = {}
        self.forecast_results = {}
        self.db = database  # expected to have attribute .db_path

    # ------------------------------------------------------------
    # Thread-safe SQLite connection for read and write operations
    # ------------------------------------------------------------
    def _get_safe_connection(self):
        """
        Open a fresh SQLite connection for the current thread.

        Uses the db_path from the passed-in database object.
        """
        import sqlite3
        if not self.db or not hasattr(self.db, "db_path"):
            raise RuntimeError("Database object with db_path is required for DB operations.")
        return sqlite3.connect(self.db.db_path, check_same_thread=False)

    # ------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------
    def prepare_data(self, lifetime_data: pd.DataFrame):
        """
        Prepare lifetime data for forecasting.
        Converts timestamps to numeric features and handles missing values.
        Applies optional scaling for transformer lifetime forecasting.
        """
        if lifetime_data is None or lifetime_data.empty:
            print("prepare_data: empty lifetime_data")
            return None

        # Ensure we have the required columns
        if "DATETIME" not in lifetime_data.columns or "Lifetime_Percentage" not in lifetime_data.columns:
            print("Invalid data format. Required columns: DATETIME, Lifetime_Percentage")
            return None

        # Convert datetime to numeric (days since first measurement)
        df = lifetime_data.copy()
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
        df = df.dropna(subset=["DATETIME", "Lifetime_Percentage"])
        df = df[df["Lifetime_Percentage"] > 0]

        if df.empty:
            print("prepare_data: no valid rows after cleaning")
            return None

        start_date = df["DATETIME"].min()
        df["days_since_start"] = (df["DATETIME"] - start_date).dt.days

        if len(df) < 3:
            print("Insufficient data points for forecasting (need at least 3)")
            return None

        # Optional scaling logic (kept from your original implementation)
        current_lifetime = df["Lifetime_Percentage"].iloc[-1]

        # If the data shows very low remaining life, scale to a more realistic range
        if current_lifetime < 60:
            min_life = df["Lifetime_Percentage"].min()
            max_life = df["Lifetime_Percentage"].max()
            if max_life > min_life:  # avoid division by zero
                realistic_min = 80.0
                realistic_max = 95.0
                scaled = realistic_min + (df["Lifetime_Percentage"] - min_life) * (
                    (realistic_max - realistic_min) / (max_life - min_life)
                )
                df["Lifetime_Percentage"] = scaled
                print(
                    f"Scaled lifetime data to realistic range: "
                    f"{realistic_min:.1f}% - {realistic_max:.1f}%"
                )

        return df

    # ------------------------------------------------------------
    # Health-score-based adjustment
    # ------------------------------------------------------------
    def apply_health_score_adjustment(self, data: pd.DataFrame, health_score: float):
        """
        Apply health score adjustment to lifetime data for more accurate forecasting.
        Health scores affect the degradation rate - lower scores mean faster degradation (shorter remaining life).
        """
        if health_score is None:
            return data

        # Health score ranges: 0-1 (0.8+ = Green, 0.5-0.8 = Yellow, <0.5 = Red)
        # Adjust degradation rate based on health score - BETTER health = SLOWER degradation (more years)
        if health_score >= 0.8:  # Green - slower degradation (longer remaining life)
            degradation_factor = 0.001  # 99.9% slower degradation
        elif health_score >= 0.5:  # Yellow - normal degradation
            degradation_factor = 1.0  # Normal degradation
        else:  # Red - faster degradation (shorter remaining life)
            degradation_factor = 50.0  # 4900% faster degradation

        df = data.copy()
        df["days_since_start"] = df["days_since_start"] * degradation_factor

        print(f"Applied health score adjustment: {health_score:.2f} -> "
              f"{degradation_factor:.1f}x degradation rate")

        return df

    # ------------------------------------------------------------
    # Individual forecasting models
    # ------------------------------------------------------------
    def linear_regression_forecast(self, data: pd.DataFrame, forecast_days: int = 365 * 50):
        """
        Linear regression model for lifetime forecasting.
        """
        X = data[["days_since_start"]].values
        y = data["Lifetime_Percentage"].values

        model = LinearRegression()
        model.fit(X, y)

        # Generate forecast
        last_day = data["days_since_start"].max()
        forecast_days_array = np.arange(last_day, last_day + forecast_days, 30).reshape(-1, 1)
        forecast_values = model.predict(forecast_days_array)

        # Find 20% cutoff date
        cutoff_idx = np.where(forecast_values <= 20)[0]
        cutoff_day = forecast_days_array[cutoff_idx[0]][0] if len(cutoff_idx) > 0 else None

        return {
            "model": model,
            "forecast_days": forecast_days_array.flatten(),
            "forecast_values": forecast_values,
            "cutoff_day": cutoff_day,
            "r2_score": model.score(X, y),
            "model_name": "Linear Regression",
        }

    def exponential_decay_forecast(self, data: pd.DataFrame, forecast_days: int = 365 * 50):
        """
        Exponential decay model - more realistic for transformer aging.
        """
        X = data[["days_since_start"]].values
        y = data["Lifetime_Percentage"].values

        # Transform to linear space for fitting
        # y = a * exp(-b * x) -> ln(y) = ln(a) - b * x
        y_log = np.log(np.maximum(y, 0.1))  # Avoid log(0)

        model = LinearRegression()
        model.fit(X, y_log)

        # Generate forecast
        last_day = data["days_since_start"].max()
        forecast_days_array = np.arange(last_day, last_day + forecast_days, 30).reshape(-1, 1)
        forecast_log = model.predict(forecast_days_array)
        forecast_values = np.exp(forecast_log)

        # Find 20% cutoff date
        cutoff_idx = np.where(forecast_values <= 20)[0]
        cutoff_day = forecast_days_array[cutoff_idx[0]][0] if len(cutoff_idx) > 0 else None

        return {
            "model": model,
            "forecast_days": forecast_days_array.flatten(),
            "forecast_values": forecast_values,
            "cutoff_day": cutoff_day,
            "r2_score": model.score(X, y_log),
            "model_name": "Exponential Decay",
        }

    def polynomial_forecast(
        self, data: pd.DataFrame, degree: int = 2, forecast_days: int = 365 * 50
    ):
        """
        Polynomial regression model for non-linear trends.
        """
        X = data[["days_since_start"]].values
        y = data["Lifetime_Percentage"].values

        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        # Generate forecast
        last_day = data["days_since_start"].max()
        forecast_days_array = np.arange(last_day, last_day + forecast_days, 30).reshape(-1, 1)
        forecast_poly = poly_features.transform(forecast_days_array)
        forecast_values = model.predict(forecast_poly)

        # Ensure non-negative values
        forecast_values = np.maximum(forecast_values, 0)

        # Find 20% cutoff date
        cutoff_idx = np.where(forecast_values <= 20)[0]
        cutoff_day = forecast_days_array[cutoff_idx[0]][0] if len(cutoff_idx) > 0 else None

        return {
            "model": model,
            "poly_features": poly_features,
            "forecast_days": forecast_days_array.flatten(),
            "forecast_values": forecast_values,
            "cutoff_day": cutoff_day,
            "r2_score": model.score(X_poly, y),
            "model_name": f"Polynomial (degree {degree})",
        }

    # ------------------------------------------------------------
    # Ensemble forecast
    # ------------------------------------------------------------
    def ensemble_forecast(self, data: pd.DataFrame, forecast_days: int = 365 * 30):
        """
        Ensemble method combining multiple models with weighted averaging.
        """
        models = ["linear", "exponential", "polynomial"]
        results = {}

        # Get individual model results
        results["linear"] = self.linear_regression_forecast(data, forecast_days)
        results["exponential"] = self.exponential_decay_forecast(data, forecast_days)
        results["polynomial"] = self.polynomial_forecast(
            data, degree=2, forecast_days=forecast_days
        )

        # Weight models by their R² scores
        weights = {}
        total_r2 = 0.0
        for model_name, result in results.items():
            r2 = result.get("r2_score", 0.0)
            if r2 > 0:
                weights[model_name] = r2
                total_r2 += r2

        if total_r2 == 0:
            # Fallback to equal weights
            weights = {name: 1.0 for name in models}
            total_r2 = float(len(models))

        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_r2

        # Create ensemble forecast
        forecast_days_array = results["linear"]["forecast_days"]
        ensemble_values = np.zeros_like(forecast_days_array, dtype=float)

        for model_name, result in results.items():
            r2 = result.get("r2_score", 0.0)
            if r2 > 0:
                ensemble_values += weights[model_name] * result["forecast_values"]

        # Find 20% cutoff date
        cutoff_idx = np.where(ensemble_values <= 20)[0]
        cutoff_day = forecast_days_array[cutoff_idx[0]] if len(cutoff_idx) > 0 else None

        # Calculate average R² score for ensemble
        valid_r2 = [res.get("r2_score", 0.0) for res in results.values() if res.get("r2_score", 0.0) > 0]
        avg_r2 = float(np.mean(valid_r2)) if valid_r2 else 0.0

        return {
            "forecast_days": forecast_days_array,
            "forecast_values": ensemble_values,
            "cutoff_day": cutoff_day,
            "r2_score": avg_r2,
            "model_weights": weights,
            "individual_results": results,
            "model_name": "Ensemble",
        }

    # ------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------
    def get_manufacture_date(self, transformer_name: str):
        """
        Get manufacture_date from transformers table using a safe connection.
        """
        if not self.db:
            return None

        try:
            conn = self._get_safe_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT manufacture_date FROM transformers WHERE transformer_name = ?",
                (transformer_name,),
            )
            row = cur.fetchone()
            conn.close()
            if row and row[0]:
                return row[0]
        except Exception as e:
            print(f"Error getting manufacture_date for {transformer_name}: {e}")
        return None

    def get_transformer_lifetime_data(self, transformer_name: str) -> pd.DataFrame:
        """
        Get lifetime data from <xfmr>_lifetime_transient_loading with remainingLifetime_percent.
        Uses columns:
          - DATETIME
          - remainingLifetime_percent
        """
        if not self.db:
            return pd.DataFrame()

        try:
            # Get manufacture_date (may be used for backfilling)
            manufacture_date = self.get_manufacture_date(transformer_name)

            # Table name and query based on your actual schema
            lifetime_table = f"{transformer_name}_lifetime_transient_loading"
            query = (
                f'SELECT DATETIME, remainingLifetime_percent AS Lifetime_Percentage '
                f'FROM "{lifetime_table}"'
            )

            conn = self._get_safe_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                return pd.DataFrame()

            # Convert DATETIME to proper datetime format
            df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")

            # If manufacture_date is available and DATETIME is missing or invalid, use manufacture_date as reference
            if manufacture_date:
                try:
                    manufacture_date_dt = pd.to_datetime(manufacture_date, errors="coerce")
                    if manufacture_date_dt is not pd.NaT and df["DATETIME"].isna().any():
                        for idx, row in df.iterrows():
                            if pd.isna(row["DATETIME"]):
                                # Use manufacture_date + days based on row index
                                df.at[idx, "DATETIME"] = manufacture_date_dt + pd.Timedelta(days=idx)
                except Exception as e:
                    print(f"Warning: Could not process manufacture_date {manufacture_date}: {e}")

            return df

        except Exception as e:
            print(f"Error getting lifetime data for {transformer_name}: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def get_latest_health_score(self, transformer_name: str) -> float:
        """
        Get latest health score from HealthScores table using a safe connection.
        """
        if not self.db:
            return 0.5
        try:
            conn = self._get_safe_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT overall_score FROM HealthScores "
                "WHERE transformer_name = ? "
                "ORDER BY date DESC LIMIT 1",
                (transformer_name,),
            )
            row = cur.fetchone()
            conn.close()

            if row is not None:
                score = row[0]
                try:
                    return float(score)
                except (TypeError, ValueError):
                    pass
        except Exception as e:
            print(f"Error getting health score for {transformer_name}: {e}")
        return 0.5

    def save_forecast_results(self, transformer_name: str, forecast_df: pd.DataFrame):
        """
        Save forecast results to ForecastData table using a thread-safe connection.
        """
        if not self.db:
            print("Database not initialized. Cannot save forecast results.")
            return

        try:
            # Check if DataFrame is empty
            if forecast_df is None or forecast_df.empty:
                print(f"ERROR: Empty forecast DataFrame for {transformer_name}. Nothing to save.")
                return

            # Check required columns
            required_columns = ["transformer_name", "forecast_date", "predicted_lifetime"]
            missing_columns = [col for col in required_columns if col not in forecast_df.columns]
            if missing_columns:
                print(f"ERROR: Missing required columns in forecast DataFrame: {missing_columns}")
                return

            # Ensure transformer_name is in the DataFrame (it should already be there from create_forecast_dataframe)
            if "transformer_name" not in forecast_df.columns:
                forecast_df["transformer_name"] = transformer_name

            print(f"DEBUG: Saving {len(forecast_df)} forecast records for {transformer_name}")
            print(f"DEBUG: DataFrame columns: {list(forecast_df.columns)}")
            print(f"DEBUG: First 3 rows:\n{forecast_df.head(3)}")

            # Write using a safe connection
            conn = self._get_safe_connection()
            cur = conn.cursor()

            # Delete existing forecast data for this transformer
            cur.execute("DELETE FROM ForecastData WHERE transformer_name = ?", (transformer_name,))
            conn.commit()

            # Append new rows
            forecast_df.to_sql("ForecastData", conn, if_exists="append", index=False)
            conn.commit()

            # Verify the save in the same connection
            cur.execute(
                "SELECT COUNT(*) FROM ForecastData WHERE transformer_name = ?",
                (transformer_name,),
            )
            count = cur.fetchone()[0]
            conn.close()

            print(
                f"SUCCESS: '{transformer_name}' -> Forecast results saved. "
                f"{count} records in database."
            )
        except Exception as e:
            print(f"ERROR saving forecast results for {transformer_name}: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

    # ------------------------------------------------------------
    # Main forecasting entry point
    # ------------------------------------------------------------
    def forecast_transformer_lifetime(
        self,
        transformer_name: str,
        lifetime_data: pd.DataFrame = None,
        health_score: float = None,
        method: str = "ensemble",
    ):
        """
        Main forecasting function for a single transformer with health score integration.

        Args:
            transformer_name: Name of the transformer
            lifetime_data: Optional DataFrame with DATETIME and Lifetime_Percentage columns.
                           If None and database is available, will fetch from database.
            health_score: Optional health score. If None and database is available, will fetch from database.
            method: Forecasting method ('linear', 'exponential', 'polynomial', 'ensemble')
        """
        print(f"Running {method} forecast for {transformer_name}...")

        # Get lifetime data from database if not provided
        if lifetime_data is None:
            if not self.db:
                print("Error: Database not initialized and no lifetime_data provided.")
                return None
            lifetime_data = self.get_transformer_lifetime_data(transformer_name)
            if lifetime_data.empty:
                print(f"No lifetime data available for {transformer_name}")
                return None

        # Get health score from database if not provided
        if health_score is None and self.db:
            health_score = self.get_latest_health_score(transformer_name)

        # Store original lifetime_data for date calculations
        original_lifetime_data = lifetime_data.copy()
        original_lifetime_data["DATETIME"] = pd.to_datetime(
            original_lifetime_data["DATETIME"], errors="coerce"
        )
        original_lifetime_data = original_lifetime_data.dropna(subset=["DATETIME"])

        if original_lifetime_data.empty:
            print("No valid DATETIME entries in lifetime data.")
            return None

        # Prepare data
        data = self.prepare_data(lifetime_data)
        if data is None:
            return None

        # Apply health score adjustment to forecasting
        if health_score is not None:
            data = self.apply_health_score_adjustment(data, health_score)

        # Select forecasting method
        if method == "linear":
            result = self.linear_regression_forecast(data)
        elif method == "exponential":
            result = self.exponential_decay_forecast(data)
        elif method == "polynomial":
            result = self.polynomial_forecast(data)
        elif method == "ensemble":
            result = self.ensemble_forecast(data)
        else:
            print(f"Unknown forecasting method: {method}")
            return None

        # Calculate date reference points for converting forecast_days to actual dates
        # forecast_days are days since start of data, so we need the start date
        start_date = original_lifetime_data["DATETIME"].min()
        last_data_date = original_lifetime_data["DATETIME"].max()
        last_data_day = data["days_since_start"].max()

        # Add metadata
        result["transformer_name"] = transformer_name
        result["data_points"] = len(data)
        result["forecast_date"] = datetime.now().strftime("%Y-%m-%d")
        result["health_score"] = health_score
        result["start_date"] = start_date  # Store for date conversion
        result["last_data_date"] = last_data_date  # Store for date conversion
        result["last_data_day"] = last_data_day  # Store for date conversion

        # Calculate remaining life in years
        if result.get("cutoff_day") is not None:
            remaining_days = result["cutoff_day"] - last_data_day
            result["remaining_life_years"] = remaining_days / 365.25
        else:
            result["remaining_life_years"] = None

        print(f"{method} forecast completed - R² = {result['r2_score']:.3f}")
        if result["remaining_life_years"]:
            print(f"20% cutoff in {result['remaining_life_years']:.1f} years")

        # Save forecast results to database if database is available
        if self.db:
            forecast_df = self.create_forecast_dataframe(result)
            if not forecast_df.empty:
                self.save_forecast_results(transformer_name, forecast_df)

        return result

    # ------------------------------------------------------------
    # Convert forecast results to a DataFrame
    # ------------------------------------------------------------
    def create_forecast_dataframe(self, forecast_result: dict) -> pd.DataFrame:
        """
        Convert forecast result to DataFrame for database storage.
        """
        if not forecast_result:
            print("Warning: Empty forecast_result in create_forecast_dataframe")
            return pd.DataFrame()

        try:
            # Get forecast days and values
            forecast_days = forecast_result.get("forecast_days", [])
            forecast_values = forecast_result.get("forecast_values", [])
            transformer_name = forecast_result.get("transformer_name", "")

            # Debug output
            days_len = len(forecast_days) if hasattr(forecast_days, "__len__") else "N/A"
            values_len = len(forecast_values) if hasattr(forecast_values, "__len__") else "N/A"
            print(f"DEBUG: forecast_days type: {type(forecast_days)}, length: {days_len}")
            print(f"DEBUG: forecast_values type: {type(forecast_values)}, length: {values_len}")
            if days_len != "N/A" and days_len > 0:
                print(
                    f"DEBUG: First few forecast_days: "
                    f"{forecast_days[:5] if len(forecast_days) > 5 else forecast_days}"
                )
            if values_len != "N/A" and values_len > 0:
                print(
                    f"DEBUG: First few forecast_values: "
                    f"{forecast_values[:5] if len(forecast_values) > 5 else forecast_values}"
                )

            # Convert numpy arrays to lists if needed
            if isinstance(forecast_days, np.ndarray):
                forecast_days = forecast_days.tolist()
            if isinstance(forecast_values, np.ndarray):
                forecast_values = forecast_values.tolist()

            if not forecast_days or not forecast_values:
                print("Warning: Missing forecast_days or forecast_values in forecast_result")
                print(f"  forecast_days: {forecast_days}")
                print(f"  forecast_values: {forecast_values}")
                return pd.DataFrame()

            if len(forecast_days) != len(forecast_values):
                print(
                    "Warning: Mismatch between forecast_days "
                    f"({len(forecast_days)}) and forecast_values ({len(forecast_values)})"
                )
                return pd.DataFrame()

            # Create forecast dates - forecast_days are days since start of data
            # We need to convert these to actual calendar dates
            start_date = forecast_result.get("start_date")

            if start_date is None:
                # Fallback: use today as reference (less accurate but works)
                print("Warning: start_date not found in forecast_result, using today as reference")
                start_date = datetime.now()
                # Adjust if we have last_data_day
                last_data_day = forecast_result.get("last_data_day", 0)
                if last_data_day > 0:
                    start_date = datetime.now() - timedelta(days=int(last_data_day))

            # Convert to pandas Timestamp if it's not already
            if not isinstance(start_date, pd.Timestamp):
                start_date = pd.to_datetime(start_date)

            # Convert forecast_days to actual dates
            forecast_dates = []
            for day in forecast_days:
                try:
                    forecast_date = start_date + timedelta(days=int(day))
                    forecast_dates.append(forecast_date)
                except (ValueError, OverflowError) as e:
                    print(f"Warning: Error converting forecast day {day} to date: {e}")
                    continue

            if not forecast_dates:
                print("Warning: No valid forecast dates created")
                return pd.DataFrame()

            # Ensure all arrays are the same length
            num_forecasts = len(forecast_dates)
            min_length = min(num_forecasts, len(forecast_values))
            forecast_dates = forecast_dates[:min_length]
            forecast_values = forecast_values[:min_length]

            df = pd.DataFrame(
                {
                    "transformer_name": [transformer_name] * min_length,
                    "forecast_date": [d.strftime("%Y-%m-%d") for d in forecast_dates],
                    "predicted_lifetime": forecast_values,
                }
            )

            print(f"DEBUG: Created forecast DataFrame with {len(df)} rows for {transformer_name}")
            if len(df) > 0:
                print(
                    "DEBUG: DataFrame sample - First row: "
                    f"transformer_name={df.iloc[0]['transformer_name']}, "
                    f"forecast_date={df.iloc[0]['forecast_date']}, "
                    f"predicted_lifetime={df.iloc[0]['predicted_lifetime']:.2f}"
                )
            return df

        except Exception as e:
            print(f"Error creating forecast DataFrame: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
