# -*- coding: utf-8 -*-
"""
Transformer Health Monitoring System - Core Health Monitor Class
Contains the main TransformerHealthMonitor class with all health monitoring functionality
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from database import Database
from forecast_engine import TransformerForecastEngine

logger = logging.getLogger(__name__)

WEIGHTS = {
    "Secondary Voltage-A-phase (V)": 0.6, "Secondary Voltage-B-phase (V)": 0.6, "Secondary Voltage-C-phase (V)": 0.6,
    "Secondary Current-A-phase(A)": 0.65, "Secondary Current-B-phase(A)": 0.65, "Secondary Current-C-phase(A)": 0.65,
    "PF%": 0.4,
    "VTHD-A-B": 0.3, "VTHD-B-C": 0.3, "VTHD-A-C": 0.3,
    "Winding-Temp-A(°C)": 0.9, "Winding-Temp-B(°C)": 0.9, "Winding-Temp-C(°C)": 0.9,
}
COLOR_SCORES = {"Green": 1.0, "Yellow": 0.5, "Red": 0.2}

# Mapping from Subsystem 1's column names to your subsystem's variable names
SUBSYSTEM1_COLUMN_MAP = {
    "avg_secondary_voltage_a_phase": "Secondary Voltage-A-phase (V)",
    "avg_secondary_voltage_b_phase": "Secondary Voltage-B-phase (V)",
    "avg_secondary_voltage_c_phase": "Secondary Voltage-C-phase (V)",
    "avg_secondary_current_a_phase": "Secondary Current-A-phase(A)",
    "avg_secondary_current_b_phase": "Secondary Current-B-phase(A)",
    "avg_secondary_current_c_phase": "Secondary Current-C-phase(A)",
    "avg_power_factor": "PF%",
    "avg_vTHD_a_phase": "VTHD-A-B", # Note: Assuming a mapping, this might need adjustment
    "avg_vTHD_b_phase": "VTHD-B-C", # Note: Assuming a mapping, this might need adjustment
    "avg_vTHD_c_phase": "VTHD-A-C", # Note: Assuming a mapping, this might need adjustment
    "avg_winding_temp_a_phase": "Winding-Temp-A(°C)",
    "avg_winding_temp_b_phase": "Winding-Temp-B(°C)",
    "avg_winding_temp_c_phase": "Winding-Temp-C(°C)",
}

class TransformerHealthMonitor:
    """
    Main class for transformer health monitoring system.
    Integrates all functionality into a single, WSL-compatible system.
    """

    #!Fixed
    def __init__(self, database:Database):
        """Initialize the health monitoring system with WSL-compatible paths."""
        # Initialize forecast engine
        self.db = database
        self.forecast_engine = TransformerForecastEngine()
        logger.info("Transformer Health Monitor initialized")
    
    #! Fixed
    def run_health_assessments(self):
        """Run health assessments for all transformers."""
        print(f"\n3. Running health assessments...")
        
        health_results = {}

        transformerMasterTable = f'''transformers'''
        query_transformerName = f'''SELECT  transformer_name FROM "{transformerMasterTable}"'''
    
        df_transformerNames = pd.read_sql_query(query_transformerName, self.db.conn)
        transformer_names = df_transformerNames.iloc[:, 0].tolist()

        for transformer_name in transformer_names:
            try:
                result = self.calculate_health_score(transformer_name)
                if result:
                    health_results[transformer_name] = result
            except Exception as e:
                logger.error(f"Error processing {transformer_name}: {e}")
                print(f"❌ Error processing {transformer_name}: {e}")
        
        # Print summary
        # self._print_overall_summary(health_results)
        
        return health_results
    
    def calculate_health_score(self, transformer_name):
        """
        Performs the health assessment calculations for a single transformer.
        Enhanced version with better error handling and recommendations.
        """
        # logger.info(f"Running health analysis for '{transformer_name}'...")

        try:
            #TODO: Fetch Rated secondary current, secondary voltage, rated winding temp rise, THD (if exists), PF (if exists)
            rated_specs = self.db.get_rated_specs(transformer_name)
            # if not rated_specs:
            #     # logger.warning(f"No rated specs found for '{transformer_name}'. Skipping.")
            #     return None



            latest_averages = self.db.get_latest_averages(transformer_name)
            if not latest_averages:
                # logger.warning(f"No average data found from Subsystem 1 for '{transformer_name}'. Skipping.")
                return None

            # 2. Perform the calculations
            results = {}
            weighted_sum, weight_total = 0, 0
            critical_issues = []
            warnings = []

            for col_name, avg_value in latest_averages.items():
                if col_name in SUBSYSTEM1_COLUMN_MAP:
                    variable_name = SUBSYSTEM1_COLUMN_MAP[col_name] #! Change to master table mapping
                    if variable_name in rated_specs and avg_value is not None:
                        rated = rated_specs[variable_name]
                        if rated == 0:
                            continue

                        # Calculate deviation percentage
                        diff_ratio = abs(avg_value - rated) / rated

                        # Enhanced status determination
                        if diff_ratio <= 0.05:
                            status = "Green"
                        elif diff_ratio <= 0.10:
                            status = "Yellow"
                            warnings.append(f"{variable_name}: {diff_ratio*100:.1f}% deviation")
                        else:
                            status = "Red"
                            critical_issues.append(f"{variable_name}: {diff_ratio*100:.1f}% deviation")

                        # Get weight and calculate score
                        weight = WEIGHTS.get(variable_name, 1)
                        score = COLOR_SCORES[status]
                        weighted_sum += score * weight
                        weight_total += weight

                        results[variable_name] = {
                            "Average": avg_value,
                            "Rated": rated,
                            "Status": status,
                            "Deviation_Percent": diff_ratio * 100,
                            "Weight": weight,
                            "Score": score
                        }

            if not results:
                # logger.warning("No matching variables found. Cannot calculate score.")
                return None

            # Calculate overall score
            overall_score = weighted_sum / weight_total if weight_total > 0 else 0
            overall_color = "Green" if overall_score >= 0.79 else "Yellow" if overall_score >= 0.49 else "Red"

            # 3. Save the results
            self.db.save_health_results(transformer_name, results, overall_score, overall_color)

            # 4. Enhanced display
            # self._print_health_summary(transformer_name, results, overall_score, overall_color, critical_issues, warnings)

            return
            # return {
            #     'transformer_name': transformer_name,
            #     'overall_score': overall_score,
            #     'overall_color': overall_color,
            #     'critical_issues': critical_issues,
            #     'warnings': warnings
            # }

        except Exception as e:
            logger.error(f"Error calculating health score for {transformer_name}: {e}")
            print(f"❌ Error analyzing {transformer_name}: {e}")
            return None
    
    def _print_health_summary(self, transformer_name, results, overall_score, overall_color, critical_issues, warnings):
        """Print detailed health assessment summary for a single transformer."""
        print(f"\n{'='*60}")
        print(f"HEALTH ASSESSMENT: {transformer_name}")
        print(f"{'='*60}")
        print(f"Overall Score: {overall_score:.2f}")
        print(f"Overall Status: {overall_color}")
        print(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nVariable Assessment:")
        print(f"{'Variable':<40} {'Status':<8} {'Deviation':<12} {'Score'}")
        print(f"{'-'*70}")
        
        for variable, data in results.items():
            print(f"{variable:<40} {data['Status']:<8} {data['Deviation_Percent']:>8.1f}% {data['Score']:>8.2f}")
        
        if critical_issues:
            print(f"\n🚨 Critical Issues:")
            for issue in critical_issues:
                print(f"  - {issue}")
        
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        recommendations = self._generate_recommendations(overall_score, critical_issues, warnings)
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    def _generate_recommendations(self, overall_score, critical_issues, warnings):
        """Generate recommendations based on health assessment."""
        recommendations = []
        
        if overall_score < 0.5:
            recommendations.append("IMMEDIATE ACTION REQUIRED: Address critical issues listed above")
            if any("Winding-Temp" in issue for issue in critical_issues):
                recommendations.append("Consider load reduction or cooling system inspection due to high winding temperatures")
        elif overall_score < 0.8:
            recommendations.append("Monitor closely and address issues before they become critical")
        
        if critical_issues or warnings:
            recommendations.append("Review loading patterns and consider load balancing")
        
        if overall_score >= 0.8:
            recommendations.append("Continue current maintenance practices")
        else:
            recommendations.append("Schedule comprehensive maintenance inspection")
        
        return recommendations
    
    def _print_overall_summary(self, health_results):
        """Print overall summary of all health assessments."""
        if not health_results:
            print("No health results to summarize.")
            return
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Calculate summary statistics
        scores = [result['overall_score'] for result in health_results.values()]
        colors = [result['overall_color'] for result in health_results.values()]
        
        avg_score = sum(scores) / len(scores)
        green_count = colors.count('Green')
        yellow_count = colors.count('Yellow')
        red_count = colors.count('Red')
        
        print(f"Health Assessment Results:")
        print(f"  - Average Health Score: {avg_score:.2f}")
        print(f"  - Green Status: {green_count}")
        print(f"  - Yellow Status: {yellow_count}")
        print(f"  - Red Status: {red_count}")
        
        # Collect all critical issues
        all_critical_issues = []
        for result in health_results.values():
            all_critical_issues.extend(result.get('critical_issues', []))
        
        if all_critical_issues:
            print(f"\n🚨 Critical Issues Found:")
            for issue in all_critical_issues:
                print(f"  - {issue}")
    
    def run_lifetime_forecasting(self, transformer_names, method='ensemble'):
        """
        Run lifetime forecasting for all transformers.
        """
        print(f"\n🔮 Running {method} lifetime forecasting...")
        print("=" * 60)
        
        forecast_results = {}
        
        for transformer_name in transformer_names:
            try:
                # Get lifetime data from database
                lifetime_data = self.db.get_transformer_lifetime_data(transformer_name)
                
                if lifetime_data.empty:
                    print(f"⚠️  No lifetime data available for {transformer_name}")
                    continue
                
                # Get latest health score for this transformer
                health_score = self.db.get_latest_health_score(transformer_name)
                
                # Run forecasting with health score integration
                forecast_result = self.forecast_engine.forecast_transformer_lifetime(
                    transformer_name, lifetime_data, health_score, method
                )
                
                if forecast_result:
                    forecast_results[transformer_name] = forecast_result
                    
                    # Save forecast results to database
                    forecast_df = self.forecast_engine.create_forecast_dataframe(forecast_result)
                    if not forecast_df.empty:
                        self.db.save_forecast_results(transformer_name, forecast_df)
                
            except Exception as e:
                logger.error(f"Error forecasting {transformer_name}: {e}")
                print(f"❌ Error forecasting {transformer_name}: {e}")
        
        return forecast_results
    
    def plot_forecast_results(self, forecast_results, save_plots=True):
        """
        Create and save forecast plots for all transformers with years on x-axis.
        """
        if not forecast_results:
            print("⚠️  No forecast results to plot")
            return
        
        print(f"\n📊 Creating forecast plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Transformer Lifetime Forecasting Results', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        for i, (transformer_name, result) in enumerate(forecast_results.items()):
            if i >= 4:  # Limit to 4 plots
                break
                
            ax = axes[i]
            
            # Convert days to years for x-axis
            current_year = datetime.now().year
            forecast_years = current_year + (result['forecast_days'] / 365.25)
            
            # Plot historical data (if available)
            if 'individual_results' in result:
                # For ensemble, plot the exponential decay as reference
                exp_result = result['individual_results'].get('exponential', {})
                if exp_result:
                    exp_years = current_year + (exp_result['forecast_days'] / 365.25)
                    ax.plot(exp_years, exp_result['forecast_values'], 
                           'b--', alpha=0.7, label='Historical Lifetime Data')
            
            # Plot forecast
            ax.plot(forecast_years, result['forecast_values'], 
                   'r-', linewidth=2, label=f"{result['model_name']} Forecast")
            
            # Add 20% cutoff line
            ax.axhline(y=20, color='orange', linestyle=':', linewidth=2, 
                      label='20% Replacement Threshold')
            
            # Highlight cutoff date
            if result['cutoff_day'] is not None:
                cutoff_year = current_year + (result['cutoff_day'] / 365.25)
                ax.axvline(x=cutoff_year, color='red', linestyle='--', 
                          alpha=0.7, label=f'Replacement Date')
            
            # Formatting
            ax.set_title(f'{transformer_name}\nR² = {result["r2_score"]:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Lifetime Percentage (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 100)
            
            # Set x-axis to show years with 5-year intervals
            min_year = int(forecast_years.min())
            max_year = int(forecast_years.max()) + 1
            year_ticks = range(min_year, max_year, 5)
            ax.set_xticks(year_ticks)
            ax.set_xlim(min_year, max_year)
            
            # Add text box with key info
            remaining_years = result.get('remaining_life_years', 'N/A')
            health_score = result.get('health_score', 'N/A')
            info_text = f'Data Points: {result["data_points"]}\n'
            if remaining_years:
                info_text += f'Remaining Life: {remaining_years:.1f} years\n'
            if health_score != 'N/A':
                info_text += f'Health Score: {health_score:.2f}'
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(forecast_results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plot_filename = f"transformer_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📈 Forecast plots saved as: {plot_filename}")
        
        plt.show()
    
    def print_forecast_summary(self, forecast_results):
        """
        Print a summary of forecast results.
        """
        if not forecast_results:
            print("No forecast results to summarize.")
            return
        
        print(f"\n{'='*60}")
        print(f"FORECAST SUMMARY")
        print(f"{'='*60}")
        
        for transformer_name, result in forecast_results.items():
            remaining_years = result.get('remaining_life_years', 'N/A')
            action_required = "YES" if remaining_years and remaining_years < 30 else "NO"
            
            print(f"\n🔮 {transformer_name}:")
            print(f"   Model: {result['model_name']}")
            print(f"   R² Score: {result['r2_score']:.3f}")
            print(f"   Data Points: {result['data_points']}")
            
            if remaining_years:
                print(f"   📅 20% Replacement Date: {remaining_years:.1f} years")
            else:
                print(f"   📅 20% Replacement Date: >30 years")
            
            print(f"   {'⚠️' if action_required == 'YES' else '✅'} Action Required: {action_required}")
    
    def run_health_monitoring(self):
        """
        Main method to run the complete health monitoring process.
        """
        try:
            print("Transformer Health Monitoring System")
            print("=" * 60)
            
            # 3. Get transformer names
            transformer_names = self.db.get_transformer_names()
            if not transformer_names:
                print("❌ No transformers found. Please check your database.")
                return False
            
            print(f"Found {len(transformer_names)} transformers: {transformer_names}")
            
            # 4. Run health assessments
            health_results = self.run_health_assessments(transformer_names)
            
            # 5. Run lifetime forecasting
            print(f"\n4. Running lifetime forecasting...")
            forecast_results = self.run_lifetime_forecasting(transformer_names, method='ensemble')
            
            if forecast_results:
                self.print_forecast_summary(forecast_results)
                self.plot_forecast_results(forecast_results, save_plots=True)
            
            print(f"\n✅ Health monitoring and forecasting completed successfully!")
            print(f"📊 Detailed logs saved to: transformer_monitoring.log")
            return True
            
        except Exception as e:
            logger.error(f"Error in health monitoring: {e}")
            print(f"💥 System encountered issues. Check the logs for details.")
            return False
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'db'):
            self.db.close()
        logger.info("Transformer Health Monitor closed")
