# -*- coding: utf-8 -*-
"""
Transformer Health Monitoring System - Main Entry Point
WSL Compatible version that integrates all functionality
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from databaseEJ import Database, SUBSYSTEM1_COLUMN_MAP, WEIGHTS, COLOR_SCORES
from forecast_engine import TransformerForecastEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transformer_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TransformerHealthMonitor:
    """
    Main class for transformer health monitoring system.
    Integrates all functionality into a single, WSL-compatible system.
    """
    
    def __init__(self, db_path: str = None):
        """Initialize the health monitoring system with WSL-compatible paths."""
        if db_path is None:
            # WSL-compatible default path
            self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'my_database.db'))
        else:
            self.db_path = db_path
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self.db = Database(self.db_path)
        
        # Initialize forecasting engine
        self.forecast_engine = TransformerForecastEngine()
        
        logger.info("Transformer Health Monitor initialized")
    
    def calculate_health_score(self, transformer_name):
        """
        Performs the health assessment calculations for a single transformer.
        Enhanced version with better error handling and recommendations.
        """
        logger.info(f"Running health analysis for '{transformer_name}'...")
        
        try:
            # 1. Fetch data from the database
            rated_specs = self.db.get_rated_specs(transformer_name)
            if not rated_specs:
                logger.warning(f"No rated specs found for '{transformer_name}'. Skipping.")
                return None

            latest_averages = self.db.get_latest_averages(transformer_name)
            if not latest_averages:
                logger.warning(f"No average data found from Subsystem 1 for '{transformer_name}'. Skipping.")
                return None

            # 2. Perform the calculations
            results = {}
            weighted_sum, weight_total = 0, 0
            critical_issues = []
            warnings = []

            for col_name, avg_value in latest_averages.items():
                if col_name in SUBSYSTEM1_COLUMN_MAP:
                    variable_name = SUBSYSTEM1_COLUMN_MAP[col_name]
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
                logger.warning("No matching variables found. Cannot calculate score.")
                return None

            # Calculate overall score
            overall_score = weighted_sum / weight_total if weight_total > 0 else 0
            overall_color = "Green" if overall_score >= 0.79 else "Yellow" if overall_score >= 0.49 else "Red"

            # 3. Save the results
            self.db.save_health_results(transformer_name, results, overall_score, overall_color)
            
            # 4. Enhanced display
            self._print_health_summary(transformer_name, results, overall_score, overall_color, critical_issues, warnings)
            
            return {
                'transformer_name': transformer_name,
                'overall_score': overall_score,
                'overall_color': overall_color,
                'critical_issues': critical_issues,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Error calculating health score for {transformer_name}: {e}")
            print(f"❌ Error analyzing {transformer_name}: {e}")
            return None
    
    def _print_health_summary(self, transformer_name, results, overall_score, overall_color, critical_issues, warnings):
        """Print detailed health assessment summary."""
        print(f"\n{'='*60}")
        print(f"HEALTH ASSESSMENT: {transformer_name}")
        print(f"{'='*60}")
        print(f"Overall Score: {overall_score:.2f}")
        print(f"Overall Status: {overall_color}")
        print(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nVariable Assessment:")
        print(f"{'Variable':<40} {'Status':<8} {'Deviation':<12} {'Score'}")
        print("-" * 70)
        
        for var, vals in results.items():
            print(f"{var:<40} {vals['Status']:<8} {vals['Deviation_Percent']:>8.1f}%     {vals['Score']:.2f}")
        
        if critical_issues:
            print(f"\n🚨 Critical Issues:")
            for issue in critical_issues:
                print(f"  - {issue}")
        
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_score, critical_issues, warnings)
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
    
    def _generate_recommendations(self, overall_score, critical_issues, warnings):
        """Generate actionable recommendations based on health assessment."""
        recommendations = []
        
        # Critical issues recommendations
        if critical_issues:
            recommendations.append("IMMEDIATE ACTION REQUIRED: Address critical issues listed above")
        
        # Temperature-related recommendations
        temp_issues = [issue for issue in critical_issues if "Winding-Temp" in issue]
        if temp_issues:
            recommendations.append("Consider load reduction or cooling system inspection due to high winding temperatures")
        
        # Current-related recommendations
        current_issues = [issue for issue in critical_issues if "Current" in issue]
        if current_issues:
            recommendations.append("Review loading patterns and consider load balancing")
        
        # Overall health recommendations
        if overall_score < 0.5:
            recommendations.append("Schedule comprehensive maintenance inspection")
        elif overall_score < 0.7:
            recommendations.append("Schedule routine maintenance within 30 days")
        else:
            recommendations.append("Continue current maintenance practices")
        
        return recommendations
    
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
            return
        
        print(f"\n" + "=" * 60)
        print("FORECAST SUMMARY")
        print("=" * 60)
        
        for transformer_name, result in forecast_results.items():
            print(f"\n🔮 {transformer_name}:")
            print(f"   Model: {result['model_name']}")
            print(f"   R² Score: {result['r2_score']:.3f}")
            print(f"   Data Points: {result['data_points']}")
            
            if result['remaining_life_years']:
                print(f"   📅 20% Replacement Date: {result['remaining_life_years']:.1f} years")
                print(f"   ⚠️  Action Required: YES")
            else:
                print(f"   📅 20% Replacement Date: >30 years")
                print(f"   ✅ Action Required: NO")
    
    def run_health_monitoring(self):
        """
        Main workflow for transformer health monitoring.
        Executes the complete Subsystem 2 workflow.
        """
        print("Transformer Health Monitoring System")
        print("=" * 60)
        
        try:
            # Test database connection first
            print("\n1. Testing database connection...")
            if not self.db.print_connection_status():
                print("\n❌ Database connection test failed. Please check the issues above.")
                print("💡 Try running: python3 create_test_data.py")
                return False
            
            # Initialize schema and seed data
            print("\n2. Initializing database schema...")
            self.db.initialize_schema()
            self.db.seed_transformer_specs()
            
            # Get available transformers
            transformer_names = self.db.get_transformer_names()
            if not transformer_names:
                print("No transformer data from Subsystem 1 found.")
                print("💡 Try running: python3 create_test_data.py")
                return False
            
            print(f"Found {len(transformer_names)} transformers: {transformer_names}")
            
            # Run health assessments
            print(f"\n3. Running health assessments...")
            health_results = {}
            for transformer_name in transformer_names:
                result = self.calculate_health_score(transformer_name)
                if result:
                    health_results[transformer_name] = result
            
            # Print overall summary
            if health_results:
                self._print_overall_summary(health_results)
            
            # Run lifetime forecasting
            print(f"\n4. Running lifetime forecasting...")
            forecast_results = self.run_lifetime_forecasting(transformer_names, method='ensemble')
            
            if forecast_results:
                self.print_forecast_summary(forecast_results)
                self.plot_forecast_results(forecast_results, save_plots=True)
            
            print(f"\n✅ Health monitoring and forecasting completed successfully!")
            print(f"📊 Detailed logs saved to: transformer_monitoring.log")
            return True
            
        except Exception as e:
            logger.error(f"Error in health monitoring workflow: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def _print_overall_summary(self, health_results):
        """Print overall summary of all health assessments."""
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Calculate summary statistics
        scores = [result['overall_score'] for result in health_results.values()]
        colors = [result['overall_color'] for result in health_results.values()]
        
        avg_score = sum(scores) / len(scores) if scores else 0
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

def main():
    """Main function - single entry point for the transformer monitoring system."""
    # Initialize the health monitor
    health_monitor = TransformerHealthMonitor()
    
    try:
        # Run the complete health monitoring workflow
        success = health_monitor.run_health_monitoring()
        
        if success:
            print("\n🎉 System completed successfully!")
        else:
            print("\n💥 System encountered issues. Check the logs for details.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  System interrupted by user.")
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        health_monitor.db.close()

if __name__ == "__main__":
    main()
