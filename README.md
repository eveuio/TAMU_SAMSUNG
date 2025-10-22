# Enhanced Transformer Health Monitoring and Forecasting System

## Overview

This enhanced system builds upon your existing `databaseEJ.py` structure to provide comprehensive transformer health monitoring and lifetime forecasting. The system is designed to work with Subsystem 1 (Data Analytics) and provide data to Subsystem 3 (UI) through a SQLite database.

## Key Improvements

### 1. **WSL Compatibility**
- All file paths are now WSL-compatible
- Uses `os.path.join()` for cross-platform path handling
- Removes hardcoded Windows paths

### 2. **Enhanced Database Integration**
- Builds upon existing `databaseEJ.py` structure
- Maintains compatibility with Subsystem 1 table naming conventions
- Adds enhanced tables for better functionality
- Improved error handling and logging

### 3. **Improved Health Monitoring**
- Enhanced scoring algorithm with multiple thresholds
- Better variable weighting system
- Trend analysis capabilities
- Comprehensive recommendations generation

### 4. **Advanced Forecasting**
- Multiple forecasting models (linear, polynomial, exponential, health-adjusted)
- Ensemble forecasting for better accuracy
- Health factor integration
- Key date calculations (warning, critical, end-of-life)

### 5. **Better Code Structure**
- Modular design with separate classes
- Comprehensive logging
- Configuration management
- Error handling

## File Structure

```
src/
├── database_enhanced.py      # Enhanced database manager (builds on databaseEJ.py)
├── main_improved.py          # Main application with enhanced features
├── data_simulator.py         # Creates test data for Subsystem 1
├── setup.py                  # Setup script for easy installation
├── requirements_enhanced.txt # Enhanced requirements
├── config.py                 # Configuration management
└── README.md                 # This file
```

## Installation and Setup

### Option 1: Quick Setup
```bash
cd src
python setup.py
```

### Option 2: Manual Setup
```bash
# Install requirements
pip install -r requirements_enhanced.txt

# Create test data
python data_simulator.py

# Run the system
python main_improved.py
```

## Usage

### Basic Usage
```python
from database_enhanced import DatabaseEnhanced
from main_improved import ImprovedHealthMonitor, ImprovedForecastEngine

# Initialize database
db = DatabaseEnhanced()

# Initialize monitoring systems
health_monitor = ImprovedHealthMonitor(db)
forecast_engine = ImprovedForecastEngine(db)

# Run health assessment
result = health_monitor.calculate_health_score('LTR_A01_Data')

# Run forecasting
forecast = forecast_engine.generate_forecast('LTR_A01_Data')
```

### Running the Complete System
```bash
python main_improved.py
```

## Database Schema

### Subsystem 1 Tables (Created by Data Analytics)
- `{transformer_name}_average_metrics_day` - Daily averaged metrics
- `{transformer_name}_lifetime_continuous_loading` - Lifetime calculations

### Subsystem 2 Tables (Created by this system)
- `TransformerSpecs` - Rated specifications
- `HealthScores` - Individual variable health scores
- `HealthSummary` - Overall health summaries
- `ForecastData` - Forecasting results
- `ForecastSummary` - Forecast summaries

## Configuration

The system uses `config.py` for easy parameter adjustment:

```python
# Health monitoring thresholds
HEALTH_CONFIG = {
    'thresholds': {
        'excellent': 0.02,  # 2% deviation
        'good': 0.05,       # 5% deviation
        'warning': 0.10,    # 10% deviation
        'critical': 0.20    # 20% deviation
    }
}

# Forecasting parameters
FORECAST_CONFIG = {
    'forecast_horizon_years': 5,
    'warning_threshold': 20,  # 20% lifetime remaining
    'critical_threshold': 10  # 10% lifetime remaining
}
```

## Health Monitoring Features

### 1. **Multi-Level Scoring**
- **Excellent**: ≤2% deviation from rated values
- **Good**: ≤5% deviation
- **Warning**: ≤10% deviation
- **Critical**: >10% deviation

### 2. **Weighted Assessment**
- Temperature measurements: Highest weight (1.0)
- Current measurements: High weight (0.9)
- Voltage measurements: Medium weight (0.8)
- Power factor: Medium weight (0.6)
- THD measurements: Lower weight (0.4)

### 3. **Trend Analysis**
- Analyzes health trends over time
- Provides trend-based recommendations
- Identifies deteriorating conditions

### 4. **Recommendations Engine**
- Generates actionable recommendations
- Prioritizes critical issues
- Provides maintenance guidance

## Forecasting Features

### 1. **Multiple Models**
- **Linear**: Simple trend-based forecasting
- **Polynomial**: Non-linear trend analysis
- **Exponential**: Decay-based modeling
- **Health-Adjusted**: Incorporates health scores

### 2. **Ensemble Forecasting**
- Combines multiple models for better accuracy
- Weighted averaging based on model performance
- Confidence interval calculations

### 3. **Key Date Calculations**
- **Warning Threshold**: 20% lifetime remaining
- **Critical Threshold**: 10% lifetime remaining
- **End of Life**: 0% lifetime remaining

### 4. **Health Integration**
- Health scores influence forecast accuracy
- Poor health accelerates predicted degradation
- Maintenance recommendations based on forecasts

## Integration with Subsystems

### Subsystem 1 (Data Analytics)
- Reads from `{transformer_name}_average_metrics_day` tables
- Reads from `{transformer_name}_lifetime_continuous_loading` tables
- Maintains existing table structure and naming

### Subsystem 3 (UI)
- Provides data through `HealthScores` and `ForecastData` tables
- Includes summary tables for easy UI integration
- Structured data format for dashboard display

## Testing

The system includes a data simulator that creates realistic test data:

```python
from data_simulator import create_test_data

# Create test data for 3 transformers with different health conditions
create_test_data('path/to/database.db')
```

## Logging

Comprehensive logging is available:
- Console output for real-time monitoring
- File logging in `transformer_monitoring.log`
- Different log levels (INFO, WARNING, ERROR)

## Samsung Transformer Specifications

The system includes specifications for Samsung LTR transformers:
- **LTR_A01_Data**: 500 kVA, 20 years old
- **LTR_A12_Data**: 1000 kVA, 20 years old  
- **LTR_B01_Data**: 1200 kVA, 20 years old

## Troubleshooting

### Common Issues

1. **Subsystem 1 Data Not Found**
   - Run `python data_simulator.py` to create test data
   - Check database path configuration

2. **Database Connection Errors**
   - Ensure database directory exists
   - Check file permissions

3. **Import Errors**
   - Install requirements: `pip install -r requirements_enhanced.txt`
   - Check Python path configuration

### Debug Mode
Enable debug logging by modifying the logging level in the main files:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Machine Learning Models**
   - LSTM for time series forecasting
   - Random Forest for health classification
   - Anomaly detection algorithms

2. **Real-time Monitoring**
   - WebSocket integration
   - Real-time alerts
   - Dashboard updates

3. **Advanced Analytics**
   - Correlation analysis
   - Predictive maintenance
   - Cost optimization

## Support

For issues or questions:
1. Check the logs in `transformer_monitoring.log`
2. Review the configuration in `config.py`
3. Test with the data simulator first
4. Ensure all requirements are installed

## License

This project is part of the Samsung Capstone project for transformer health monitoring and forecasting.
