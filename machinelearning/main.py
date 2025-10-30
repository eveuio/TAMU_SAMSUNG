# -*- coding: utf-8 -*-
"""
Transformer Health Monitoring System - Main Entry Point
WSL Compatible version that integrates all functionality
"""
import os
import sys
import logging

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_health_monitor import TransformerHealthMonitor

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

def main():
    """
    Main entry point for the Transformer Health Monitoring System.
    """
    health_monitor = None
    
    try:
        # Initialize the health monitoring system
        health_monitor = TransformerHealthMonitor()
        
        # Run the complete health monitoring process
        success = health_monitor.run_health_monitoring()
        
        if success:
            print("\nSystem completed successfully!")
        else:
            print("\nSystem encountered issues. Check the logs for details.")
            
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"\nUnexpected error: {e}")
    finally:
        # Clean up
        if health_monitor:
            health_monitor.close()
        print("Database connection closed.")

if __name__ == "__main__":
    main()