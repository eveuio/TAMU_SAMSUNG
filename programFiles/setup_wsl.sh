#!/bin/bash
# Setup script for WSL Ubuntu environment

echo "Setting up Transformer Health Monitoring System for WSL Ubuntu"
echo "=============================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Install required packages
echo ""
echo "Installing required packages..."
pip3 install -r requirements_wsl.txt

if [ $? -eq 0 ]; then
    echo "✅ Packages installed successfully"
else
    echo "❌ Error installing packages"
    exit 1
fi

# Create data directory
echo ""
echo "Creating data directory..."
mkdir -p ../data
echo "✅ Data directory created"

# Note: Test data creation removed - use real Subsystem 1 data

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Ensure Subsystem 1 data is available in ../data/my_database.db"
echo "2. Run: python3 main.py"
echo "3. Check the output and logs"
echo ""
echo "The system will:"
echo "- Connect to Subsystem 1's database"
echo "- Run health assessments on all transformers"
echo "- Generate lifetime forecasts (when lifetime data is available)"
echo "- Create forecast plots and save results"
echo ""
echo "Ready to run the transformer monitoring system!"
