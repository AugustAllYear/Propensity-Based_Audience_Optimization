#!/bin/bash
# Full pipeline runner for Propensity-Based Audience Optimization
# Prerequisite: Activate your virtual environment before running (e.g., `pyenv activate Propensity_venv`)

set -e  # exit on error

echo "Starting Propensity Optimization Pipeline"

# Verify environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Please activate your environment first."
    echo "Example: pyenv activate Propensity_venv"
    exit 1
fi

# Train models
echo "Training models..."
python -m src.train

# Evaluate and run simulation
echo "Evaluating model and running simulation..."
python -m src.evaluate

# Run tests
echo "Running tests..."
pytest tests/ -v

# Start MLflow UI in background
echo "Starting MLflow UI at http://localhost:5000"
mlflow ui --backend-store-uri ./mlruns --port 5000 &
MLFLOW_PID=$!

# Start FastAPI in background
echo "Starting FastAPI at http://localhost:8000"
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start Streamlit dashboard
echo "Starting Streamlit dashboard at http://localhost:8501"
streamlit run app.py --server.port 8501

# Cleanup on exit
trap "kill $MLFLOW_PID $API_PID" EXIT