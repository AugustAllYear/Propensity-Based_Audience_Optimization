# Full CLI Script to Run pipeline

```bash
#!/bin/bash
# Full pipeline runner for Propensity‑Based Audience Optimization

set -e  # exit on error

echo "Starting Propensity Optimization Pipeline"

# Try to activate pyenv virtualenv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if command -v pyenv &> /dev/null; then
        # Attempt to activate a known environment (adjust name)
        env_name="Propensity_venv"   # change to your actual env name
        if pyenv versions --bare | grep -q "^Propensity_venv$"; then
            echo "Activating pyenv virtualenv: Propensity_venv"
            eval "$(pyenv virtualenv-init -)"
            pyenv activate "Propensity_venv"
        else
            echo "Pyenv environment 'Propensity_venv' not found. Please activate manually."
            exit 1
        fi
    else
        echo "No virtual environment active and pyenv not found. Please activate manually."
        exit 1
    fi
fi

# Activate virtual environment (adjust path if needed)
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Run setup first."
    exit 1
fi

# Train models
echo "Training models..."
python -m src.train

# Evaluate and run simulation
echo "Evaluating model and running simulation..."
python -m src.evaluate

# Run tests (optional)
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