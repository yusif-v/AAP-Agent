#!/bin/bash
# CI test script for AAP-Agent submission
# Runs all tests and validates the submission pipeline
set -e

echo "=== AAP-Agent CI Tests ==="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "1. Generating test data..."
python3 generate_test_data.py
echo ""

echo "2. Running unit tests..."
python3 -m pytest test_train_automl.py -v 2>/dev/null || python3 test_train_automl.py
echo ""

echo "3. Running local evaluation (ensemble)..."
python3 run_local_eval.py --experiment ensemble
echo ""

echo "4. Validating submission format..."
python3 validate_submission.py test_data/final_submission.csv --test-csv test_data/test.csv
echo ""

echo "=== All tests passed ==="
