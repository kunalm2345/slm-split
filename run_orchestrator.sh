#!/bin/bash
# Start the Python orchestrator

source venv/bin/activate
python3 split_inference/python/orchestrator.py --benchmark "$@"
