#!/usr/bin/env python3
"""
Entry point for Kaggle AAP Agent submission.
This script runs the autonomous agent in the Kaggle sandbox.
"""

import sys
import os

# Add submission directory to path
sys.path.insert(0, '/kaggle/working/submission')

# Run the agent
from skills.model_training.scripts.train_automl import main

if __name__ == '__main__':
    main()