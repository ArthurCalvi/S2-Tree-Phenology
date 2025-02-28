#!/usr/bin/env python3
"""
Runner script for the IRLS iteration comparison test.
This script will execute the test_compare_irls_iterations function from the tests module.
"""

import sys
from pathlib import Path
from qa.compare_irls_iterations import test_compare_irls_iterations

if __name__ == "__main__":
    print("Starting IRLS comparison test...")
    test_compare_irls_iterations()
    print("Test completed. Check the results directory for the PDF report.") 