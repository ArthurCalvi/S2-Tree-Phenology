#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Random Forest on embedding features for species classification."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training._train_rf_embeddings_multiclass import run


if __name__ == '__main__':
    run(
        target_column='species',
        target_name='Species',
        default_tag='species',
        log_filename='train_rf_embeddings_species.log'
    )
