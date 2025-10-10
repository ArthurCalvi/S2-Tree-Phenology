#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Random Forest on embedding features for genus classification."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training._train_rf_embeddings_multiclass import run


if __name__ == '__main__':
    run(
        target_column='genus',
        target_name='Genus',
        default_tag='genus',
        log_filename='train_rf_embeddings_genus.log'
    )
