#!/bin/sh
eval "$(conda shell.bash hook)"
conda env create --file environment.yml -n opend5
conda activate opend5
python scripts/pull_data.py --access
python scripts/make_benchmark.py --full