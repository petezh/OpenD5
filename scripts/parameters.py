"""
Relevant parameters for constructions.
"""

from datetime import datetime
from os.path import join

# folder names
DOWNLOAD_FOLDER = "downloads"
DATASET_FOLDER = "datasets"
MANUAL_FOLDER = "manual"
OUTPUT_FOLDER = "output"
UNLABELED_FOLDER = "unlabeled"
SCHEMA_FOLDER = "schema"
PAIRS_SCHEMA = "pairs.yaml"
DATASET_SCHEMA = "datasets.yaml"
BENCHMARK_FOLDER = "benchmarks"
COMPONENTS_FOLDER = "components"
BENCHMARK_NAME = f'benchmark_{datetime.now().strftime("%m%d")}.pkl'
BENCHMARK_FILE = join(BENCHMARK_FOLDER, BENCHMARK_NAME)

SNIPPET_MAXLEN = 256

CLUSTER_DATASETS = ["all_the_news", "wikitext", "debate", "poetry"]

# pair structure
PAIR_FIELDS = {
    "dataset",
    "generation",
    "applications",
    "pair_type",
    "pair",
    "pos_samples",
    "neg_samples",
    "pos_desc",
    "neg_desc",
    "flip",
}

APPLICATION_FIELDS = {"target", "user", "purely_exploratory"}

# parameters
DISC_THRESH = 0.58
