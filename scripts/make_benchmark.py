"""
Constructs and cleans the benchmark from components.

Author: Peter Zhang
"""

import argparse
from collections import defaultdict, Counter
from copy import deepcopy
from itertools import chain, combinations
import json
from os.path import join
import pickle as pkl
import random
from typing import List
import yaml
from yaml.loader import SafeLoader

import numpy as np

from add_pairs import *
from parameters import *
from test_discriminative import discriminated_w_simple_feature, discriminated_w_mnb
from utils import remove_empty, split_truncate_word

with open(join(SCHEMA_FOLDER, DATASET_SCHEMA), 'r') as f:
    dataset_metadata = yaml.load(f, Loader=SafeLoader)

def filter_pairs_for_desc(
    pairs: List[dict],
    threshold: int = DISC_THRESH,
    discriminator: str = 'auc_roc_mnb',
) -> List[dict]:
    """
    Removes pairs from a list that are too easily
    discriminated as judged from a threshold and
    discriminator.
    """

    filtered_pairs = []

    for pair in pairs:
        disc = pair[discriminator]
        if disc < threshold:
            pairs.append(construct_pair(**pair))

    return filtered_pairs


def construct_pair(
    include_stats: bool = True,
    include_roc: bool = True,
    truncate: bool = True,
    **pair_data,
) -> dict:
    """
    Constructs a pair from pair data and optionally adds metadata.
    """
    missing_fields = [
        field for field in PAIR_FIELDS if field not in pair_data.keys()]
    # check for all required fields
    assert not missing_fields, f'Fields ({", ".join(missing_fields)}) missing'

    for i, application in enumerate(pair_data['applications']):
        missing_fields = [
            field for field in APPLICATION_FIELDS if field not in application.keys()]
        assert not missing_fields, f'Fields  ({", ".join(missing_fields)}) missing from application {i}'

    dataset = pair_data['dataset']
    pos_samples = remove_empty(pair_data['pos_samples'])
    neg_samples = remove_empty(pair_data['neg_samples'])

    if truncate:
        pos_weighted_samples = split_truncate_word(pos_samples)
        neg_weighted_samples = split_truncate_word(neg_samples)

        pos_weights = [s[1] for s in pos_weighted_samples]
        neg_weights = [s[1] for s in neg_weighted_samples]
        pos_samples = [s[0] for s in pos_weighted_samples]
        neg_samples = [s[0] for s in neg_weighted_samples]

        pair_data['pos_weights'] = pos_weights
        pair_data['neg_weights'] = neg_weights
        pair_data['pos_samples'] = pos_samples
        pair_data['neg_samples'] = neg_samples
    else:
        pair_data['pos_weights'] = [1] * len(pos_samples)
        pair_data['neg_weights'] = [1] * len(neg_samples)

    pair_data['discipline'] = dataset_metadata[dataset]['discipline']
    if not ('dataset_description' in pair_data and pair_data['dataset_description']):
        pair_data['dataset_description'] = dataset_metadata[dataset]['description']
    else:
        desc = pair_data['dataset_description'].strip()
        pair_data['dataset_description'] = desc[0].lower() + desc[1:]
    pair_data['status'] = dataset_metadata[dataset]['status']
    pair_data['expertise'] = dataset_metadata[dataset]['expertise']
    pair_data['hash'] = hash(tuple(pos_samples + neg_samples))

    if include_stats:
        pair_data['avg_chars'] = float(
            np.mean(list(map(len, pos_samples + neg_samples))))
        pair_data['avg_words'] = float(
            np.mean(list(map(lambda s: len(s.split(' ')), pos_samples + neg_samples))))

    if include_roc:
        auc_roc_simple = discriminated_w_simple_feature(
            pos_samples, neg_samples)
        auc_roc_mnb = discriminated_w_mnb(pos_samples, neg_samples)
        pair_data['auc_roc_simple'] = auc_roc_simple
        pair_data['auc_roc_mnb'] = auc_roc_mnb

    return pair_data


def generate_pair(
    **pair_data
) -> dict:
    """
    Creates pair of distributions and relevant metadata.
    """

    dataset = pair_data['dataset']
    pos_class = pair_data['pos_class']
    neg_class = pair_data['neg_class']
    del pair_data['pos_class']
    del pair_data['neg_class']

    path = f'{OUTPUT_FOLDER}/{dataset}.json'
    output = json.load(open(path, 'r'))
    pos_dists = [output['data'][dist] for dist in pos_class]
    neg_dists = [output['data'][dist] for dist in neg_class]
    pos_samples = list(chain.from_iterable(pos_dists))
    neg_samples = list(chain.from_iterable(neg_dists))

    return construct_pair(
        pos_samples=pos_samples,
        neg_samples=neg_samples,
        **pair_data
    )


def add_spurious_data(path: str) -> List[dict]:
    """
    Adds externally generated spurious pairs.
    """

    data = json.load(open(path))

    pairs = []

    for pair in data:
        pairs.append(construct_pair(**pair))

    return pairs


def add_cluster_data(
    sample_path: str,
    distance_path: str,
    dataset: str,
) -> List[dict]:
    """
    Adds externally generated cluster data.
    """

    pairs = []
    generation = 'which automatically generated "cluster" the snippet is from'
    user = 'a data scientist performing unsupervised clustering'
    target = 'what each cluster represents'
    purely_exploratory = True

    applications = [{
        'user': user,
        'target': target,
        'purely_exploratory': purely_exploratory,
    }]

    sample_data = json.load(open(sample_path, 'r'))
    distance_data = json.load(open(distance_path, 'r'))

    print('1 v all')

    # 1 v all framing
    flip = False

    for pos_id, pos_samples in tqdm(sample_data.items()):

        pair_type = 'cluster_versus_all'

        neg_ids = [id for id in sample_data if id != pos_id]
        neg_samples = list(chain(*[sample_data[id] for id in neg_ids]))

        pair = f'{dataset}_cluster_{pos_id}_v_all'
        pos_desc = 'are from a particular cluster'
        neg_desc = 'are from the rest of the cluster'

        random.seed(0)
        neg_samples = random.sample(neg_samples, k=len(pos_samples))

        p = construct_pair(
            pair=pair,
            dataset=dataset,
            generation=generation,
            applications=applications,
            example_hypotheses=[],
            pos_desc=pos_desc,
            neg_desc=neg_desc,
            pos_samples=pos_samples,
            neg_samples=neg_samples,
            pair_type=pair_type,
            flip=flip,
            note='',
            include_roc=True,
        )

        pairs.append(p)

    clusters = sample_data.keys()  # pairs of clusters, 3% closest distance

    distances = list(chain(*distance_data.values()))
    distances = [d for d in distances if d]  # remove 0

    def get_cutoff(p): return np.percentile(
        distances, p)  # returns the percentile distance

    CLOSEST = 3
    CLOSE = 8
    SOMEWHAT_CLOSE = 15

    def get_bucket(d):
        """Decides how close a given cluster is."""
        if d < get_cutoff(CLOSEST):
            return 'very close'
        if d < get_cutoff(CLOSE):
            return 'close'
        if d < get_cutoff(SOMEWHAT_CLOSE):
            return 'somewhat close'
        return None

    # 1v1 by closeness
    flip = True

    bucketed_combos = defaultdict(list)
    for cluster1, cluster2 in combinations(clusters, 2):
        distance = distance_data[cluster1][int(cluster2)]
        bucketed_combos[get_bucket(distance)].append((cluster1, cluster2))

    SAMPLE_SIZE = 20

    print('closeness')
    for closeness in ('very close', 'close', 'somewhat close'):

        np.random.seed(1)
        np.random.shuffle(bucketed_combos[closeness])
        samples = bucketed_combos[closeness][:SAMPLE_SIZE]

        for cluster1, cluster2 in tqdm(samples):

            pair_type = f'{closeness.replace(" ","_")}_clusters'

            pos_samples = sample_data[cluster1]
            neg_samples = sample_data[cluster2]

            pair = f'{dataset}_cluster_{cluster1}_v_{cluster2}'

            pos_desc = 'are from one cluster'
            neg_desc = f'are from a {closeness} cluster'

            p = construct_pair(
                pair=pair,
                dataset=dataset,
                generation=generation,
                applications=applications,
                example_hypotheses=[],
                pos_desc=pos_desc,
                neg_desc=neg_desc,
                pos_samples=pos_samples,
                neg_samples=neg_samples,
                pair_type=pair_type,
                flip=flip,
                note='',
                include_roc=True,
            )

            pairs.append(p)

    return pairs


def post_processing(benchmark, max_size=55, train_size=.5, remove_private=True):
    """
    Flatten applications, resample overrepresented samples,
    and create train-test splits.
    """

    random.seed(0)

    # flatten applications
    flattened_benchmark = []
    for pair in benchmark:
        if not 'status' in pair:
            print(pair['pair'])
        if remove_private and (pair['status'] == 'private'):
            continue
        for application in pair['applications']:
            new_pair = deepcopy(pair)
            del new_pair['applications']
            new_pair['application'] = application
            flattened_benchmark.append(new_pair)
    benchmark = flattened_benchmark

    # downsample
    by_dataset = defaultdict(list)
    for app in benchmark:
        by_dataset[app['dataset']].append(app['hash'])
    keep_hashes = set()
    for dataset in by_dataset:
        hashes = by_dataset[dataset]
        random.shuffle(hashes)
        keep_hashes.update(hashes[:max_size])

    benchmark = [p for p in benchmark if p['hash'] in keep_hashes]
    print(Counter(p['dataset'] for p in benchmark))

    # train-test split
    for application in benchmark:
        pos_samples, neg_samples = application['pos_samples'], application['neg_samples']

        pos_samples = deepcopy(pos_samples)
        neg_samples = deepcopy(neg_samples)

        random.shuffle(pos_samples)
        random.shuffle(neg_samples)

        train_pos_samples = pos_samples[:int(len(pos_samples) * train_size)]
        train_neg_samples = neg_samples[:int(len(neg_samples) * train_size)]

        test_pos_samples = pos_samples[int(len(pos_samples) * train_size):]
        test_neg_samples = neg_samples[int(len(neg_samples) * train_size):]

        application['split'] = {
            'train': {
                'pos_samples': train_pos_samples,
                'neg_samples': train_neg_samples
            },
            'test': {
                'pos_samples': test_pos_samples,
                'neg_samples': test_neg_samples
            }
        }

    benchmark = [reformat(problem) for problem in benchmark]

    return benchmark

def reformat(example_problem):
    """
    Final formatting for paper.
    """

    example_problem = deepcopy(example_problem)
    example_problem['A_desc'] = example_problem['pos_desc']
    example_problem['B_desc'] = example_problem['neg_desc']
    example_problem.update(example_problem['application'])

    del example_problem['pos_desc']
    del example_problem['neg_desc']
    del example_problem['pos_samples']
    del example_problem['neg_samples']
    del example_problem['pos_weights']
    del example_problem['neg_weights']
    del example_problem['application']

    example_problem['split'] = {
        'research': {
            'A_samples': example_problem['split']['train']['pos_samples'],
            'B_samples': example_problem['split']['train']['neg_samples'],
        },
        'validation': {
            'A_samples': example_problem['split']['test']['pos_samples'],
            'B_samples': example_problem['split']['test']['neg_samples'],
        }
    }
    example_problem['dataset_abbreviation'] = example_problem['dataset']
    if 'example_hypotheses' not in example_problem:
        example_problem['example_hypotheses'] = []

    for key in ['dataset', 'note', 'hash', 'avg_chars', 'avg_words', 'auc_roc_simple', 'auc_roc_mnb', 'pair_id', 'application_idx_in_pair', 'hypotheses_is_native', 'v1_id', 'v2_id', 'pair_type', 'pair']:
        if key in example_problem:
            del example_problem[key]
    
    return example_problem

def describe_pair(pair):
    """Prints relevant information about dataset"""

    print('----')
    print(f'pair: {pair["pair"]}')
    print(f'# pos samples: {len(pair["pos_samples"])}')
    print(f'# neg samples: {len(pair["neg_samples"])}')
    print(f'avg chars: {pair["avg_chars"]:.2f}')
    print(f'avg words: {pair["avg_words"]:.2f}')
    if 'auc_roc_simple' in pair and 'auc_roc_mnb' in pair:
        print(f'simple auc: {pair["auc_roc_simple"]:.3f}')
        print(f'mnb auc: {pair["auc_roc_mnb"]:.3f}')


def export_schema(pairs):
    pairs = deepcopy(pairs)
    for p in pairs:
        del p['pos_samples']
        del p['neg_samples']
    with open('schema/pairs.yaml', 'w') as f:
        f.write(yaml.dump(pairs))


def main():

    parser = argparse.ArgumentParser(
        prog='Make Pairs',
        description='Creates the pairs comprising the D3 benchmark.')

    parser.add_argument('--full', action='store_true')
    parser.add_argument('--add_errors', action='store_true')
    parser.add_argument('--add_clusters', action='store_true')
    parser.add_argument('--add_datasets', action='store_true')
    parser.add_argument('--add_pairs', action='store_true')
    parser.add_argument('--add_spurious', action='store_true')
    parser.add_argument('--access', action='store_true')
    parser.add_argument('--test_dataset')

    args = parser.parse_args()
    if args.full:
        args.add_errors = True
        args.add_clusters = True
        args.add_datasets = True
        args.add_pairs = True
        args.add_spurious = True

    with open(join(SCHEMA_FOLDER, PAIRS_SCHEMA)) as f:
        pair_schema = yaml.load(f, Loader=SafeLoader)

    all_pairs = []

    if args.test_dataset:
        dataset = args.test_dataset
        generations = pair_schema[dataset]
        for gen_dict in generations:
            if 'note' not in gen_dict:
                gen_dict['note'] = ''
            pairs = gen_dict['pairs']
            del gen_dict['pairs']
            for pair_name, pair_info in pairs.items():
                new_pair = generate_pair(dataset=dataset,
                                         pair=f'{dataset}_{pair_name}',
                                         **gen_dict,
                                         **pair_info)
                describe_pair(new_pair)
                all_pairs.append(new_pair)

    if args.add_errors:

        print('errors')
        task_errors_file = join(COMPONENTS_FOLDER, 'error_analysis_1203.json')
        task_error_pairs = json.load(open(task_errors_file))
        for pair in task_error_pairs:
            all_pairs.append(construct_pair(**pair))

    if args.add_spurious:

        print('spurious')
        spurious_file = join(COMPONENTS_FOLDER, 'spurious_0110.json')

        all_pairs.extend(add_spurious_data(spurious_file))

    if args.add_clusters:

        print('clusters')

        for dataset in CLUSTER_DATASETS:
            print(dataset)
            sample_path = f'clusters/{dataset}/clusters/clusters.json'
            distance_path = f'clusters/{dataset}/clusters/l2_distance.json'

            cluster_pairs = add_cluster_data(
                sample_path=sample_path,
                distance_path=distance_path,
                dataset=dataset,
            )

            all_pairs.extend(cluster_pairs)

    if args.add_datasets:

        print('datasets')
        dataset_constructors = [add_debate, add_amazon_reviews]

        for con in dataset_constructors:
            print('loading')
            pairs = con()
            print('processing')
            for pair in tqdm(pairs):
                all_pairs.append(construct_pair(**pair, truncate=False))

    if args.add_pairs or args.access:

        print('pairs')

        if args.access:
            accessible_datasets = [d for d, metadata in dataset_metadata.items() if metadata['status'] == 'accessible']
            pair_schema = {dataset:generation for dataset, generation in pair_schema.items()
                                if dataset in accessible_datasets}

        for dataset, generations in pair_schema.items():
            for gen_dict in generations:
                if 'note' not in gen_dict:
                    gen_dict['note'] = ''
                pairs = gen_dict['pairs']
                del gen_dict['pairs']
                for pair_name, pair_info in pairs.items():
                    new_pair = generate_pair(dataset=dataset,
                                             pair=f'{dataset}_{pair_name}',
                                             **gen_dict,
                                             **pair_info)
                    describe_pair(new_pair)
                    all_pairs.append(new_pair)

    all_pairs = post_processing(all_pairs, remove_private=False)

    pkl.dump(all_pairs, open(BENCHMARK_FILE, 'wb'))


if __name__ == '__main__':
    main()
