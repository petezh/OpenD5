"""
Utilities for generating distributions from datasets.

Author: Peter Zhang
"""

from itertools import combinations, product
from typing import Callable, Dict, List

import pandas as pd


def make_distributions(
    df: pd.DataFrame,
    text_col: str,
    label_col: str = None,
    label_cols: List[str] = None,
    dropna: bool = True,
) -> Dict[str, List[str]]:
    """
    Accepts a Dataframe, text column, and label column
    and creates a dictionary mapping each value of the
    label to a list of strings from the text column.
    """
    assert label_col or label_cols, "Must include one or more label columns"

    label_cols = label_cols or [label_col]

    distributions = {}

    all_labels = []
    for col in label_cols:
        labels = df[col]
        if dropna:
            labels = labels.dropna()
        all_labels.append(labels.unique())

    for levels in product(*all_labels):
        conditions = df[label_cols[0]] == levels[0]
        dist_name = f"{label_cols[0]}_{levels[0]}"
        for label_col, level in zip(label_cols[1:], levels[1:]):
            conditions = conditions & (df[label_col] == level)
            dist_name += f"_{label_col}_{level}"

        texts = df[conditions][text_col].tolist()
        distributions[dist_name] = texts

    return distributions


def make_pair(
    pair_name: str,
    label_col: str,
    labels_pos: str,
    labels_neg: str,
    desc_pos: str,
    desc_neg: str,
    metadata: Dict = {},
    prefix: str = "",
) -> List[Dict]:
    """
    Accepts the label columns name, two labels, a template
    and a label2kw function and constructs a pair's
    metadata.
    """

    pair = {
        "pair": pair_name,
        "pos_desc": desc_pos,
        "pos_class": [f"{prefix}{label_col}_{label_pos}" for label_pos in labels_pos],
        "neg_desc": desc_neg,
        "neg_class": [f"{prefix}{label_col}_{label_neg}" for label_neg in labels_neg],
    }
    pair.update(metadata)

    return pair


def generate_stepwise(
    df: pd.DataFrame,
    label_col: str,
    desc_template: str,
    name_template: str,
    label2kw: Callable = lambda x: x,
    metadata: Dict = {},
) -> List:
    """
    Accepts a dataframe, text column, and an ordered label column
    and returns a 2-tuple of a distributions and pairs by considering
    pairs of labels in order.
    """

    unique_labels = df[label_col].dropna().unique()
    unique_labels = sorted(unique_labels)

    pairs = []
    for label_pos, label_neg in zip(unique_labels[:-1], unique_labels[1:]):
        kw_pos = label2kw(label_pos)
        kw_neg = label2kw(label_neg)
        pair_name = name_template.format(
            label_col=label_col, label_pos=label_pos, label_neg=label_neg
        )
        desc_pos = desc_template.format(keyword=kw_pos)
        desc_neg = desc_template.format(keyword=kw_neg)
        pair = make_pair(
            pair_name, label_col, [label_pos], [label_neg], desc_pos, desc_neg, metadata
        )
        pairs.append(pair)

    return pairs


def generate_all_pairs(
    df: pd.DataFrame,
    label_col: str,
    desc_template: str,
    name_template: str,
    label2kw: Callable = lambda x: x,
    metadata: Dict = {},
) -> List[Dict]:
    """
    Accepts a dataframe, text column, and a category label column
    and returns a list of pair metadata comparing all pairs.
    """
    assert df[label_col].dtype == "category"

    pairs = []
    labels = df[label_col].dropna().unique()
    for label_pos, label_neg in combinations(labels, 2):
        kw_pos = label2kw(label_pos)
        kw_neg = label2kw(label_neg)
        pair_name = name_template.format(
            label_col=label_col, label_pos=label_pos, label_neg=label_neg
        )
        desc_pos = desc_template.format(keyword=kw_pos)
        desc_neg = desc_template.format(keyword=kw_neg)
        pair = make_pair(
            pair_name, label_col, [label_pos], [label_neg], desc_pos, desc_neg, metadata
        )
        pairs.append(pair)

    return pairs


def generate_one_v_all(
    df: pd.DataFrame,
    label_col: str,
    name_template: str,
    desc_template: str,
    label2kw: Callable = lambda x: x,
    metadata: Dict = {},
    dropna: bool = True,
    all_desc: str = None,
) -> List[Dict]:
    """
    Accepts a dataframe, text column, and a category label column
    and returns a list of pair metadata comparing each pair
    with every other.
    """

    pairs = []
    labels = df[label_col].unique()
    if dropna:
        labels = labels.dropna()
    labels = labels.tolist()

    for label_pos in labels:
        labels_neg = labels.copy()
        labels_neg.remove(label_pos)
        pair_name = name_template.format(label_col=label_col, label=label_pos)
        kw_pos = label2kw(label_pos)
        desc_pos = desc_template.format(keyword=kw_pos)
        if all_desc:
            desc_neg = all_desc
        else:
            desc_neg = desc_template.format(keyword=f"not {kw_pos}")

        pair = make_pair(
            pair_name, label_col, [label_pos], labels_neg, desc_pos, desc_neg, metadata
        )
        pairs.append(pair)

    return pairs
