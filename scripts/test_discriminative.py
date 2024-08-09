"""
Baselines for problem difficulty.
"""

import random
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SAMPLE_SIZE = 1000


def discriminated_w_simple_feature(
    positive_samples: List[str],
    negative_samples: List[str],
    sample_size: int = SAMPLE_SIZE,
    k: int = 20,
) -> float:
    """
    Assess basic discriminatory power with top k words,
    length, number of words, capital letters, and numbers.
    Returns max(auc, 1-auc) for the basic single feature.
    """

    # sample down if necessary
    random.seed(2022)
    if len(positive_samples) > sample_size:
        positive_samples = random.sample(
            positive_samples,
            sample_size,
        )
    if len(negative_samples) > sample_size:
        negative_samples = random.sample(negative_samples, sample_size)

    # ground truth
    labels = np.append(np.ones(len(positive_samples)), np.zeros(len(negative_samples)))

    def eval_power(preds: List[float]):
        """Evaluates predictions against ground truth"""
        roc_auc = roc_auc_score(labels, preds)
        return max(roc_auc, 1 - roc_auc)

    # vectorize
    vectorizer = CountVectorizer()
    all_samples = positive_samples + negative_samples
    vectorizer.fit(all_samples)
    pos_counts = vectorizer.transform(positive_samples)
    neg_counts = vectorizer.transform(negative_samples)

    # count freqs
    pos_freq = pos_counts.sum(axis=0)
    neg_freq = neg_counts.sum(axis=0)
    net_freq = pos_freq - neg_freq
    arr = abs(np.array(net_freq.flatten())[0])
    args = np.argsort(arr)[-k:]

    def test_word_power(arg: int):
        """Tests a given word index discriminatory power"""
        pos_pred = pos_counts[:, arg].toarray().flatten() > 0
        neg_pred = neg_counts[:, arg].toarray().flatten() > 0
        preds = np.append(pos_pred, neg_pred)
        return eval_power(preds)

    # engineer other features
    max_word_auc = max(map(test_word_power, args))
    pos_words = np.asarray(pos_counts.sum(axis=1)).flatten()
    neg_words = np.asarray(neg_counts.sum(axis=1)).flatten()
    total_words_auc = eval_power(np.append(pos_words, neg_words))
    len_auc = eval_power(list(map(len, all_samples)))
    number_idxs = [idx for w, idx in vectorizer.vocabulary_.items() if w.isnumeric()]
    pos_nums = np.asarray(pos_counts[:, number_idxs].sum(axis=1)).flatten()
    neg_nums = np.asarray(neg_counts[:, number_idxs].sum(axis=1)).flatten()
    num_auc = eval_power(np.append(pos_nums, neg_nums))
    capital_auc = eval_power(list(map(lambda s: s[0].isupper(), all_samples)))

    # return max discrim
    return max(max_word_auc, total_words_auc, len_auc, capital_auc, num_auc)


def discriminated_w_mnb(
    positive_samples: List[str],
    negative_samples: List[str],
    sample_size: int = SAMPLE_SIZE,
    k: int = 20,
) -> float:
    """
    Assess basic discriminatory power with top k words,
    length, number of words, capital letters, and numbers.
    Returns max(auc, 1-auc) for the basic single feature.
    """

    # build pipeline
    text_clf = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("sel", SelectKBest(chi2, k=k)),
            ("clf", MultinomialNB()),
        ]
    )

    # sample down
    if len(positive_samples) > sample_size:
        positive_samples = random.sample(positive_samples, sample_size)
    if len(negative_samples) > sample_size:
        negative_samples = random.sample(negative_samples, sample_size)

    # build design
    X = positive_samples + negative_samples  # ground truth
    y = np.append(np.ones(len(positive_samples)), np.zeros(len(negative_samples)))

    # fit classifier
    clf = text_clf.fit(X, y)

    return roc_auc_score(y, clf.predict_proba(X)[:, 1])


if __name__ == "__main__":

    import pickle as pkl

    pairs = pkl.load(open("benchmarks/benchmark_1201.pkl", "rb"))
    for pair in pairs:
        positive_samples = pair["pos_samples"]
        negative_samples = pair["neg_samples"]
        s = discriminated_w_simple_feature(positive_samples, negative_samples)
        print(s)
        print(pair["pair"])
    exit(0)
