"""
Code for automatic generation of pairs from datasets.

Author: Peter Zhang
"""

from itertools import chain
from os.path import join
from typing import Dict, List

import pandas as pd
from generate import *
from parameters import *
from tqdm import tqdm


def add_debate() -> List[Dict]:
    """
    Returns a list of pairs for debate.
    """

    df = pd.read_pickle(join(DATASET_FOLDER, "debate.pkl"))

    distributions = {}
    text_col = "abstract"

    df = df[df[text_col].str.split().str.len() < SNIPPET_MAXLEN]

    for label_col in ("year", "arg_type", "argument", "debate_camp"):
        distributions.update(make_distributions(df, text_col, label_col))

    pairs = []

    metadata = {
        "dataset": "debate",
        "generation": "which year the evidence was published",
        "applications": [
            {
                "target": "how debate topics have shifted over time",
                "user": "a coach reflecting on the debate community",
                "purely_exploratory": False,
            }
        ],
        "pair_type": "time",
        "example_hypotheses": [
            "relies on identity politics",
            "mentions the risk of extinction",
        ],
        "flip": True,
    }

    pairs.extend(
        generate_stepwise(
            df=df,
            label_col="year",
            name_template="debate_{label_col}_{label_pos}_{label_neg}",
            desc_template="were published in the year {keyword}",
            metadata=metadata,
        )
    )

    argtype2kw = {
        "k": "Kritiks",
        "aff": "Affirmatives",
        "case_neg": "Case Negatives",
        "cp": "Counterplans",
        "da": "Disadvantages",
        "a2_k": "Kritik Answers",
        "t": "Topicality arguments",
        "th": "Theory arguments",
        "ld": "Lincoln Douglas arguments",
        "politics": "Politics",
        "a2_cp": "Counterplan Answers",
        "imp": "Impact Files",
        "a2_da": "Disadvantage Answers",
        "fw": "Framework arguments",
    }

    metadata = {
        "dataset": "debate",
        "generation": "the category of argument",
        "applications": [
            {
                "target": "the general topics of each category",
                "user": "a novice to policy debate",
                "purely_exploratory": True,
            }
        ],
        "pair_type": "subject",
        "example_hypotheses": [
            "brings up policy issues",
            "critiques societal structures",
        ],
        "flip": True,
    }

    pairs.extend(
        generate_one_v_all(
            df,
            label_col="arg_type",
            name_template="debate_{label_col}_{label}_v_all",
            desc_template='are "{keyword}"',
            label2kw=argtype2kw.get,
            metadata=metadata,
        )
    )

    pairs.extend(
        generate_all_pairs(
            df,
            label_col="arg_type",
            name_template="debate_{label_col}_{label_pos}_{label_neg}",
            desc_template='are "{keyword}"',
            label2kw=argtype2kw.get,
            metadata=metadata,
        )
    )

    argument2kw = {
        # kritiks
        "ableism": "ableism",
        "anthro": "anthropocentrism",
        "afropess": "afropessimism",
        "antiblackness": "blackness",
        "baudrillard": "Baudrillard",
        "cap": "capitalism",
        "fem": "feminism",
        "foucault": "Foucault",
        "heidegger": "Heidegger",
        "militarism": "militarism",
        "neolib": "neoliberalism",
        "psycho": "psychoanalysis",
        "queerness": "queer pessimism",
        "security": "securitization",
        "settcol": "settler colonialism",
        # politics
        "midterms": "midterms",
        "elections": "elections",
        "politics": "politics",
        # counterplans
        "consult": "consultation",
        "states": "states taking action instead",
        "advantage_cp": "other ways to solve the problem",
        "courts": "courts taking action instead",
    }

    metadata = {
        "dataset": "debate",
        "generation": "the argument made",
        "applications": [
            {
                "target": "the claims of each type of argument",
                "user": "a novice to policy debate",
                "purely_exploratory": False,
            }
        ],
        "pair_type": "subject",
        "example_hypotheses": [
            "mentions solving policy problems",
            "critiques societal structures",
        ],
        "flip": True,
    }

    pairs.extend(
        generate_one_v_all(
            df,
            label_col="argument",
            desc_template='are arguments about "{keyword}"',
            name_template="debate_{label_col}_{label}_v_all",
            all_desc="are all other arguments",
            label2kw=argument2kw.get,
            metadata=metadata,
        )
    )

    pairs.extend(
        generate_all_pairs(
            df,
            label_col="argument",
            name_template="debate_{label_col}_{label_pos}_{label_neg}",
            desc_template='are arguments about "{keyword}"',
            label2kw=argument2kw.get,
            metadata=metadata,
        )
    )

    camp2kw = {
        "gdi": "Gonzaga (GDI)",
        "ddi": "Dartmouth DDIx",
        "nhsi": "Northwestern (NHSI)",
        "cdni": "Berkeley (CNDI)",
        "wyoming": "Wyoming",
        "gds": "Georgetown (GDS)",
        "utnif": "Texas (UTNIF)",
        "msdi": "Missouri State (MSDI)",
        "jdi": "Kansas (JDI)",
        "mich_7week": "Michigan (7-week)",
        "scdi": "Sun Country (SCDI)",
        "unt": "North Texas (UNT)",
        "samford": "Samford",
        "endi": "Emory (ENDI)",
        "hss": "Hoya-Spartan Scholars",
        "sdi": "Michigan State (SDI)",
        "mich_classic": "Michigan (Classic)",
        "mndi": "Michigan (MNDI)",
        "rks": "Wake Forest (RKS)",
        "georgia": "Georgia",
        "harvard": "Harvard",
        "wsdi": "Weber State (WSDI)",
        "utd": "UT Dallas (UTD)",
        "naudl": "NAUDL",
        "baylor": "Baylor",
        "mgc": "Mean Green Comet",
        "tdi": "The Debate Intensive",
        "nsd": "National Symposium for Debate",
    }

    metadata = {
        "dataset": "debate",
        "generation": "the debate camp that published the evidence",
        "applications": [
            {
                "target": "what specific topics each debate camp focuses on",
                "user": "a debater deciding which camp to go to",
                "purely_exploratory": False,
            }
        ],
        "pair_type": "author",
        "example_hypotheses": [
            "mentions solving policy problems",
            "critiques societal structures",
        ],
        "flip": True,
    }

    pairs.extend(
        generate_one_v_all(
            df,
            label_col="debate_camp",
            desc_template="are pieces of evidence compiled by {keyword}, a debate camp",
            all_desc="are pieces of evidence from every other debate camp",
            name_template="debate_{label_col}_{label}_v_all",
            label2kw=camp2kw.get,
            metadata=metadata,
        )
    )

    pairs.extend(
        generate_all_pairs(
            df,
            label_col="debate_camp",
            desc_template="are pieces of evidence compiled by {keyword}, a debate camp",
            label2kw=camp2kw.get,
            name_template="debate_{label_col}_{label_pos}_{label_neg}",
            metadata=metadata,
        )
    )

    for pair in tqdm(pairs):
        dists_pos = list(chain(*[distributions[c] for c in pair["pos_class"]]))
        dists_neg = list(chain(*[distributions[c] for c in pair["neg_class"]]))
        pair["pos_samples"] = dists_pos
        pair["neg_samples"] = dists_neg
        del pair["pos_class"]
        del pair["neg_class"]
        pair["hash"] = hash(tuple(dists_pos) + tuple(dists_neg))

    return pairs


def add_amazon_reviews() -> List[Dict]:
    """
    Returns a list of pairs for Amazon reviews.
    """

    df = pd.read_pickle(join(DATASET_FOLDER, "amazon_reviews.pkl"))

    distributions = {}
    text_col = "text"

    df = df[df[text_col].str.split().str.len() < SNIPPET_MAXLEN]

    for label_col in ("year", "product_category"):
        distributions.update(make_distributions(df, text_col, label_col))

    distributions.update(
        make_distributions(df, text_col, label_cols=["product_category", "stars"])
    )

    pairs = []

    metadata = {
        "dataset": "amazon_reviews",
        "generation": "how many stars the review gave",
        "applications": [
            {
                "target": "which specific aspects users dislike, such as the price, features, or performance",
                "user": "a seller of various products on Amazon",
                "purely_exploratory": False,
            }
        ],
        "pair_type": "sentiment",
        "example_hypotheses": [
            "mentions missing a critical piece",
            "complains about the lack of instructions",
        ],
        "flip": False,
    }

    category2kw = {
        "amazon_fashion": "fashion items",
        "beauty": "beauty products",
        "appliances": "appliances",
        "arts_crafts": "arts, crafts, and sewing products",
        "automotive": "automotive",
        "cds": "CDs",
        "cell_phones": "cell phones and accessories",
        "digital_music": "digital music",
        "gift_cards": "gift cards",
        "grocery": "grocery and gourmet food",
        "industrial_scientific": "industrial and scientific products",
        "luxury_beauty": "luxury beauty products",
        "magazines": "magazines",
        "music_instruments": "music instruments",
        "office": "office products",
        "patio": "patio products",
        "pantry": "pantry goods",
        "software": "software",
        "video_games": "video games",
    }

    desc_template = "are reviews of {prodkw} on Amazon giving {starkw} star"
    star_comp = [
        ((1,), (5,)),
        ((1,), (3,)),
        ((2,), (4,)),
        ((4,), (5,)),
    ]

    def stars2keyword(stars: List[int]):
        assert all(s in [1, 2, 3, 4, 5] for s in stars)
        if len(stars) == 1:
            return str(stars[0])
        stars = sorted(stars)
        s = ""
        while len(stars) > 1:
            s += stars.pop(0) + ", "
        return s + "or " + stars[0]

    for product_category in df["product_category"].unique():
        prodkw = category2kw[product_category]
        for pos_stars, neg_stars in star_comp:
            pos_kw = stars2keyword(pos_stars)
            neg_kw = stars2keyword(neg_stars)
            pos_desc = desc_template.format(prodkw=prodkw, starkw=pos_kw)
            neg_desc = desc_template.format(prodkw=prodkw, starkw=neg_kw)
            pair_name = f'amazon_reviews_{product_category}_stars_{"".join(map(str, pos_stars))}_{"".join(map(str, neg_stars))}'
            if pos_stars != (1,):
                pos_desc += "s"
            if neg_stars != (1,):
                neg_desc += "s"

            metadata["dataset_description"] = "Amazon reviews of {prodkw}"
            pair = make_pair(
                pair_name=pair_name,
                label_col="stars",
                labels_pos=pos_stars,
                labels_neg=neg_stars,
                desc_pos=pos_desc,
                desc_neg=neg_desc,
                prefix=f"product_category_{product_category}_",
                metadata=metadata,
            )
            pairs.append(pair)

    for pair in tqdm(pairs):
        dists_pos = list(chain(*[distributions[c] for c in pair["pos_class"]]))
        dists_neg = list(chain(*[distributions[c] for c in pair["neg_class"]]))
        pair["pos_samples"] = dists_pos
        pair["neg_samples"] = dists_neg
        pair["hash"] = hash(tuple(dists_pos) + tuple(dists_neg))
        del pair["pos_class"]
        del pair["neg_class"]

    return pairs
