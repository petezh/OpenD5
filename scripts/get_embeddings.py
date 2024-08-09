"""
Utillities for creating embeddings.

Author: Ruiqi Zhong
"""

import glob
import json
import os
from argparse import ArgumentParser
from functools import partial
from os.path import join
from typing import List

import numpy as np
import torch
import tqdm
from transformers import (BertModel, BertTokenizer, RobertaModel,
                          RobertaTokenizer, T5EncoderModel, T5Tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
BSIZE = 32
SAVE_EVERY = 10000
DEFAULT_SAMPLES = 100000
DATA_FOLDER = "unlabeled"


def roberta_embed(model_tokenizer, sentences: List[str]):
    """
    Embeds a list of sentences using Roberta.
    """
    model, tokenizer = model_tokenizer
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = model(**inputs).pooler_output
        return outputs.cpu().numpy()


def t5_embed(model_tokenizer, sentences: List[str]):
    """
    Embeds a list of sentences using T5.
    """

    model, tokenizer = model_tokenizer
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = torch.mean(model(**inputs).last_hidden_state, dim=1)
        return outputs.cpu().numpy()


def bert_embed(model_tokenizer, sentences: List[str]):
    """
    Embeds a list of sentences using BERT.
    """

    model, tokenizer = model_tokenizer
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = model(**inputs).pooler_output
        return outputs.cpu().numpy()


def embed_sentences(
    embed_func,
    sentences: List[str],
    samples: int,
    bsize: int = BSIZE,
    save_dir: str = None,
):
    """
    Embeds a list of sentences using a given embedding function.
    """

    embeddings, texts = [], []
    save_threshold = [i * SAVE_EVERY for i in range(1, samples // SAVE_EVERY + 2)]
    for i in tqdm.trange(0, len(sentences), bsize):
        sentence_batch = sentences[i : i + bsize]
        embeddings.extend(embed_func(sentence_batch))
        texts.extend(sentence_batch)
        finished_count = i + bsize
        if save_dir is not None and finished_count > save_threshold[0]:
            embeddings = np.array(embeddings)
            np.save(os.path.join(save_dir, f"{finished_count}.npy"), embeddings)
            json.dump(
                texts, open(os.path.join(save_dir, f"{finished_count}.json"), "w")
            )
            save_threshold.pop(0)
            embeddings = []
            texts = []
    if len(embeddings) > 0:
        np.save(
            os.path.join(save_dir, f"{finished_count}.npy"),
            np.concatenate(embeddings, axis=0),
        )
        json.dump(texts, open(os.path.join(save_dir, f"{finished_count}.json"), "w"))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--get_all", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)

    args = parser.parse_args()

    get_all = args.get_all
    model_name = args.model_name
    samples = args.samples
    dataset = args.dataset

    if "roberta" in model_name:
        model = RobertaModel.from_pretrained(model_name).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model_tokenizer = (model, tokenizer)
        embed_func = partial(roberta_embed, model_tokenizer)
    elif "t5" in model_name:
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model_tokenizer = (model, tokenizer)
        embed_func = partial(t5_embed, model_tokenizer)
    elif "bert" in model_name:
        model = BertModel.from_pretrained(model_name).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model_tokenizer = (model, tokenizer)
        embed_func = partial(bert_embed, model_tokenizer)

    if get_all:
        files = glob.glob("unlabeled/*")
        datasets = [file[10:-5] for file in files]
    else:
        datasets = [dataset]

    for dataset in datasets:

        print(f"embedding {dataset}")
        save_dir = f"results/{dataset}_embeddings"

        os.makedirs(save_dir, exist_ok=True)

        filename = join(DATA_FOLDER, f"{dataset}.json")
        data = json.load(open(filename, "r"))[:samples]

        embeddings = embed_sentences(embed_func, data, samples, save_dir=save_dir)
