"""
General utiltiies.
"""

import gzip
import json
import os
import shutil
import tarfile
from html.parser import HTMLParser
from io import BytesIO, StringIO
from itertools import chain
from os.path import join
from typing import Callable, Dict, List, Tuple, Union
from zipfile import ZipFile

import gdown
import nltk
import pandas as pd
import requests
from markdown import Markdown
from nltk.tokenize import sent_tokenize
from parameters import *

"""
=============
Text cleaning
=============
"""


def remove_empty(samples: Union[List[str], List[Tuple]]) -> List[str]:
    """Utility for removing empty strings"""
    if not samples:
        return samples
    s = samples[0]
    if isinstance(s, str):
        return [s for s in samples if s.strip()]
    return [s for s in samples if s[0].strip()]


def split_delimiter(
    snippets: List[str], delimiter: str = "\n"
) -> List[Tuple[str, float]]:
    """
    Splits a list of snippets on a specific delimiter and
    returns snippets weighted by word count.
    """
    return list(chain(*[split_delimiter_(w_s, delimiter) for w_s in snippets]))


def split_delimiter_(snippet: str, delimiter: str = "\n"):
    """
    Splits a snippet based on a delimiter and returns
    snippets weighted by word count.
    """
    new_snippets = []

    total_weight = len(snippet.split())
    split_snips = snippet.split(delimiter)
    for split_snip in split_snips:
        weight = len(split_snip.split())
        if weight:
            new_snippets.append((split_snip, weight / total_weight))

    return new_snippets


def give_full_weight(snippets: List[str]) -> List[Tuple[str, int]]:
    """Assigns weight 1 to every snippet"""
    return [(snippet, 1) for snippet in snippets]


def split_truncate(
    weighted_snippets: Union[List[Tuple[str, float]], List[str]],
    cap: int = SNIPPET_MAXLEN,
) -> List[str]:
    """Applies split_truncate_ to a list of snippets"""
    if isinstance(weighted_snippets[0], str):
        weighted_snippets = give_full_weight(weighted_snippets)
    return list(chain(*[split_truncate_(w_s, cap=cap) for w_s in weighted_snippets]))


def split_truncate_(
    weighted_snippet: Tuple[str, float], cap: int = SNIPPET_MAXLEN
) -> List[Tuple[str, float]]:
    """
    Splits any snippet over the truncation limit and returns
    new snippets weighted by word count.
    """
    if isinstance(weighted_snippet, str):
        weighted_snippet = (weighted_snippet, 1)
    assert len(weighted_snippet) == 2, "Did not provide weighted snippets"

    snippet, weight = weighted_snippet
    assert len(snippet.split()) > 0, f"String is empty: {snippet}"
    new_snippets = []
    new_snippet = ""
    word_count = 0

    # split snippet into sentences
    total_weight = len(snippet.split())
    sentences = sent_tokenize(snippet)

    for sentence in sentences:

        sentence_length = len(sentence.split())

        # if we need to tokenize
        if word_count + sentence_length > cap:
            new_snippet = new_snippet.strip()
            new_weight = word_count / total_weight * weight  # new weight
            new_snippets.append((new_snippet, new_weight))
            new_snippet, word_count = "", 0

        # check if sentence is too long
        if sentence_length > cap:
            continue

        # else add sentence
        new_snippet += sentence + " "
        word_count += sentence_length

    # add additional snippet
    new_snippet = new_snippet.strip()
    new_weight = word_count / total_weight * weight  # new weight
    new_snippets.append((new_snippet, new_weight))

    return new_snippets


def split_truncate_word(
    weighted_snippets: Union[List[Tuple[str, float]], List[str]],
    cap: int = SNIPPET_MAXLEN,
) -> List[str]:
    """Applies split_truncate_ to a list of snippets"""

    if isinstance(weighted_snippets[0], str):
        weighted_snippets = give_full_weight(weighted_snippets)
    return list(
        chain(*[split_truncate_word_(w_s, cap=cap) for w_s in weighted_snippets])
    )


def split_truncate_word_(
    weighted_snippet: Tuple[str, float], cap: int = SNIPPET_MAXLEN
):
    """
    Splits any snippet over the truncation limit at the
    level of the word and returns new snippets weighted by word count.
    """
    if isinstance(weighted_snippet, str):
        weighted_snippet = (weighted_snippet, 1)
    assert len(weighted_snippet) == 2, "Did not provide weighted snippets"

    snippet, weight = weighted_snippet
    assert len(snippet.split()) > 0, f"String is empty: {snippet}"

    words = snippet.split()
    total_words = len(words)

    # split snippet into sentences
    i = 0
    while (i + 1) * cap < total_words:
        new_snippet = " ".join(words[i * cap : (i + 1) * cap])
        yield (new_snippet, cap / total_words * weight)
        i += 1

    tail = total_words - i * cap
    if tail:
        new_snippet = " ".join(words[i * cap :])
        yield (new_snippet, tail / total_words * weight)


def split_df(df: pd.DataFrame, text_col: str, splitter: Callable = split_truncate_):
    """
    Accepts a DataFrame and a column of text, applies truncation and
    splits rows into multiple as necessary.
    """
    assert text_col in df.columns, f"Columns {text_col} not in DataFrame"

    df[text_col] = df[text_col].apply(splitter)
    df = df.explode(text_col)

    df = df[df[text_col].str.len() > 0]

    return df


def shorten_snippet(snippet: str, cap: int = 256) -> str:
    """
    Shortens a text subject to some limit.
    """

    sentences = sent_tokenize(snippet)
    new_snippet = ""
    num_words = 0

    for sentence in sentences:
        num_words = len(sentence.split())
        if num_words > cap:
            return new_snippet.strip()
        new_snippet += sentence + " "

    return new_snippet.strip()


def sentence_tokenize(snippets: List[str]) -> List[str]:
    """
    Uses NLTK to sentence toknize a list of snippets.
    """

    all_sentences = []
    for snippet in snippets:
        sentences = nltk.tokenize.sent_tokenize(snippet)
        all_sentences += sentences

    return all_sentences


class MLStripper(HTMLParser):
    """
    Class to strip HTML from a string.

    Borrowed from https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python.
    """

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html: str) -> str:
    """Removes HTML from a string."""
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def encode_ascii(text: str) -> str:
    """Encode a text into ASCII."""
    return text.encode("ascii", "ignore").decode("utf-8")


def unmark_element(element, stream=None):
    """Outputs plain, unformatted markdown."""
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False


def unmark(text: str) -> str:
    """Removes markdown formatting from a Markdown object."""
    return __md.convert(text)


"""
=============
File download
=============
"""


def download_zip(url: str, directory: str):
    """Downloads and extracts contents of a zip folder."""

    req = requests.get(url)
    zip = ZipFile(BytesIO(req.content))
    zip.extractall(directory)


def extract_zip(path: str, directory: str):
    """Extracts all files from a zip folder."""
    with ZipFile(path, "r") as zip:
        zip.extractall(directory)


def download_tar(url: str, directory: str):
    """Downloads and extracts contents of a tar file."""

    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode="r|gz")
    file.extractall(path=directory)


def download_file(url: str, directory: str, filename: str):
    """Downloads and names file."""
    req = requests.get(url)
    os.makedirs(directory, exist_ok=True)
    with open(join(directory, filename), "wb") as f:
        f.write(req.content)


def download_drive_file(id: str, directory: str, filename: str):
    """Downloads files from Google Drive."""

    url = f"https://drive.google.com/uc?id={id}"
    os.makedirs(directory, exist_ok=True)
    gdown.download(url, join(directory, filename))


def download_drive_zip(id: str, directory: str) -> None:
    """Downloads files from Google Drive."""

    url = f"https://drive.google.com/uc?id={id}"
    os.makedirs(directory, exist_ok=True)
    gdown.download(url, join(directory, "drive.zip"))
    with ZipFile(join(directory, "drive.zip"), "r") as zip_ref:
        zip_ref.extractall(directory)


def download_gz(url, directory, filename) -> None:
    """Downloads and opens gzip file."""

    download_file(url, directory, filename + ".gz")

    with gzip.open(join(directory, filename + ".gz"), "rb") as f_in:
        with open(join(directory, filename), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


"""
=========
File save
=========
"""


def save_dataset(df: pd.DataFrame, name: str):
    """Saves dataframe to datasets folder."""

    os.makedirs(DATASET_FOLDER, exist_ok=True)
    df.to_pickle(join(DATASET_FOLDER, f"{name}.pkl"))


def save_output_json(data: Dict, name: str):
    """Saves output data to output folder."""

    output = {
        "name": name,
        "data": data,
    }

    output_file = f"{OUTPUT_FOLDER}/{name}.json"
    with open(output_file, "w") as outfile:
        json.dump(output, outfile)


def save_unlabeled_json(sentences: List[str], name: str):
    """Saves unlabeled data to output folder."""

    os.makedirs(UNLABELED_FOLDER, exist_ok=True)
    output_file = f"{UNLABELED_FOLDER}/{name}.json"
    with open(output_file, "w") as outfile:
        json.dump(sentences, outfile)


def delete_downloads():
    """Clears the downloads folder."""
    shutil.rmtree(DOWNLOAD_FOLDER)
