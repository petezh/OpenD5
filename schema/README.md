OpenD5 Schema
===

Each problem is [represented](https://github.com/ruiqi-zhong/D5#problem-representation) as a combination of a pair of corpora and a research goal. See below for an example.

![](../img/example.jpeg)

[`pairs.yaml'](pairs.yaml) contains most of the metadata:
- The highest level of structure is each dataset's abbreviation (e.g. `abc_headlines`)
- Each dataset contains several *generations*, which describe different ways of splitting a dataset.
  - A generation description is comprised of the splitting feature (e.g. `the year they were published`).
  - Applications are comprised of a `target` and `user` with some `example_hypotheses`.
  - The `v2-origid` field is for internal tracking and is non-essential.
  - If the context should allow for any kind of hypothesis, the `purely_exploratory` flag is set to `True`.
  - The `pair_type` is assigned according to the taxonomy outlined below.
  - If a `flip` is appropriate, the user should consider both the original and swapped (B vs. A) versions of the problem.
- Each generation contains a list of `pairs`.
  - Each pair has `pos_desc` and `neg_desc` (e.g. `are ABC news headlines from 2007`)
  - The classes correspond to distribution names.

[`dataset.yaml`](datasets.yaml) contains dataset-level features:
- A description of the types of text samples, e.g. `headlines published by ABC news, an American news company`.
- The `discipline` and `expertise` required for the dataset.
- The `status` of the dataset, which will be:
  - `public` if there is a public license.
  - `private` if the dataset was privately shared.
  - `accessible` if the dataset can be accessed easily but has unclear license.
- The preprocessing steps, which are mostly for internal tracking.