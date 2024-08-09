"""
Constructs the open_review distributions.
"""

from collections import defaultdict

import numpy as np
import openreview
import pandas as pd

client = openreview.Client(baseurl="https://api.openreview.net")


def scrape():
    """
    Uses OenReview API to scrape ICLR reviews.
    """

    invites = [
        (
            2018,
            "ICLR.cc/2018/Conference/-/Blind_Submission",
            "ICLR.cc/2018/Conference/-/Paper.*/Official_Review",
        ),
        (
            2019,
            "ICLR.cc/2019/Conference/-/Blind_Submission",
            "ICLR.cc/2019/Conference/-/Paper.*/Official_Review",
        ),
        (
            2020,
            "ICLR.cc/2020/Conference/-/Blind_Submission",
            "ICLR.cc/2020/Conference/Paper.*/-/Official_Review",
        ),
        (
            2021,
            "ICLR.cc/2021/Conference/-/Blind_Submission",
            "ICLR.cc/2021/Conference/Paper.*/-/Official_Review",
        ),
    ]

    metadata = []

    for year, submission_invite, review_invite in invites:

        submissions = openreview.tools.iterget_notes(
            client, invitation=submission_invite
        )
        submissions_by_forum = {n.forum: n for n in submissions}

        reviews = openreview.tools.iterget_notes(client, invitation=review_invite)
        reviews_by_forum = defaultdict(list)
        for review in reviews:
            reviews_by_forum[review.forum].append(review)

        for forum in submissions_by_forum:

            forum_reviews = reviews_by_forum[forum]
            review_ratings = [int(n.content["rating"][0]) for n in forum_reviews]
            average_rating = np.mean(review_ratings)

            submission_content = submissions_by_forum[forum].content
            abstract = submission_content["abstract"]

            forum_metadata = {
                "forum": forum,
                "review_ratings": review_ratings,
                "average_rating": average_rating,
                "abstract": abstract,
                "year": year,
            }

            metadata.append(forum_metadata)

    df = pd.DataFrame(metadata)
    great_papers = df[df.average_rating >= 7].abstract.tolist()
    good_papers = df[
        (df.average_rating >= 5) & (df.average_rating < 7)
    ].abstract.tolist()
    bad_papers = df[df.average_rating < 5].abstract.tolist()

    return great_papers, good_papers, bad_papers
