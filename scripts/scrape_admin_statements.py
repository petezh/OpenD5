"""
Constructs the admin_statements distributions.
"""

import glob

import pdfplumber

from utils import *
from parameters import *

def scrape():
    """
    Scrapes the statements of administration policy from the
    statements-of-administration-policy-main repository.
    """

    NAME = 'admin_statements'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'

    administrations = ['44-Obama', '45-Trump', '46-Biden']

    data = {}

    for admin in administrations:
        print(admin)

        files = glob.glob(
            f'{directory}/statements-of-administration-policy-main/archive/statements/{admin}/**/*.pdf')

        statements = []

        for file in files:
            text = ""
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + " "

                if (loc := text.find("The Administration")) != 0:
                    text = text[loc:].replace('\n', '')
                    text = text.replace('*', '').strip()
                    texts = split_delimiter_(text, '\n')
                    texts = split_truncate(texts)
                    statements.extend(texts)
            except:
                pass

        data[admin] = statements

    return data
