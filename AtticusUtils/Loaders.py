import pandas as pd
import random


def create_training_set(train_data=[{}], limit=0, split=0.8):
    """Load data from the Atticus dataset, splitting off a held-out set."""
    random.shuffle(train_data)
    train_data = train_data[-limit:]

    texts, labels = zip(*train_data)
    split = int(len(train_data) * split)

    # Return data in format that matches example here:
    # https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
    return (texts[:split], labels[:split]), (texts[split:], labels[split:])


def load_atticus_data(filepath='./data/master_clauses.csv'):
    """
    Load data from the atticus csv (omitting the answer cols as we want to train classifiers
    not question answering).

    Data is returned in the Spacy training format:
        TRAIN_DATA = [
            ("text1", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}})
        ]

    A list of headers is also returned so you can add these labels. FYI, the Filename and Doc name
    columns are dropped as well.

    """

    # Load csv
    atticus_clauses_df = pd.read_csv(filepath)

    # Do a little post-processing
    data_headers = [h for h in list(atticus_clauses_df.columns) if not "Answer" in h]
    data_headers.pop(0)  # Drop filename col (index 0 for col 1)
    data_headers.pop(0)  # Drop doc name (orig col 2 (index 1) but now first col (index 0))

    training_values = {i: 0 for i in data_headers}
    atticus_clauses_data_df = atticus_clauses_df.loc[:, data_headers]

    train_data = []

    # Iterate over csv to build training data dict
    for header in atticus_clauses_data_df.columns:

        for row in atticus_clauses_data_df[[header]].iterrows():

            value = row[1][header]

            if not pd.isnull(value):
                train_data.append((value, {'cats': {**training_values, header: 1}}))

    return train_data, data_headers
