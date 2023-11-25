import os
import pandas as pd

from functions import constants


def read_json_dataset(filename):
    file = os.path.join(os.getcwd(), constants.DATASETS_FOLDER + filename)
    df = pd.read_json(file, lines=True)
    return df


def read_csv_dataset(filename, separator):
    file = os.path.join(os.getcwd(), constants.DATASETS_FOLDER + filename)
    df = pd.read_csv(file, sep=separator)
    return df


def read_table_dataset(filename, columns):
    file = os.path.join(os.getcwd(), constants.DATASETS_FOLDER + filename)
    df = pd.read_table(file, header=None, names=columns)
    return df


def save_dataset(dataset, filename):
    file = os.path.join(os.getcwd(), constants.DATASETS_FOLDER + filename)
    dataset.to_csv(file, sep=';')


def load_dataset(filename):
    return pd.read_csv(filename, sep=';', index_col=0)

