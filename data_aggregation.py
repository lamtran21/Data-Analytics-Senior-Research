import pandas as pd
import get_landmarks
import csv
import numpy as np
import matplotlib.pyplot as plt


def create_attributes_df(path):
    """
    A program that reads in an excel file that contains attribute scores for 2222 people and turns it into
    a pandas data frame. The program only reads in the primary key column (filename), secondary key column
    (Image #), and attractiveness ratings.

    :param path: path to the excel file
    :return: a data frame that contains attribute scores for 2222 people.
    """

    # Read in files
    attributes = pd.read_excel(path, na_values=['NaN'], usecols=['Filename', 'Image #', 'attractive'])

    # Drop NA values
    attributes = attributes.dropna()  # drop 58 ratings

    # Min-max normalize the attractive ratings
    attributes['attractive'] = (attributes['attractive'] - attributes['attractive'].min()) / \
                               (attributes['attractive'].max() - attributes['attractive'].min())

    # Comment: some images have 16 ratings which is weird because it was claimed each has 15 before removing NAs
    return attributes


def create_demographic_df(path):
    """
    A program that reads in an excel file that contains demographic information for 2222 people and turns it into
    a pandas data frame. The program only reads in the primary key column (filename), secondary key column
    (Image #), and attractiveness ratings.

    :param path: path to the excel file
    :return: a data frame that contains demographic information for 2222 people.
    """
    demographic = pd.read_excel(path, na_values=['NaN'], usecols=['Filename', 'Image #', 'Attractive'])

    # Drop NA values
    demographic = demographic.dropna()  # remove 34 observations

    # Min-max normalize attractive ratings
    demographic['Attractive'] = (demographic['Attractive'] - demographic['Attractive'].min()) / \
                                (demographic['Attractive'].max() - demographic['Attractive'].min())

    demographic.columns = ['Filename', 'Image #', 'attractive']
    return demographic


def data_aggregation():
    path = "L:/Spring 2021/DA 401/10k US Faces Data/"
    attributes = create_attributes_df(path + "2kattributes/Full Attribute Scores/psychology attributes/"
                                             "psychology-attributes.xlsx")
    demographic = create_demographic_df(path + "2kattributes/Full Attribute Scores/demographic & others labels/"
                                               "demographic-others-labels.xlsx")
    landmarks = get_landmarks.get_landmarks()
    landmarks = pd.Series(landmarks).to_frame().reset_index()
    landmarks.columns = ['Image #', 'landmarks']

    # concat attributes + demographic then merge with landmarks
    agg = pd.concat([attributes, demographic], axis=0)
    agg = pd.merge(agg, landmarks, on=['Image #'])
    agg.to_csv('aggregated_df2.csv', index=False)


def data_aggregation_49():
    path = folder = "L:/Spring 2021/DA 401/10k US Faces Data/49faces/Publication Friendly 49-Face Database/"
    attributes = create_attributes_df(path + "Attribute Scores/psychology attributes/psychology-attributes.xlsx")
    demographic = create_demographic_df(path + "Attribute Scores/demographic & others labels/demographic-others-labels.xlsx")
    landmarks = get_landmarks.get_landmarks_49()
    landmarks = pd.Series(landmarks).to_frame().reset_index()
    landmarks.columns = ['Filename', 'landmarks']

    # merge datasets together
    agg = pd.concat([attributes, demographic], axis=0)
    agg = pd.merge(agg, landmarks, on=['Image #'])

    agg.to_csv('aggregated_df_49.csv', index=False)


if __name__ == "__main__":
    data_aggregation()
    # data_aggregation_49()
