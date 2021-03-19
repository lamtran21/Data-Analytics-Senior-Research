import pandas as pd
import get_landmarks
import csv
import numpy as np
import matplotlib.pyplot as plt


# todo: run facedetection for 2222 images then get into one df then merge


def create_attributes_df(path):
    """
    A program that reads in an excel file that contains attribute scores for 2222 people and turns it into
    a pandas data frame. The program only reads in the primary key column (filename), secondary key column
    (Image #), attractiveness ratings and unattractiveness ratings. It then outputs the primary key column
    (filename), secondary key column (Image #), sum attractive ratings (to be combined with other dataset),
    sum unattractive ratings, and count (to be used to calculate the average later on) (count = how many ratings
    this person gets, as I have dropped NAs and some do not have 15 ratings anymore)

    :param path: path to the excel file
    :return: a data frame that contains attribute scores for 2222 people.
    """

    # Read in files
    attributes = pd.read_excel(path, na_values=['NaN'], usecols="A,B,W,AD")

    # Drop NA values
    attributes = attributes.dropna()  # drop 58 ratings
    # print(len(set(attributes.iloc[:, 0])))  # no one among 2222 people is completely removed from data

    # Min-max normalize the attractive ratings
    attributes['attractive'] = (attributes['attractive'] - attributes['attractive'].min()) / \
                               (attributes['attractive'].max() - attributes['attractive'].min())
    attributes['unattractive'] = (attributes['unattractive'] - attributes['unattractive'].min()) / \
                                 (attributes['unattractive'].max() - attributes['unattractive'].min())

    # Group by and summarize by person
    agg_attributes = attributes.groupby(['Filename', 'Image #'], as_index=False).agg({
        'attractive': ['sum', 'count'],
        'unattractive': 'sum'
    })
    # drop second-level column names that were created by group_by (eg: sum, count)
    agg_attributes.columns = agg_attributes.columns.droplevel(1)

    # rename the columns as we have 2 columns named attractive
    agg_attributes.columns = ['Filename', 'Image #', 'attractive', 'count', 'unattractive']

    # Comment: some images have 16 ratings which is weird because it was claimed each has 15 before removing NAs
    # print(agg_attributes['count'].value_counts())
    return agg_attributes


def create_demographic_df(path):
    """
    A program that reads in an excel file that contains demographic information for 2222 people and turns it into
    a pandas data frame. The program only reads in the primary key column (filename), secondary key column
    (Image #), age, gender, race, and attractiveness ratings.

    :param path: path to the excel file
    :return: a data frame that contains demographic information for 2222 people.
    """
    demographic = pd.read_excel(path, na_values=['NaN'], usecols="A,B,C,D,O,S")

    # Drop NA values
    demographic = demographic.dropna()  # remove 34 observations

    # Min-max normalize attractive ratings
    demographic['Attractive'] = (demographic['Attractive'] - demographic['Attractive'].min()) / \
                                (demographic['Attractive'].max() - demographic['Attractive'].min())

    # Group by and summarize by person
    agg_demographic = demographic.groupby(['Filename', 'Image #'], as_index=False).agg({
        'Attractive': ['sum', 'count'],
        'Gender': (lambda x: pd.Series.mode(x).values[0]),  # find mode
        'Race': (lambda x: pd.Series.mode(x).values[0]),
        'Age': (lambda x: pd.Series.mode(x).values[0]),
    })

    # drop second-level column names that were created by group_by (eg: sum, count)
    agg_demographic.columns = agg_demographic.columns.droplevel(1)

    # rename the columns as we have 2 columns named attractive
    agg_demographic.columns = ['Filename', 'Image #', 'attractive', 'count', 'gender', 'race', 'age']

    return agg_demographic


def data_aggregation():
    path = "L:/Spring 2021/DA 401/10k US Faces Data/"
    attributes = create_attributes_df(path + "2kattributes/Full Attribute Scores/psychology attributes/"
                                             "psychology-attributes.xlsx")
    demographic = create_demographic_df(path + "2kattributes/Full Attribute Scores/demographic & others labels/"
                                               "demographic-others-labels.xlsx")
    landmarks = get_landmarks.get_landmarks()
    landmarks = pd.Series(landmarks).to_frame().reset_index()
    landmarks.columns = ['Image #', 'landmarks']

    # merge datasets together
    agg = pd.merge(attributes, demographic, on=['Filename', 'Image #'])
    agg = pd.merge(agg, landmarks, on=['Image #'])
    # calculate the average ratings
    agg['average_attractive'] = (agg['attractive_x'] + agg['attractive_y']) / (agg['count_x'] + agg['count_y'])
    agg['average_unattractive'] = agg['unattractive'] / agg['count_x']
    # two sets of ratings have corr = -0.855
    agg = agg[['Filename', 'Image #', 'landmarks', 'average_attractive', 'average_unattractive', 'age', 'gender', 'race']]
    agg.to_csv('aggregated_df.csv', index=False)

    # Data exploration
    # print(agg.groupby('gender')['average_attractive'].mean())
    # print(agg.groupby('age')['average_attractive'].mean())
    # print(agg.groupby('race')['average_attractive'].mean())
    # plt.hist(agg['average_attractive'])
    # plt.show()


if __name__ == "__main__":
    data_aggregation()
