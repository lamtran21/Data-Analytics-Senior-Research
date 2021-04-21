import pandas as pd
import get_landmarks
import csv
import numpy as np
import matplotlib.pyplot as plt


def create_attributes_df(path, demo=0):
    """
    A program that reads in an excel file that contains attribute scores for 2222 people and turns it into
    a pandas data frame. If demo = 0, the program only reads in the primary key column (filename), secondary key column
    (Image #), and attractiveness ratings. If demo = 1, the program also reads in the rater's race, age, and gender.

    :param path: path to the excel file
    :param demo: whether read in the demographic info or not
    :return: a data frame that contains attribute scores for 2222 people.
    """
    # Read in files
    if demo == 0:
        attributes = pd.read_excel(path, na_values=['NaN'], usecols=['Filename', 'Image #', 'attractive'])
    else:
        attributes = pd.read_excel(path, na_values=['NaN'], usecols='A,B,AD,AV,AU,AW')
        attributes.columns = ['Filename', 'Image #', 'attractive', 'rater_age', 'rater_gender', 'rater_race']

    # Drop NA values
    attributes = attributes.dropna()  # drop 58 ratings

    # Min-max normalize the attractive ratings
    attributes['attractive'] = (attributes['attractive'] - attributes['attractive'].min()) / \
                               (attributes['attractive'].max() - attributes['attractive'].min())

    # Comment: some images have 16 ratings which is weird because it was claimed each has 15 before removing NAs
    return attributes


def create_demographic_df(path, demo=0):
    """
    A program that reads in an excel file that contains attractive ratings for 2222 people and turns it into
    a pandas data frame. If demo = 0, the program only reads in the primary key column (filename), secondary key column
    (Image #), and attractiveness ratings. If demo = 1, the program reads in the demographics of the image and not
    the attractiveness ratings.

    :param path: path to the excel file
    :param demo: whether read in the demographic information or not
    :return: a data frame that contains attractive ratings for 2222 people.
    """
    if demo == 0:
        demographic = pd.read_excel(path, na_values=['NaN'], usecols=['Filename', 'Image #', 'Attractive'])
        # Min-max normalize attractive ratings
        demographic['Attractive'] = (demographic['Attractive'] - demographic['Attractive'].min()) / \
                                    (demographic['Attractive'].max() - demographic['Attractive'].min())
        # Drop NA values
        demographic = demographic.dropna()
        demographic.columns = ['Filename', 'Image #', 'attractive']
    else:
        demographic = pd.read_excel(path, na_values=['NaN'], usecols=['Filename', 'Image #', 'Age', 'Gender', 'Race'])
        # Drop NA values
        demographic = demographic.dropna()
        demographic = demographic.groupby(['Filename', 'Image #'], as_index=False).agg({
                                                                    'Gender': (lambda x: pd.Series.mode(x).values[0]),
                                                                    'Race': (lambda x: pd.Series.mode(x).values[0]),
                                                                    'Age': (lambda x: pd.Series.mode(x).values[0])})
        demographic.columns = ['Filename', 'Image #', 'image_gender', 'image_race', 'image_age']
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
    agg.to_csv('aggregated_df.csv', index=False)

    # create dataframe with demographic information
    attributes_demo = create_attributes_df(path + "2kattributes/Full Attribute Scores/psychology attributes/"
                                             "psychology-attributes.xlsx", demo=1)
    demographic_demo = create_demographic_df(path + "2kattributes/Full Attribute Scores/demographic & others labels/"
                                               "demographic-others-labels.xlsx", demo=1)
    agg_demo = pd.concat([attributes_demo, demographic], axis=0)
    agg_demo = pd.merge(agg_demo, demographic_demo, how='left', on=['Filename', 'Image #'])
    agg_demo = pd.merge(agg_demo, landmarks, on=['Image #'])
    agg_demo.to_csv('aggregated_df_with_demo.csv', index=False)

    # trial code: perform if already have aggregated_df.csv and do not want to run get_landmarks() again
    # attributes_demo = create_attributes_df(path + "2kattributes/Full Attribute Scores/psychology attributes/"
    #                                          "psychology-attributes.xlsx", demo=1)
    # demographic_demo = create_demographic_df(path + "2kattributes/Full Attribute Scores/demographic & others labels/"
    #                                            "demographic-others-labels.xlsx", demo=1)
    # demographic = create_demographic_df(path + "2kattributes/Full Attribute Scores/demographic & others labels/"
    #                                            "demographic-others-labels.xlsx")
    # landmarks = pd.read_csv('aggregated_df.csv', usecols=['Image #', 'landmarks'])
    # landmarks = landmarks.groupby(['Image #'], as_index=False).agg({
    #                                                                 'landmarks': (lambda x: pd.Series.mode(x).values[0])})
    # agg_demo = pd.concat([attributes_demo, demographic], axis=0)
    # agg_demo = pd.merge(agg_demo, demographic_demo,  how='left', on=['Filename', 'Image #'])
    # agg_demo = pd.merge(agg_demo, landmarks, on=['Image #'])
    # agg_demo.to_csv('aggregated_df_with_demo.csv', index=False)


def data_aggregation_49():
    path = "L:/Spring 2021/DA 401/10k US Faces Data/49faces/Publication Friendly 49-Face Database/"
    attributes = create_attributes_df(path + "Attribute Scores/psychology attributes/psychology-attributes.xlsx")
    demographic = create_demographic_df(path + "Attribute Scores/demographic & others labels/demographic-others-labels.xlsx")
    landmarks = get_landmarks.get_landmarks_49()
    landmarks = pd.Series(landmarks).to_frame().reset_index()
    landmarks.columns = ['Filename', 'landmarks']

    # merge datasets together
    agg = pd.concat([attributes, demographic], axis=0)
    agg = pd.merge(agg, landmarks, on=['Filename'])

    agg.to_csv('aggregated_df_49.csv', index=False)


if __name__ == "__main__":
    # data_aggregation()
    data_aggregation_49()
