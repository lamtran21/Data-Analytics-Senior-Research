import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

# todo: join demographic, attributes and landmarks. Then scale the landmarks (in str rn) (in landmarks)


def create_landmarks_df(path):
    """
    A program that creates a data frame for xy-coordinates from facial landmarks that were contained in text files.
    In the data frame, the columns are the landmarks (0-77) and the rows are the people (1-2222) (marked by the
    "person_number" column)

    :param path: path to the folder that contains the landmarks coordinate. Name of each file in that folder
                is the number of the person who that face belongs to
    :return: a dataframe that contains 77 xy-coordinates for facial landmarks of 2222 images.
    """
    row_index = [str(x) for x in range(1, 2223)]
    landmarks_dict = {'person_number': row_index}

    # iterate through files 1-2222
    for i in range(1, 2223):
        file_path = os.path.join(path, "{}_landmarks.txt".format(i))
        print("Check-in point: {}_landmarks.txt".format(i))
        file = open(file_path)
        coordinates = file.read()
        list_coordinates = coordinates.split('\n')
        # iterate through each landmark in text file and put into corresponding columns
        for j in range(0, 77):
            # key = column index. value = list of coordinates for 1-2222 targets
            if str(j) in landmarks_dict:
                landmarks_dict[str(j)].append(str(list_coordinates[j].split()))
            else:
                landmarks_dict[str(j)] = [str(list_coordinates[j].split())]

    landmarks = pd.DataFrame.from_dict(landmarks_dict)
    return landmarks


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
    print(attributes['attractive'].head(), attributes['attractive'].min(), attributes['attractive'].max())

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
    print(agg_attributes['count'].value_counts())
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
    demographic['Attractive'] = (demographic['Attractive'] - demographic['Attractive'].min()) /\
                                (demographic['Attractive'].max() - demographic['Attractive'].min())

    # Group by and summarize by person
    agg_demographic = demographic.groupby(['Filename', 'Image #'], as_index=False).agg({
                                                                'Attractive': ['sum', 'count'],
                                                                'Gender': (lambda x: pd.Series.mode(x).values[0]),
                                                                'Race': (lambda x: pd.Series.mode(x).values[0]),
                                                                'Age': (lambda x: pd.Series.mode(x).values[0]),
                                                                })

    # drop second-level column names that were created by group_by (eg: sum, count)
    agg_demographic.columns = agg_demographic.columns.droplevel(1)

    # rename the columns as we have 2 columns named attractive
    agg_demographic.columns = ['Filename', 'Image #', 'attractive', 'count', 'gender', 'race', 'age']

    print(agg_demographic.head())
    return agg_demographic


def main():
    path = "L:/Spring 2021/DA 401/10k US Faces Data/"
    # landmarks = create_landmarks_df(path + "annotations/Face Annotations/Images and Annotations/")
    # attributes = create_attributes_df(path + "2kattributes/Full Attribute Scores/psychology attributes/"
    #                                         "psychology-attributes.xlsx")
    demographic = create_demographic_df(path + "2kattributes/Full Attribute Scores/demographic & others labels/"
                                               "demographic-others-labels.xlsx")


main()
