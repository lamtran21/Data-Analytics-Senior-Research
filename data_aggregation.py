import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

# todo: remove Nan => normalize => group by (mean for ratings, mode for demographics) => join 3 dfs

# todo: xy in landmarks are strings. Needs to convert to int. (will do later)


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
    (Image #), attractiveness ratings and unattractiveness ratings.

    :param path: path to the excel file
    :return: a data frame that contains attribute scores for 2222 people.
    """
    attributes = pd.read_excel(path, na_values=['NaN'], usecols="A,B,W,AD")
    # print(attributes.head())
    # print(attributes.shape)
    return attributes


def create_demographic_df(path):
    """
    A program that reads in an excel file that contains demographic information for 2222 people and turns it into
    a pandas data frame. The program only reads in the primary key column (filename), secondary key column
    (Image #), age, gender, race, and attractiveness ratings.

    :param path: path to the excel file
    :return: a data frame that contains demographic information for 2222 people.
    """
    demographic = pd.read_excel(path, na_values=['NaN'], usecols="A,B,C,D,O,S")
    # print(demographic.head())
    # print(demographic.shape)
    # look into correlation between attractiveness - unattractiveness before deciding if use unattractiveness
    return demographic


def normalize_ratings(df, rating_col):
    x_array = np.array(df[rating_col])
    normalized_x = preprocessing.normalize([x_array])
    print(normalized_x)
    print(max(normalized_x))
    print(min(normalized_x))
    print(np.average(normalized_x))


def main():
    path = "L:/Spring 2021/DA 401/10k US Faces Data/"
    # landmarks = create_landmarks_df(path + "annotations/Face Annotations/Images and Annotations/")
    attributes = create_attributes_df(path + "2kattributes/Full Attribute Scores/psychology attributes/"
                                             "psychology-attributes.xlsx")
    # demographic = create_demographic_df(path + "2kattributes/Full Attribute Scores/demographic & others labels/"
    #                                           "demographic-others-labels.xlsx")
    # normalize_ratings(attributes, "attractive") <- needs to remove NaN from df


main()
