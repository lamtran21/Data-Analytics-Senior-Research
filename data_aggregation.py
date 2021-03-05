import pandas as pd
import os

# todo: create 3 dataframes. 1: landmarks. 2: attributes. 3: demographic
# todo: 3 columns for attractiveness: 2 from attributes and 1 from demographic

# todo: xy in landmarks are strings. Needs to convert to int.


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
    a pandas data frame. The program also filters out columns that are not needed in the future, then group by
    to get the average score of each person (as one person is rated by multiple people).

    :param path: path to the excel file
    :return: a data frame that contains attribute scores for 2222 people.
    """
    attributes = pd.read_excel(path)
    # print(attributes.head())
    print(attributes.shape)
    print(set(attributes['Filename']))
    return attributes


def main():
    path = "L:/Spring 2021/DA 401/10k US Faces Data/"
    # landmarks = create_landmarks_df(path + "annotations/Face Annotations/Images and Annotations/")
    attributes = create_attributes_df(path + "2kattributes/Full Attribute Scores/psychology attributes/"
                                             "psychology-attributes.xlsx")


main()
