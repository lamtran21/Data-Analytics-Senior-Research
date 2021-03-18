import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


# todo: rotate then crop all images
# todo: join demographic, attributes and landmarks.


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


def distance(a, b):
    """ Helper function for normalize_landmarks() to find distance between 2 points"""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_formula(length_line1, length_line2, length_line3):
    """ Helper function for normalize_landmarks() to find an angle given 3 sides of a triangle"""
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a


def rotate_point(origin, point, angle):
    """ Helper function for normalize_landmarks() to rotate a point around the origin given an angle"""
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False


def normalize_landmarks():
    # read in landmarks file
    landmarks_dict = {}
    file_path = "L:/Spring 2021/DA 401/10k US Faces Data/annotations/Face Annotations/Images and Annotations/6_landmarks.txt"
    file = open(file_path)
    coordinates = file.read()
    list_coordinates = coordinates.split('\n')
    # iterate through each landmark in text file and put into corresponding columns
    for j in range(0, 77):
        # key = column index. value = list of coordinates for 1-2222 targets
        if str(j) in landmarks_dict:
            landmarks_dict[str(j)].append(list_coordinates[j].split())
        else:
            landmarks_dict[str(j)] = [list_coordinates[j].split()]

    # read in images to get width + height
    img = plt.imread(
        "L:/Spring 2021/DA 401/10k US Faces Data/annotations/Face Annotations/Images and Annotations/6.jpg")
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # since opencv reads in BGR channel while the pic is in RGB channel
    height, width, channels = bw_img.shape  # get dimensions to rescale later

    # get coordinates of nose (38: mid point of nose) and eyes (49: left eye corner, 53: right eye corner)
    nose = landmarks_dict['38'][0]
    for i in range(len(nose)):
        nose[i] = float(nose[i])
    left_eye = landmarks_dict['49'][0]
    right_eye = landmarks_dict['53'][0]

    # get coordinate of the median point between 2 eyes
    center_2_eyes = ((float(left_eye[0]) + float(right_eye[0])) / 2, (float(left_eye[1]) + float(right_eye[1])) / 2)

    # get coordinate of the center of top of face/image
    center_top_face = (width/2, height-255)

    # find the angle between length_line1 and length_line2.
    length_line1 = distance(center_2_eyes, nose)
    length_line2 = distance(center_top_face, nose)
    length_line3 = distance(center_top_face, center_2_eyes)
    cos_a = cosine_formula(length_line1, length_line2, length_line3)
    angle = np.arccos(cos_a)

    # if after rotation, the point is in the triangle by 3 lines, then this is the correct angle.
    # if after rotation, the point is not, then we need to rotate the other way
    rotated_point = rotate_point(nose, center_2_eyes, angle)
    #rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
    if is_between(nose, center_2_eyes, center_top_face, rotated_point):
        d_angle = np.degrees(-angle)
    else:
        d_angle = np.degrees(angle)

    # show landmarks
    for i in range(0, 77):
        bw_img[int(float(landmarks_dict[str(i)][0][1])), int(float(landmarks_dict[str(i)][0][0]))] = [0, 0, 255]

    # rotate image and show
    bw_img_to_show = Image.fromarray(bw_img)
    bw_img_to_show = np.array(bw_img_to_show.rotate(d_angle, expand=True))
    output = cv2.resize(bw_img_to_show, (width * 2, height * 2))
    cv2.imshow('image', output)
    cv2.imwrite('rotated.jpg', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    return 1
    # path = "L:/Spring 2021/DA 401/10k US Faces Data/"
    # landmarks = create_landmarks_df(path + "annotations/Face Annotations/Images and Annotations/")
    # attributes = create_attributes_df(path + "2kattributes/Full Attribute Scores/psychology attributes/"
    #                                         "psychology-attributes.xlsx")
    # demographic = create_demographic_df(path + "2kattributes/Full Attribute Scores/demographic & others labels/"
    #                                           "demographic-others-labels.xlsx")


# main()
normalize_landmarks()
