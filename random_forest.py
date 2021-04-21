import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import facial_attributes
import cv2


def linear_model():
    """ Run 25 univariate linear model to find the sign of each """
    df = pd.read_csv('facial_attributes.csv')
    # transform everything to np.array
    attractive_rating = np.array(df['attractive'])
    # remove unused columns from df (axis = 1 refers to column)
    df = df.drop(
        ['Filename', 'Image #', 'landmarks', 'attractive'], axis=1)
    # get the sign of each variable by running univariate model
    for i in range(26):
        col = df.iloc[:, i]
        col = np.array(col)

        X = sm.add_constant(col)
        model = sm.OLS(attractive_rating, X)
        result = model.fit()
        print(i+1, round(result.params[1],4), result.pvalues[1] < 0.05)


def random_forest():
    """ Model 1 and Model 2: Run on whole dataset. For model 1, in df.drop do not drop any features"""
    df = pd.read_csv('facial_attributes_with_demo.csv')
    # transform everything to np.array
    attractive_rating = np.array(df['attractive'])
    # remove unused columns from df (axis = 1 refers to column)
    df = df.drop(['Filename', 'Image #', 'landmarks', 'attractive', '19_ratio_18_2', '10_Eyebrow_arch', '15_nose_width',
                  '14_distance_2_eyes', '11_Eye_height', '16_nose_length', '21_thickness_middle_lower_lip',
                  '22_thickness_middle_both_lip', '7_ratio_1_3', '3_face_width_at_mouth', '1_face_length',
                  '18_cheekbone_width', '4_distance_pupil', '17_nose_size', '20_average_thick_top_lip',
                  '25_jaw_length', '8_ratio_4_2', '9_Eyebrow_thick', '2_face_width_at_eye', '24_chin_length',
                  '23_lip_length'], axis=1)
    # saving column names for visualization
    df_cols = list(df.columns)
    # convert to numpy array
    df = np.array(df)

    # split the data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(df, attractive_rating, test_size=0.2, random_state=21)

    # control for run time
    start_time = time.time()

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=21)
    # Train the model on training data
    rf.fit(train_x, train_y)
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_x)
    predictions_train = rf.predict(train_x)
    # Visualize

    sns.set_style('whitegrid')

    x = predictions
    y = test_y
    plt.scatter(x, y, color='lightskyblue', s=0.7, alpha=0.5)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "k--")
    plt.xlabel('Predicted Attractiveness Ratings')
    plt.ylabel('Actual attractiveness ratings')
    plt.show()

    # Print out the mean squared error (mse)
    print('Test Mean Squared Error:', mean_squared_error(test_y, predictions))
    print('Train Mean Squared Error:', mean_squared_error(train_y, predictions_train))
    print("---%s seconds ---" % (time.time() - start_time))

    # Get numerical variable importance
    importance = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importance = [(feature, round(importance, 4)) for feature, importance in zip(df_cols, importance)]
    # Sort the variable importance by most important first
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    # Print out the variables and importance
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance];


def random_forest_subset():
    """ Models that run on demographic subset of the data, determined by df.loc[]"""
    df = pd.read_csv('facial_attributes_with_demo.csv')
    df = df.loc[(df['rater_race'] == 1) & (df['image_race'] == 5)]
    print(df.shape)
    # transform everything to np.array
    attractive_rating = np.array(df['attractive'])
    # remove unused columns from df (axis = 1 refers to column)
    df = df.drop(['Filename', 'Image #', 'landmarks', 'attractive', 'rater_race', 'image_race', 'rater_age',
                  'image_age', 'rater_gender', 'image_gender', '21_thickness_middle_lower_lip', '5_ratio_2_3',
                  '19_ratio_18_2', '14_distance_2_eyes', '10_Eyebrow_arch', '1_face_length', '13_Eye_size',
                  '18_cheekbone_width', '24_chin_length', '8_ratio_4_2', '2_face_width_at_eye',
                  '15_nose_width', '11_Eye_height', '22_thickness_middle_both_lip', '6_ratio_1_2',
                  '20_average_thick_top_lip', '16_nose_length', '7_ratio_1_3', '26_ratio_bottom_mid_face',
                  '23_lip_length', '9_Eyebrow_thick'], axis=1)
    # saving column names for visualization
    df_cols = list(df.columns)
    # convert to numpy array
    df = np.array(df)

    # split the data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(df, attractive_rating, test_size=0.2, random_state=21)

    # control for run time
    start_time = time.time()

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=21)
    # Train the model on training data
    rf.fit(train_x, train_y)
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_x)
    predictions_train = rf.predict(train_x)
    # Print out the mean squared error (mse)
    print('Test Mean Squared Error:', mean_squared_error(test_y, predictions))
    print('Train Mean Squared Error:', mean_squared_error(train_y, predictions_train))
    print("---%s seconds ---" % (time.time() - start_time))

    # Get numerical variable importance
    importance = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importance = [(feature, round(importance, 4)) for feature, importance in zip(df_cols, importance)]
    # Sort the variable importance by most important first
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    # Print out the variables and importance
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance];


if __name__ == "__main__":
    linear_model()
    # random_forest()
    # random_forest_subset()



