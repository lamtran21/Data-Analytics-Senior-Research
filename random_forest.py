import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time


def base_model():
    """ A linear regression model that serves as a baseline. The goal for the random forest model is to
    have lower mean_squared_error than this model"""

    df = pd.read_csv('facial_attributes.csv')
    df_X_train = df.iloc[1:2000, 8:15]
    df_y_train = df.iloc[1:2000, 3]
    df_X_test = df.iloc[2001:, 8:15]
    df_y_test = df.iloc[2001:, 3]
    regr = linear_model.LinearRegression()
    regr.fit(df_X_train, df_y_train)
    pred = regr.predict(df_X_test)
    # Calculate the absolute errors
    errors = abs(pred - df_y_test)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / df_y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Base line accuracy:', round(accuracy, 2), '%.')
    return mean_squared_error(df_y_test, pred)
    # has negative r2 because too many variables. if 5-15 (include control vars) it goes to 30%


def random_forest():
    df = pd.read_csv('facial_attributes.csv')

    # one-hot encoding for gender, race, and age
    df['gender'] = df['gender'].replace([0, 1], ['female', 'male'])
    df['race'] = df['race'].replace([0, 1, 2, 3, 4, 5, 6], ['other', 'white', 'black', 'east_asian', 'south_asian', 'hispanic', 'middle_eastern'])
    df['age'] = df['age'].replace([1, 2, 3, 4, 5], ['less_20', '20_30', '30_45', '45-60', '60+'])
    df = pd.get_dummies(df, columns=['age', 'gender', 'race'])

    # transform everything to np.array
    attractive_rating = np.array(df['average_attractive'])
    # remove unused columns from df (axis = 1 refers to column)
    df = df.drop(['Filename', 'Image #', 'landmarks', 'average_attractive', 'average_unattractive'], axis=1)
    # saving column names for visualization
    df_cols = list(df.columns)
    # convert to numpy array
    df = np.array(df)

    # split the data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(df, attractive_rating, test_size=0.2, random_state=21)

    # control for run time
    start_time = time.time()

    # Instantiate model with 100 decision trees (1000 trees improve 0.05% accuracy)
    rf = RandomForestRegressor(n_estimators=100, random_state=21)
    # Train the model on training data
    rf.fit(train_x, train_y)
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_x)
    # Calculate the absolute errors
    errors = abs(predictions - test_y)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_y)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Model accuracy:', round(accuracy, 2), '%.')
    print("---%s seconds ---" % (time.time() - start_time))

    # Get numerical feature importances
    importance = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(df_cols, importance)]
    # Sort the feature importance by most important first
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    # Print out the features and importance
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance];

    # Note: race is not important. features of the upper face (eyebrows, eyes) are not important
    # of the features that are of lower face, only the lower lip is not important


if __name__ == "__main__":
    # base_model()
    random_forest()
