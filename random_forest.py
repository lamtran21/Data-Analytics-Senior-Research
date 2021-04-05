import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
import statsmodels.api as sm
import facial_attributes


def base_model():
    """ A linear regression model that serves as a baseline. The goal for the random forest model is to
    have lower mean_squared_error than this model"""
    df = pd.read_csv('facial_attributes.csv')
    # one-hot encoding for gender, race, and age
    df['gender'] = df['gender'].replace([0, 1], ['female', 'male'])
    df['age'] = df['age'].replace([1, 2, 3, 4, 5], ['less_20', '20_30', '30_45', '45-60', '60+'])
    df['race'] = df['race'].replace([0, 1, 2, 3, 4, 5, 6],
                                    ['other', 'white', 'black', 'east_asian', 'south_asian', 'hispanic',
                                     'middle_eastern'])
    df = pd.get_dummies(df, columns=['age', 'gender', 'race'])

    # transform everything to np.array
    attractive_rating = np.array(df['average_attractive'])
    # remove unused columns from df (axis = 1 refers to column)
    df = df.drop(
        ['Filename', 'Image #', 'landmarks', 'average_attractive', 'average_unattractive'], axis=1)
    # convert to numpy array
    # df1 = np.array(df)
    # # base Linear model
    # X2 = sm.add_constant(df1)
    # est = sm.OLS(attractive_rating, X2)
    # est2 = est.fit()
    # print(est2.summary())

    # get the sign of each variable by running univariate model
    for i in range(10):
        col = df.iloc[:, i]
        col = np.array(col)

        X = sm.add_constant(col)
        est = sm.OLS(attractive_rating, X)
        est2 = est.fit()
        print(est2.summary())


def random_forest():
    df = pd.read_csv('facial_attributes.csv')

    # one-hot encoding for gender, race, and age
    df['gender'] = df['gender'].replace([0, 1], ['female', 'male'])
    df['race'] = df['race'].replace([0, 1, 2, 3, 4, 5, 6],
                                    ['other', 'white', 'black', 'east_asian', 'south_asian', 'hispanic',
                                     'middle_eastern'])
    df['age'] = df['age'].replace([1, 2, 3, 4, 5], ['less_20', '20_30', '30_45', '45-60', '60+'])
    # df = pd.get_dummies(df, columns=['age', 'gender', 'race'])

    # transform everything to np.array
    attractive_rating = np.array(df['average_attractive'])
    # remove unused columns from df (axis = 1 refers to column)
    df = df.drop(
        ['Filename', 'Image #', 'landmarks', 'average_attractive', 'average_unattractive', 'gender', 'race', 'age'],
        axis=1)
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

    # Get numerical variable importance
    importance = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importance = [(feature, round(importance, 4)) for feature, importance in zip(df_cols, importance)]
    # Sort the variable importance by most important first
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    # Print out the variables and importance
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance];

    # Note: features of the upper face (eyebrows, eyes) are not important. Of the features that are of lower face,
    # only the lower lip is not important


def visual():
    """ For visualizations in final report"""
    # df = pd.read_csv('facial_attributes.csv')

    # Figure 1: Distribution of attractiveness rating
    # sb.set_style('whitegrid')
    # sb.distplot(df['average_attractive'], kde=False)
    # plt.xlabel('Normalized attractiveness ratings')
    # plt.ylabel('Frequency')
    # plt.savefig('figure1.jpg')
    # plt.show()

    # Figure 5: Correlation matrix for all attributes
    # df = df.drop(
    #     ['Filename', 'Image #', 'landmarks', 'average_attractive', 'average_unattractive', 'gender', 'race', 'age'],
    #     axis=1)
    # corrMatrix = df.corr()
    # fig, ax = plt.subplots()
    # sb.heatmap(corrMatrix)
    # ax.set_xticklabels([i for i in range(1, 36, 2)])
    # ax.set_yticklabels([i for i in range(1, 36, 2)])
    # plt.savefig('figure5.jpg')
    # plt.show()

    # Figure 6:
    df = facial_attributes.facial_attributes('aggregated_df_49.csv', None, 0)
    # min-max normalize the attributes:
    col_names = df.columns[7:42]
    for name in col_names:
        df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
    df['average_index'] = df[list(df.columns[7:42])].sum(axis=1)
    most_attractive = df.sort_values(by=['average_attractive'], ascending=False)[['Filename', 'average_attractive', 'average_index']].head(6)
    most_average = df.sort_values(by=['average_index'])[['Filename', 'average_index', 'average_attractive']].head(6)
    print(most_average)
    print(most_attractive)
    print(most_average.merge(most_attractive, on='Filename'))

    # 4418 is the most attractive with average index of 10.19, 2711 is the most average with attractive of 0.58
    # 9055 is the in both top 6 attractive (0.80) and average (6.63)

def summary_table():
    """ For statistical testing in Results"""
    df = pd.read_csv('facial_attributes.csv')
    print(df.groupby('gender')['average_attractive'].mean())
    print(df.groupby('age')['average_attractive'].mean())
    print(df.groupby('race')['average_attractive'].mean())

    t, p = stats.ttest_ind(np.array(df['average_attractive'].loc[df['gender'] == 0]), np.array(df['average_attractive'].loc[df['gender'] == 1]), equal_var=False)
    print(p)


if __name__ == "__main__":
    # base_model()
    # random_forest()
    visual()
    # summary_table()
