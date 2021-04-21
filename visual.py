import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import facial_attributes


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def visualize_demo():
    """ For visualizations in final report"""
    # df = pd.read_csv('facial_attributes.csv')
    #
    # # Figure 2: Distribution of attractiveness rating
    # sns.set_style('whitegrid')
    # sns.distplot(df['attractive'], kde=False)
    # plt.xlabel('Normalized attractiveness ratings')
    # plt.ylabel('Frequency')
    # plt.show()

    # Figure 3, 4, 5
    df_demo = pd.read_csv('facial_attributes_with_demo.csv')
    #
    # df_demo['image_race'] = df_demo['image_race'].replace({0: 'Other', 1: 'White', 2: 'Black', 3: 'East Asian', 4: 'South Asian', 5: 'Hispanic', 6: 'Middle Eastern'})
    # df_demo['rater_race'] = df_demo['rater_race'].replace({0: 'Other', 1: 'White', 2: 'Black', 3: 'East Asian', 4: 'South Asian', 5: 'Hispanic', 6: 'Middle Eastern'})
    # df_demo['image_gender'] = df_demo['image_gender'].replace({0: 'Female', 1: 'Male'})
    # df_demo['rater_gender'] = df_demo['rater_gender'].replace({0: 'Female', 1: 'Male'})
    # df_demo['image_age'] = df_demo['image_age'].replace({1: '<20', 2: '20-30', 3: '30-45', 4: '45-60', 5: '60+'})
    # df_demo['rater_age'] = df_demo['rater_age'].replace({1: '<20', 2: '20-30', 3: '30-45', 4: '45-60', 5: '60+'})

    # heatmap1_data = pd.pivot_table(df_demo.loc[(df_demo['image_race'] != 'Other') & (df_demo['rater_race'] != 'Other')],
    #                                             values='attractive',
    #                                             index=['image_race'],
    #                                             columns='rater_race')
    # sns.heatmap(heatmap1_data, cmap="YlGnBu", annot=True, vmin=0, vmax=1,
    #             yticklabels=['White', 'Black', 'E. Asian', 'S. Asian', 'Middle Eastern', 'Hispanic'],
    #             xticklabels=['White', 'Black', 'E. Asian', 'S. Asian', 'Middle Eastern', 'Hispanic'])
    # plt.xlabel('Race of Raters')
    # plt.ylabel('Race of Images')
    # plt.subplots_adjust(bottom=0.3, left=0.2)
    # plt.show()
    #
    # heatmap2_data = pd.pivot_table(df_demo,
    #                                values='attractive',
    #                                index=['image_gender'],
    #                                columns='rater_gender')
    # sns.heatmap(heatmap2_data, cmap="YlGnBu", annot=True, vmin=0, vmax=1)
    # plt.xlabel('Gender of Raters')
    # plt.ylabel('Gender of Images')
    # plt.show()
    #
    heatmap3_data = pd.pivot_table(df_demo,
                                   values='attractive',
                                   index=['image_age'],
                                   columns='rater_age')
    sns.heatmap(heatmap3_data, cmap="YlGnBu", annot=True, vmin=0, vmax=1,
                yticklabels=['<20', '20-30', '30-45', '45-60', '60+'],
                xticklabels=['<20', '20-30', '30-45', '45-60', '60+'])
    plt.xlabel('Age of Raters')
    plt.ylabel('Age of Images')
    plt.show()

    # Figure 10:
    df = facial_attributes.facial_attributes('aggregated_df_49.csv')
    # min-max normalize the attributes:
    col_names = df.columns[4:]
    for name in col_names:
        df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())

    df['average_index'] = df[list(df.columns[4:])].sum(axis=1)
    new_df = df.groupby('Filename').agg({'attractive': 'mean'})  # get the average attractiveness rating for each image
    most_attractive = new_df.sort_values(by=['attractive'], ascending=False)
    most_average = df[['Filename', 'average_index']].groupby('Filename').first().reset_index().sort_values(by=['average_index'])

    print(most_average.head(1))  # print most attractive image
    print(most_attractive.head(1))  # print most average image
    merge = pd.merge(most_attractive, most_average, how='outer', on='Filename').reset_index()
    print(merge['attractive'].describe())  # find min-max of attractiveness
    print(merge['average_index'].describe())  # find min-max of averageness
    merge = merge[(merge['Filename'] == '494043893_511442d9ca_z.jpg') | (merge['Filename'] == '4418664902_10c6e5d831_o.jpg')]
    print(merge)  # print both attractiveness + averageness of the most attractive image and the most average image
    # find image that is in top 6 of both attractiveness and averageness
    print(pd.merge(most_attractive.head(6), most_average.head(6), on='Filename'))

    # 4418 is the most attractive with average index of 7.83 (range: 3.57 - 14.25), attractive index of 0.84 (range: 0.19 - 0.84)
    # 4940 is the most average with average index of 3.57 (range: 3.57 - 14.25), attractive index of 0.62 (range: 0.19 - 0.84)
    # 9055 is the in both top 6 attractive (0.80) and average (5.05)


def summary_table_image():
    """ For statistical testing in Results"""
    df = pd.read_csv('facial_attributes_with_demo.csv')

    # Attractive by gender
    print(df[['image_gender', 'attractive']].groupby('image_gender').agg(['mean', 'count']))
    t, p = stats.ttest_ind(np.array(df['attractive'].loc[df['image_gender'] == 0]),
                           np.array(df['attractive'].loc[df['image_gender'] == 1]), equal_var=False)
    print('t-test p-value: ', p)  # t test

    # Attractive by age
    print(df[['image_age', 'attractive']].groupby('image_age').agg(['mean', 'count']))
    f, p = stats.f_oneway(np.array(df['attractive'].loc[df['image_age'] == 1]),
                          np.array(df['attractive'].loc[df['image_age'] == 2]),
                          np.array(df['attractive'].loc[df['image_age'] == 3]),
                          np.array(df['attractive'].loc[df['image_age'] == 4]),
                          np.array(df['attractive'].loc[df['image_age'] == 5]))
    print('ANOVA p-value: ', p)  # ANOVA test

    # Attractive by race
    print(df[['image_race', 'attractive']].groupby('image_race').agg(['mean', 'count']))
    # Anova test
    f, p = stats.f_oneway(np.array(df['attractive'].loc[df['image_race'] == 0]),
                          np.array(df['attractive'].loc[df['image_race'] == 1]),
                          np.array(df['attractive'].loc[df['image_race'] == 2]),
                          np.array(df['attractive'].loc[df['image_race'] == 3]),
                          np.array(df['attractive'].loc[df['image_race'] == 4]),
                          np.array(df['attractive'].loc[df['image_race'] == 5]),
                          np.array(df['attractive'].loc[df['image_race'] == 6]))
    print('ANOVA p-value: ', p)


def summary_table_rater():
    """ For statistical testing in Results"""
    df = pd.read_csv('facial_attributes_with_demo.csv')

    # Attractive by gender
    print(df[['rater_gender', 'attractive']].groupby('rater_gender').agg(['mean', 'count']))
    t, p = stats.ttest_ind(np.array(df['attractive'].loc[df['rater_gender'] == 0]),
                           np.array(df['attractive'].loc[df['rater_gender'] == 1]), equal_var=False)
    print('t-test p-value: ', p)  # t test

    # Attractive by age
    print(df[['rater_age', 'attractive']].groupby('rater_age').agg(['mean', 'count']))
    f, p = stats.f_oneway(np.array(df['attractive'].loc[df['rater_age'] == 1]),
                          np.array(df['attractive'].loc[df['rater_age'] == 2]),
                          np.array(df['attractive'].loc[df['rater_age'] == 3]),
                          np.array(df['attractive'].loc[df['rater_age'] == 4]),
                          np.array(df['attractive'].loc[df['rater_age'] == 5]))
    print('ANOVA p-value: ', p)  # ANOVA test

    # Attractive by race
    print(df[['rater_race', 'attractive']].groupby('rater_race').agg(['mean', 'count']))
    # Anova test
    f, p = stats.f_oneway(np.array(df['attractive'].loc[df['rater_race'] == 0]),
                          np.array(df['attractive'].loc[df['rater_race'] == 1]),
                          np.array(df['attractive'].loc[df['rater_race'] == 2]),
                          np.array(df['attractive'].loc[df['rater_race'] == 3]),
                          np.array(df['attractive'].loc[df['rater_race'] == 4]),
                          np.array(df['attractive'].loc[df['rater_race'] == 5]),
                          np.array(df['attractive'].loc[df['rater_race'] == 6]))
    print('ANOVA p-value: ', p)



def visualize_attribute():
    df = pd.read_csv('facial_attributes.csv')
    df = df.groupby(['Filename', 'Image #']).first().reset_index()

    # Figure 6, 7, 8
    # sns.set_style('whitegrid')
    # sns.distplot(df['13_Eye_size'], kde=False)
    # plt.xlabel('Difference between eye sizes and its mean')
    # plt.ylabel('Frequency')
    # plt.show()
    #
    # sns.distplot(df['26_ratio_bottom_mid_face'], kde=False)
    # plt.xlabel('Difference between ratios of chin-nostril to nostril-eyebrows and its mean')
    # plt.ylabel('Frequency')
    # plt.show()
    #
    # sns.distplot(df['25_jaw_length'], kde=False)
    # plt.xlabel('Difference between jaw lengths and its mean')
    # plt.ylabel('Frequency')
    # plt.show()

    # Figure 9
    sns.set_style('whitegrid')
    df['sum'] = df.iloc[:, 4:].sum(axis=1, skipna=True)

    x = df['sum']
    y = df['attractive']
    plt.scatter(x, y, color='lightskyblue', s=0.7, alpha=0.5)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "k--")
    plt.xlabel('Sum of the difference between all features and their means')
    plt.ylabel('Attractiveness ratings')
    plt.show()


if __name__ == "__main__":
    visualize_demo()
    # summary_table_image()
    # summary_table_rater()
    # visualize_attribute()
