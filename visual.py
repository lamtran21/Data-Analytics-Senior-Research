import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
from scipy import stats



def visualize_demo():
    """ For visualizations in final report"""
    df = pd.read_csv('facial_attributes.csv')

    # Figure 1: Distribution of attractiveness rating
    # sb.set_style('whitegrid')
    # sb.distplot(df['attractive'], kde=False)
    # plt.xlabel('Normalized attractiveness ratings')
    # plt.ylabel('Frequency')
    # plt.savefig('figure1.jpg')
    # plt.show()

    # Figure 2
    df_demo = pd.read_csv('facial_attributes_with_demo.csv')
    df_demo['image_race'] = df_demo['image_race'].replace({0: 'Other', 1: 'White', 2: 'Black', 3: 'East Asian', 4: 'South Asian', 5: 'Hispanic', 6: 'Middle Eastern'})
    df_demo['rater_race'] = df_demo['rater_race'].replace({0: 'Other', 1: 'White', 2: 'Black', 3: 'East Asian', 4: 'South Asian', 5: 'Hispanic', 6: 'Middle Eastern'})
    df_demo['image_gender'] = df_demo['image_gender'].replace({0: 'Female', 1: 'Male'})
    df_demo['rater_gender'] = df_demo['rater_gender'].replace({0: 'Female', 1: 'Male'})

    heatmap1_data = pd.pivot_table(df_demo.loc[(df_demo['image_race'] != 'Other') & (df_demo['rater_race'] != 'Other')],
                                                values='attractive',
                                                index=['image_race'],
                                                columns='rater_race')
    sns.heatmap(heatmap1_data, cmap="YlGnBu", annot=True)
    plt.show()

    heatmap2_data = pd.pivot_table(df_demo,
                                   values='attractive',
                                   index=['image_gender'],
                                   columns='rater_gender')
    sns.heatmap(heatmap2_data, cmap="YlGnBu", annot=True)
    plt.show()

    heatmap3_data = pd.pivot_table(df_demo,
                                   values='attractive',
                                   index=['image_age'],
                                   columns='rater_age')
    sns.heatmap(heatmap3_data, cmap="YlGnBu", annot=True)
    plt.show()


    # Figure 6:
    # df = facial_attributes.facial_attributes('aggregated_df_49.csv', None, 0)
    # # min-max normalize the attributes:
    # col_names = df.columns[7:42]
    # for name in col_names:
    #     df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
    # df['average_index'] = df[list(df.columns[7:42])].sum(axis=1)
    # most_attractive = df.sort_values(by=['average_attractive'], ascending=False)[['Filename', 'average_attractive', 'average_index']].head(6)
    # most_average = df.sort_values(by=['average_index'])[['Filename', 'average_index', 'average_attractive']].head(6)
    # print(most_average)
    # print(most_attractive)
    # print(most_average.merge(most_attractive, on='Filename'))

    # 4418 is the most attractive with average index of 10.19, 2711 is the most average with attractive of 0.58
    # 9055 is the in both top 6 attractive (0.80) and average (6.63)

    # font = {'family': 'serif',
    #         'weight': 'normal'
    #         }
    # image1 = cv2.imread('L:/Spring 2021/DA 401/10k US Faces Data/49faces/Publication Friendly 49-Face Database/49 Face images/4418664902_10c6e5d831_o.jpg')
    # image1 = cv2.resize(image1, (160, 200))
    # image2 = cv2.imread('L:/Spring 2021/DA 401/10k US Faces Data/49faces/Publication Friendly 49-Face Database/49 Face images/2711735584_08733d1e45_o.jpg')
    # image2 = cv2.resize(image2, (160, 200))
    # image3 = cv2.imread("L:/Spring 2021/DA 401/10k US Faces Data/49faces/Publication Friendly 49-Face Database/49 Face images/9053039024_d553f12079_o.jpg")
    # image3 = cv2.resize(image3, (160, 200))
    # fig = plt.figure()
    # fig.suptitle('Figure 4: (a) The most attractive face (b) The most average face\n(c) A face that is among the top 6 most attractive '
    #              'and average faces', y=0.255, fontsize=12, fontdict=font)
    #
    # plt.subplot(1, 3, 1), plt.imshow(image1[:, :, ::-1], 'gray')
    # plt.axis('off')
    # plt.title('(a)', fontsize=10, fontdict=font)
    # plt.subplot(1, 3, 2), plt.imshow(image2[:, :, ::-1], 'gray')
    # plt.axis('off')
    # plt.title('(b)', fontsize=10, fontdict=font)
    # plt.subplot(1, 3, 3), plt.imshow(image3[:, :, ::-1], 'gray')
    # plt.title('(c)', fontsize=10, fontdict=font)
    # plt.axis('off')
    # plt.savefig('figure6.jpg')  # To save figure
    # plt.show()  # To show figure


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


def summary_table_cross():
    df = pd.read_csv('facial_attributes_with_demo.csv')
    print(pd.crosstab(df['image_race'], df['rater_race'], values=df['attractive'], aggfunc=np.mean).apply(lambda x: round(x,2), axis=1))
    print(pd.crosstab(df['image_gender'], df['rater_gender'], values=df['attractive'], aggfunc=np.mean).apply(lambda x: round(x,2), axis=1))


def visualize_attribute():
    df = pd.read_csv('facial_attributes.csv')
    df = df.groupby(['Filename', 'Image #']).first().reset_index()
    # print(df.iloc[:, 4:].describe())
    # print(df.columns)
    # sns.distplot(df['13_Eye_size'], kde=False)
    # plt.show()
    # sns.distplot(df['1_face_length'], kde=False)
    # plt.show()
    # sns.distplot(df['25_jaw_length'], kde=False)
    # plt.show()

    normalized_df = (df.iloc[:, 4:] - df.iloc[:, 4:].min()) / (df.iloc[:, 4:].max() - df.iloc[:, 4:].min())
    df['sum'] = df.iloc[:, 4:].sum(axis=1, skipna=True)

    x = df['sum']
    y = df['attractive']
    plt.scatter(x, y, color='black', s=0.7, alpha=0.5)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    plt.show()

if __name__ == "__main__":
    # visualize_demo()
    # summary_table_image()
    # summary_table_rater()
    # summary_table_cross()
    visualize_attribute()
