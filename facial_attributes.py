import pandas as pd
import ast
import math


def distance(p1, p2):
    """ A function that calculates the distance between p1 and p2"""
    return math.sqrt(((int(p1[0]) - int(p2[0])) ** 2) + ((int(p1[1]) - int(p2[1])) ** 2))


def midpoint(p1, p2):
    """ A function that get the midpoint from 2 points p1 and p2"""
    return [(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2]


def facial_attributes():
    """ A function that creates 35 facial attributes from facial landmarks"""

    agg = pd.read_csv('aggregated_df.csv')
    agg['landmarks'] = agg['landmarks'].apply(lambda x: x[1:-1].split('\n '))
    agg['landmarks'] = agg['landmarks'].apply(lambda x: [ast.literal_eval(i.replace(' ', ',')) for i in x])

    # calculate facial attributes
    agg['1_face_length'] = agg['landmarks'].apply(lambda x: distance(x[27], x[8]))
    agg['2_face_width_at_eye'] = agg['landmarks'].apply(lambda x: distance(x[0], x[16]))
    agg['3_face_width_at_mouth'] = agg['landmarks'].apply(lambda x: distance(x[4], x[12]))
    agg['4_distance_pupil'] = agg['landmarks'].apply(lambda x: distance(midpoint(x[37], x[40]), midpoint(x[44], x[47])))
    agg['5_ratio_2_3'] = agg['2_face_width_at_eye'] / agg['3_face_width_at_mouth']
    agg['6_ratio_1_2'] = agg['1_face_length'] / agg['2_face_width_at_eye']
    agg['7_ratio_1_3'] = agg['1_face_length'] / agg['3_face_width_at_mouth']
    agg['8_ratio_4_2'] = agg['4_distance_pupil'] / agg['2_face_width_at_eye']
    agg['10_lEyebrow_thick'] = agg['landmarks'].apply(lambda x: distance(x[19], midpoint(x[17], x[21])))
    agg['9_rEyebrow_thick'] = agg['landmarks'].apply(lambda x: distance(x[24], midpoint(x[22], x[26])))
    agg['11_rEyebrow_arch'] = agg['landmarks'].apply(lambda x: x[24][1] - x[22][1])
    agg['12_lEyebrow_arch'] = agg['landmarks'].apply(lambda x: x[19][1] - x[21][1])
    agg['13_rEye_height'] = agg['landmarks'].apply(lambda x: distance(midpoint(x[43], x[44]), midpoint(x[47], x[46])))
    agg['14_lEye_height'] = agg['landmarks'].apply(lambda x: distance(midpoint(x[37], x[38]), midpoint(x[41], x[40])))
    agg['15_rEye_width'] = agg['landmarks'].apply(lambda x: distance(x[42], x[45]))
    agg['16_lEye_width'] = agg['landmarks'].apply(lambda x: distance(x[39], x[36]))
    agg['17_rEye_size'] = agg['15_rEye_width'] * agg['13_rEye_height']
    agg['18_lEye_size'] = agg['16_lEye_width'] * agg['14_lEye_height']
    agg['19_distance_2_eyes'] = agg['landmarks'].apply(lambda x: distance(x[42], x[39]))
    agg['20_nose_width'] = agg['landmarks'].apply(lambda x: distance(x[31], x[35]))
    agg['21_nose_length'] = agg['landmarks'].apply(lambda x: distance(x[27], x[33]))
    agg['22_nose_size'] = agg['20_nose_width'] * agg['21_nose_length']
    agg['23_cheekbone_width'] = agg['2_face_width_at_eye'] - agg['3_face_width_at_mouth']
    agg['24_ratio_23_2'] = agg['23_cheekbone_width'] / agg['2_face_width_at_eye']
    agg['25_thickness_middle_top_lip'] = agg['landmarks'].apply(lambda x: distance(x[51], x[62]))
    agg['26_thickness_right_top_lip'] = agg['landmarks'].apply(lambda x: distance(x[53], x[64]))
    agg['27_thickness_left_top_lip'] = agg['landmarks'].apply(lambda x: distance(x[49], x[60]))
    agg['28_average_thick_top_lip'] = (agg['25_thickness_middle_top_lip'] + agg['26_thickness_right_top_lip'] +
                                                                            agg['27_thickness_left_top_lip']) / 3
    agg['29_thickness_middle_lower_lip'] = agg['landmarks'].apply(lambda x: distance(x[66], x[57]))
    agg['30_thickness_middle_both_lip'] = agg['26_thickness_right_top_lip'] + \
                                                    agg['29_thickness_middle_lower_lip']
    agg['31_lip_length'] = agg['landmarks'].apply(lambda x: distance(x[48], x[54]))
    agg['32_chin_length'] = agg['landmarks'].apply(lambda x: distance(x[8], x[57]))
    agg['33_right_jaw_length'] = agg['landmarks'].apply(lambda x: distance(x[8], x[12]))
    agg['34_left_jaw_length'] = agg['landmarks'].apply(lambda x: distance(x[8], x[4]))
    agg['35_ratio_bottom_mid_face'] = agg['landmarks'].apply(lambda x: x[33][1] - midpoint(x[19],
                                        x[24])[1]) / agg['landmarks'].apply(lambda x: distance(x[8], x[33]))
    # subtract each value from the average of each column
    for i in range(8, 43):
        agg.iloc[:, i] = agg.iloc[:, i].apply(lambda x: abs(x - agg.iloc[:, i].mean()))
    
    agg.to_csv('facial_attributes.csv', index=False)
    return agg


if __name__ == "__main__":
    facial_attributes()
