import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
import time

this_path = os.path.dirname(os.path.abspath(__file__))


def explore_images():
    # Image has average size (203, 256). will resize to (80, 101) so dlib works better (it was trained on
    # (80, 80) images)
    sum_height, sum_width, count = 0, 0, 2222

    for i in range(1, 2223):
        print('Running...')
        file_path = os.path.join(
            "L:/Spring 2021/DA 401/10k US Faces Data/annotations/Face Annotations/Images and Annotations/",
            "{}.jpg".format(i))
        img = cv2.imread(file_path, 1)
        print(i)
        height, width, channels = img.shape
        sum_height += height
        sum_width += width
    print(sum_width / 2222, sum_height / 2222)
    return 0


def get_landmarks():
    # check for dlib saved weights for face landmark detection
    # if it fails, download and extract it manually from
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    check.check_dlib_landmark_weights()

    # load detections performed by dlib library on 3D model and Reference Image
    model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')

    # these images cannot be recognized due to cropping of the face (crop out the hair and ear)
    list_unrecognizable = [42, 162, 215, 303, 451, 650, 770, 829, 901, 938, 1134, 1603, 1726, 1727, 2087, 2211]

    landmarks_dict = {}

    start_time = time.time()  # control for run time. 100 images ~ 6.5 minutes => 2200 images ~ 2.4 hours

    # load query image
    for i in range(1, 2223):
        if i not in list_unrecognizable:
            file_path = os.path.join(
                "L:/Spring 2021/DA 401/10k US Faces Data/annotations/Face Annotations/Images and Annotations/",
                "{}.jpg".format(i))
            img = cv2.imread(file_path, 1)
            img = cv2.resize(img, (80, 101))

            # plt.title('Query Image')
            # plt.imshow(img[:, :, ::-1])  #::-1 because cv2 reads in BGR which is inverse of RGB
            # plt.show()

            # extract landmarks from the query image
            # list containing a 2D array with points (x, y) for each face detected in the query image
            lmarks = feature_detection.get_landmarks(img)

            # plt.figure()
            # plt.title('Landmarks Detected')
            # plt.imshow(img[:, :, ::-1])
            # plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1])

            # perform camera calibration according to the first face detected
            try:
                proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
            except Exception:  # except clause to catch faces that dlib cannot detect
                print(i)
                continue

            # load mask to exclude eyes from symmetry
            eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
            # perform frontalization
            frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)

            # plt.figure()
            # plt.title('Frontalized no symmetry')
            # plt.imshow(frontal_raw[:, :, ::-1])
            # plt.figure()
            # plt.title('Frontalized with soft symmetry')
            # plt.imshow(frontal_sym[:, :, ::-1])
            # plt.show()

            frontal_lmarks = feature_detection.get_landmarks(frontal_raw)

            # put the new frontal landmarks into dictionary so could be input into dataframe later
            landmarks_dict[i] = frontal_lmarks[0]

    print("---%s seconds ---" % (time.time() - start_time))
    return landmarks_dict


if __name__ == "__main__":
    get_landmarks()
    # explore_images()
