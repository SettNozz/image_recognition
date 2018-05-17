import argparse as ap
import cv2
from imutils import paths
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

def create_dis_kp(image_paths, sift_object, des_list):
    for image_path in image_paths:
        im = cv2.imread(image_path)
        if np.any(im == None):
            print ("No such file {}\nCheck if the file exists".format(image_path))
            exit()
        kpts, des = sift_object.detectAndCompute(im, None)
        des_list.append((image_path, des))

def array_of_des(des_list, descriptors):
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))

def im_features_create(im_features, des_list, voc, image_paths):
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        #print len(test_features[i])
        for w in words:
            im_features[i][w] += 1
    #print(test_features)

def scale_features(stdSlr, im_features):
    return stdSlr.transform(im_features)

def save_image_test(name, path):
    image = cv2.imread(name)
    cv2.imwrite(path + name, image)

def save_image_to_path(name, path):
    image = cv2.imread(name)
    #print path + name
    cv2.imwrite(path + name, image)

def image_train(image, description):
    training_set = 'train/'
    if not os.path.exists(training_set +  description):
        os.makedirs(training_set + description)
    save_image_to_path(image, training_set + description + '/')
    image_paths = []
    training_names = os.listdir(training_set)
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(training_set, training_name)
        class_path = list(paths.list_images(dir))
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1

    # Create feature extraction and keypoint detector objects
    sift_object = cv2.xfeatures2d.SIFT_create()
    des_list = []
    create_dis_kp(image_paths, sift_object, des_list)


    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    array_of_des(des_list, descriptors)

    #Preform k-means clustering
    n_clusters = 300
    voc, variance = kmeans(descriptors, n_clusters, 128)
    idx, _ = vq(descriptors,voc)
    arr_of_count_clusters = np.bincount(idx)
    print ('Done')


    #test features
    im_features = np.zeros((len(image_paths), n_clusters), "float32")
    im_features_create(im_features, des_list, voc, image_paths)
    #print len(test_features)

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Train the Linear SVM
    clf = LinearSVC()
    clf.fit(im_features, np.array(image_classes))

    # Save the SVM
    joblib.dump((clf, training_names, stdSlr, n_clusters, voc), "bof.pkl", compress=3)

def image_test(image):

    # Load the classifier, class names, scaler, number of clusters and vocabulary
    clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

    test_set = 'test/'
    #if not os.path.exists(test_set +  description):
    #    os.makedirs(test_set + description)
    save_image_test(image, test_set + '/')
    image_paths = os.listdir(test_set)
    # testing_names = os.listdir(test_set)
    # for testing_name in testing_names:
    #     dir = os.path.join(test_set, testing_name)
    #     class_path = imutils.imlist(dir)
    #     image_paths+=class_path

    print (image_paths)

    # Create feature extraction and keypoint detector objects
    sift_object = cv2.xfeatures2d.SIFT_create()
    des_list = []
    create_dis_kp(image_paths, sift_object, des_list)

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    array_of_des(des_list, descriptors)

    #test features
    test_features = np.zeros((len(image_paths), k), "float32")
    im_features_create(test_features, des_list, voc, image_paths)

    # Scale the features
    test_features = scale_features(stdSlr, test_features)

    # Perform the predictions
    predictions =  [classes_names[i] for i in clf.predict(test_features)]
    #print predictions

    print (predictions)
    os.remove(test_set+image    )
    return predictions[0]

#image_train('1.jpg', 'borsch')
#image_test('1.jpg')
