import numpy as np
from sklearn.cluster import KMeans
import cv2
from scipy.cluster.vq import vq
from webbrowser import open
from os import listdir

def create_img_kp(key_p, img):
    """
    Get coordinates of key point, and create patches 
    """
    const_size = 128
    x = int(key_p.pt[0])
    y = int(key_p.pt[1])
    img1_new = img[x:const_size+x, y:const_size+y]
    list_of_img_kp.append(img1_new)
    
    
def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]


def ClusterIndicesComp(clustNum, labels_array): #list comprehension
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])


def compute_detect_kp_descr(img_name):
    """
    read images, create SIFT object, detect key points, descriptors. Create array of key_points
    """
    img = cv2.imread(img_name, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    des_all.append(des)
    for i in range(len(kp)):
        create_img_kp(kp[i], img)
    
def write_img(name, ind):
    """
    write the most common part of images
    """
    cv2.imwrite(name, ind)


des_all = []
list_of_img_kp = []
img_path = 'dataset/train/dota/'
list_of_files = os.listdir(img_path)
list_of_files
for i in list_of_files:
    compute_detect_kp_descr(img_path + i)
print '-_-'
descriptors_ = des_all[0][0]
for i in des_all:
    for descriptor in i:
        descriptors_ = np.vstack((descriptors_, descriptor))
        descriptors = np.float32(descriptors_)
descriptors = np.delete(descriptors_, 1, 0)
print '=)'


n_clusters = 250
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(descriptors,n_clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
idx,_ = vq(descriptors,center)
print 'Done!'

arr_of_all = np.bincount(idx)
print(arr_of_all)
index_of_max = np.argmax(arr_of_all)
print(index_of_max)
list_ind_des = ClusterIndicesComp(index_of_max,label)


path_img = 'result/desc'
for i in range(1, 5):
    write_img(path_img + str(i) + '.jpg', list_of_img_kp[list_ind_des[i*3]])
webbrowser.open('file:///home/settnozz/Desktop/5_semestr/git_img/result/start.html')
print 'Well!'

