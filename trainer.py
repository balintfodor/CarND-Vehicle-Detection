import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import os

from skimage.color import rgb2gray, rgb2hsv, rgb2yuv
from skimage.feature import hog
from skimage.io import imread, imsave
from skimage.util import view_as_windows

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC, NuSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.externals import joblib

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        self.clf = None
        self.ss = None
        self.acc = 0

        self.sample_size = (64, 64)
        self.pixels_per_cell = (16, 16)
        self.cells_per_block = (2, 2)
        self.orientations = 11
        self.color_hist_bins = 9
        
    def search_window_size(self):
        return self.img_to_win(self.sample_size)

    def img_to_win(self, im_size):
        return tuple(im_size[i] // self.pixels_per_cell[i] - self.cells_per_block[i] + 1 for i in range(2))

    def win_to_img(self, yx):
        return tuple((yx[k] - 1 + self.cells_per_block[k]) * self.pixels_per_cell[k] for k in range(2))

    def load(self, file):
        self.ss, self.clf, self.acc = joblib.load(file)

    def _load_samples(self, image_list):
        if self.args.test:
            image_list = image_list[:300]

        images = []
        for f in tqdm(image_list):
            im = imread(f)
            im_flipped = im[:, ::-1, :]
            images.append(im)
            images.append(im_flipped)
        return np.array(images)

    def hog_map(self, gray):
        hog_map = hog(gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            transform_sqrt=False)
        blocks_row, blocks_col = self.img_to_win(gray.shape[:2])
        hog_map = hog_map.reshape((blocks_row, blocks_col, -1))
        return hog_map

    def preprocess(self, image):
        yuv = rgb2yuv(image)
        hm_y = self.hog_map(yuv[:, :, 0])
        hm_u = self.hog_map(yuv[:, :, 1])
        hm_v = self.hog_map(yuv[:, :, 2])
        feat_map = np.concatenate((hm_y, hm_u, hm_v), axis=2)
        return feat_map

    def _preprocess_list(self, samples):
        images = []
        print('preprocess')
        for image in tqdm(samples):
            images.append(self.preprocess(image))
        return np.array(images)

    def train(self, vehicle_images, non_vehicle_images):
        if os.path.exists('preprocessed.bin'):
            print('loading preprocessed.bin')
            X_train, X_test, y_train, y_test = joblib.load('preprocessed.bin')
        else:
            print('vehicle samples')
            v_im = self._load_samples(vehicle_images)
            print('non-vehicle samples')
            nv_im = self._load_samples(non_vehicle_images)
            samples = np.r_[v_im, nv_im]
            samples = self._preprocess_list(samples)
            samples = samples.reshape((samples.shape[0], -1))

            labels = np.hstack((np.ones(v_im.shape[0]), np.zeros(nv_im.shape[0])))
            X_train, X_test, y_train, y_test = train_test_split(samples, labels, shuffle=True, test_size=0.2)

            joblib.dump([X_train, X_test, y_train, y_test], 'preprocessed.bin')

        print('X_train.shape=', X_train.shape)

        ss = StandardScaler()
        ss.fit(X_train)

        X_train = ss.transform(X_train)

        print('training')

        clf = LinearSVC(
            C=self.args.C,
            dual=True,
            max_iter = 6000,
            fit_intercept=True,
            verbose=1)

        clf.fit(X_train, y_train)

        X_test = ss.transform(X_test)

        acc = clf.score(X_test, y_test)
        print('acc=', acc)

        joblib.dump([ss, clf, acc], 'model.bin')
        print('saved to model.bin')
        self.clf = clf

        xx = sigmoid(clf.decision_function(X_test))
        a = xx[y_test == 0]
        b = xx[y_test == 1]
        plt.scatter(a, range(a.shape[0]), color='r')
        plt.scatter(b, range(b.shape[0]), color='b')
        plt.show()

    def classify(self, feats, only_one=True):
        if only_one:
            feats = self.ss.transform([feats])
        else:
            feats = self.ss.transform(feats)
        return sigmoid(self.clf.decision_function(feats))