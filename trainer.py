import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import os

from skimage.color import rgb2gray, rgb2hsv
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
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (2, 2)
        self.orientations = 9
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

    def color_hist_map(self, hsv):
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        cell_map = np.zeros((hsv.shape[0] // self.pixels_per_cell[0], 
            hsv.shape[1] // self.pixels_per_cell[1],
            3, self.color_hist_bins))
        for i in range(cell_map.shape[0]):
            for j in range(cell_map.shape[1]):
                roi_H = H[i * self.pixels_per_cell[0], j * self.pixels_per_cell[1]]
                roi_S = S[i * self.pixels_per_cell[0], j * self.pixels_per_cell[1]]
                roi_V = V[i * self.pixels_per_cell[0], j * self.pixels_per_cell[1]]
                cell_map[i, j, 0] = np.histogram(roi_H, self.color_hist_bins, (0, 1))[0]
                cell_map[i, j, 1] = np.histogram(roi_S, self.color_hist_bins, (0, 1))[0]
                cell_map[i, j, 2] = np.histogram(roi_V, self.color_hist_bins, (0, 1))[0]

        hist_map = np.zeros((cell_map.shape[0] - self.cells_per_block[0] + 1,
            cell_map.shape[1] - self.cells_per_block[1] + 1,
            self.cells_per_block[0], self.cells_per_block[1],
            3, self.color_hist_bins))

        k = 1 / (self.cells_per_block[0] * self.cells_per_block[1])
        for i in range(hist_map.shape[0]):
            for j in range(hist_map.shape[1]):
                el_sum = 0
                ys = np.minimum(i + self.cells_per_block[0], hist_map.shape[0]) - i
                xs = np.minimum(j + self.cells_per_block[1], hist_map.shape[1]) - j
                for y in range(ys):
                    for x in range(xs):
                        el_sum += cell_map[i + y, j + x] ** 2

                el_sum =  1 / np.maximum(np.sqrt(el_sum), 10e-4)
                for y in range(ys):
                    for x in range(xs):
                        hist_map[i + y, j + x, y, x] = el_sum * cell_map[i + y, j + x]

        return hist_map.reshape((hist_map.shape[0], hist_map.shape[1], -1))

    def hog_map(self, gray):
        hog_map = hog(gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            transform_sqrt=True)
        blocks_row, blocks_col = self.img_to_win(gray.shape[:2])
        hog_map = hog_map.reshape((blocks_row, blocks_col, -1))
        return hog_map

    def preprocess(self, image):
        hsv = rgb2hsv(image)
        cm_hsv = rgb2hsv(image - np.mean(image, axis=(0, 1)))
        cm = self.color_hist_map(cm_hsv)
        # hm_h = self.hog_map(hsv[:, :, 0])
        # hm_s = self.hog_map(hsv[:, :, 1])
        hm_v = self.hog_map(cm_hsv[:, :, 2])
        hm_s = self.hog_map(cm_hsv[:, :, 1])
        feat_map = np.concatenate((cm, hm_v, hm_s), axis=2)
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
            X_train, X_test, y_train, y_test = train_test_split(samples, labels, shuffle=True, test_size=0.1)

            joblib.dump([X_train, X_test, y_train, y_test], 'preprocessed.bin')

        print(X_train.shape)

        ss = StandardScaler()
        ss.fit(X_train)

        X_train = ss.transform(X_train)
        print(np.min(X_train), np.max(X_train), np.mean(X_train), np.std(X_train))

        print('training')

        clf = LinearSVC(
            C=self.args.C,
            dual=True,
            max_iter = 1000,
            fit_intercept=False,
            verbose=1)

        # clf = SVC(
        #     C=self.args.C,
        #     gamma=0.01,
        #     kernel='poly',
        #     degree=5,
        #     max_iter=1500,
        #     verbose=1)

        clf.fit(X_train, y_train)

        X_test = ss.transform(X_test)

        # pred = np.c_[clf.predict(X_test) > 0.5, y_test]
        # acc = np.sum(pred[:, 0] == pred[:, 1]) / pred.shape[0]
        # print('acc=', acc)
        acc = clf.score(X_test, y_test)
        print('acc=', acc)

        joblib.dump([ss, clf, acc], 'model.bin')
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