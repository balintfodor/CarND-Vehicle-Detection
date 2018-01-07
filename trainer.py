import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import os

from skimage.color import rgb2gray, rgb2hsv, rgb2yuv
from skimage.feature import hog
from skimage.io import imread, imsave
from skimage.util import view_as_windows, view_as_blocks
from skimage.exposure import rescale_intensity, equalize_hist, equalize_adapthist
from skimage.transform import resize
from skimage.filters import sobel

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
        self.sample_feat_shape = None

        self.sample_size = (64, 64)
        self.pixels_per_cell = (16, 16)
        self.cells_per_block = (2, 2)
        self.orientations = 11
        self.color_hist_bins = 17

    def slide_feat_map(self, feat_map, step=1):
        wins = view_as_windows(feat_map, window_shape=self.sample_feat_shape, step=step)
        ss0, ss1 = self.sample_size
        ppc0, ppc1 = self.pixels_per_cell
        blocks = []
        for i in range(wins.shape[0]):
            for j in range(wins.shape[1]):
                coords = (i * ppc1, i * ppc1 + ss1, j * ppc0, j * ppc0 + ss0)
                blocks.append((wins[i, j], coords))
        return blocks


    def load(self, file):
        self.ss, self.clf, self.sample_feat_shape = joblib.load(file)

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

    def _hog_map(self, gray):
        hog_map = hog(gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=False)
        hog_map = hog_map.reshape((hog_map.shape[0], hog_map.shape[1], -1))
        return hog_map

    def _hist_map(self, gray, ranges=[0, 1]):
        gray = np.copy(gray)
        win_shape = (self.pixels_per_cell[1] * self.cells_per_block[1],
            self.pixels_per_cell[0] * self.cells_per_block[0])
        step = (self.pixels_per_cell[1], self.pixels_per_cell[0])
        blocks = view_as_windows(gray,
            window_shape=win_shape,
            step=step)
        hist_map = np.empty((blocks.shape[0], blocks.shape[1], 4 * self.color_hist_bins), dtype=gray.dtype)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                block = np.copy(blocks[i, j]).astype(np.float64)
                hist = np.histogram(block, bins=self.color_hist_bins, range=ranges)[0].astype(np.float64)

                hist /= np.linalg.norm(hist)
                hist = np.maximum(0.2, hist)
                hist /= np.linalg.norm(hist)

                hist_sorted = np.sort(hist)

                block_grad = sobel(block)
                grad_hist = np.histogram(block_grad, bins=self.color_hist_bins, range=ranges)[0].astype(np.float64)

                grad_hist /= np.linalg.norm(grad_hist)
                grad_hist = np.maximum(0.2, grad_hist)
                grad_hist /= np.linalg.norm(grad_hist)

                grad_hist_sorted = np.sort(grad_hist)

                hist_map[i, j] = np.concatenate((hist, hist_sorted, grad_hist, grad_hist_sorted), axis=0)
        return hist_map

    def image_to_feat_map(self, image):
        col_conv = rgb2hsv(image)
        feat_layers = []
        for i in range(3):
            hist_layer = self._hist_map(col_conv[:, :, i])
            hog_layer = self._hog_map(col_conv[:, :, i])
            feat_layers.append(hist_layer)
            feat_layers.append(hog_layer)
        feat_map = np.concatenate(feat_layers, axis=2)
        return feat_map

    def _batch_image_to_feat_map(self, samples):
        feats = []
        for image in tqdm(samples):
            feat = self.image_to_feat_map(image)
            feats.append(feat)
        return np.array(feats)

    def train(self, vehicle_images, non_vehicle_images):
        if os.path.exists('preprocessed.bin'):
            print('loading preprocessed.bin')
            X_train, X_test, y_train, y_test, self.sample_feat_shape = joblib.load('preprocessed.bin')
        else:
            print('vehicle samples')
            v_im = self._load_samples(vehicle_images)
            print('non-vehicle samples')
            nv_im = self._load_samples(non_vehicle_images)
            sample_images = np.r_[v_im, nv_im]
            print('preprocess')
            sample_feat_maps = self._batch_image_to_feat_map(sample_images)
            self.sample_feat_shape = sample_feat_maps.shape[1:]
            sample_feat_maps = sample_feat_maps.reshape((sample_feat_maps.shape[0], -1))

            labels = np.hstack((np.ones(v_im.shape[0]), np.zeros(nv_im.shape[0])))
            X_train, X_test, y_train, y_test = train_test_split(sample_feat_maps, labels, shuffle=True, test_size=0.2)

            joblib.dump([X_train, X_test, y_train, y_test, self.sample_feat_shape], 'preprocessed.bin')

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

        clf = BaggingClassifier(clf, n_estimators=5, max_samples=1, max_features=0.2)

        clf.fit(X_train, y_train)

        X_test = ss.transform(X_test)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        print('')
        print('train acc=', train_acc)
        print('test acc=', test_acc)

        joblib.dump([ss, clf, self.sample_feat_shape], 'model.bin')
        print('saved to model.bin')
        self.clf = clf

        xx = sigmoid(clf.decision_function(X_test))
        a = xx[y_test == 0]
        b = xx[y_test == 1]
        plt.scatter(a, range(a.shape[0]), color='r')
        plt.scatter(b, range(b.shape[0]), color='b')
        plt.show()

    def classify(self, feats):
        feats = self.ss.transform([feats.ravel()])
        return sigmoid(self.clf.decision_function(feats))[0]