import numpy as np
from tqdm import tqdm
import os

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread, imsave
from skimage.util import view_as_windows

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.externals import joblib


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.clf = None
        self.a = 0
        self.b = 1
        self.acc = 0
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (2, 2)
        self.orientations = 9
        self.sample_block_h = 64 // self.pixels_per_cell[0] - self.cells_per_block[0] + 1
        self.sample_block_w = 64 // self.pixels_per_cell[1] - self.cells_per_block[1] + 1

    def load(self, file):
        self.a, self.b, self.clf, self.acc = joblib.load(file)

    def _load_samples(self, image_list):
        if self.args.test:
            image_list = image_list[:100]

        images = []
        for f in tqdm(image_list):
            images.append(imread(f))
        return np.array(images)

    def preprocess(self, image):
        gray = rgb2gray(image)
        
        hog_map = hog(gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys')
        blocks_row = image.shape[0] // self.pixels_per_cell[0] - self.cells_per_block[0] + 1
        blocks_col = image.shape[1] // self.pixels_per_cell[1] - self.cells_per_block[1] + 1
        hog_map = hog_map.reshape((blocks_row, blocks_col,
            self.cells_per_block[0], self.cells_per_block[1], 9))
        return hog_map

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
            X_train, X_test, y_train, y_test = train_test_split(samples, labels, shuffle=True)

            joblib.dump([X_train, X_test, y_train, y_test], 'preprocessed.bin')

        print(X_train.shape)

        a = np.mean(X_train, axis=0)
        X_train = X_train - a
        b = np.std(X_train, axis=0)
        X_train = X_train / b

        print('training')

        clf = VotingClassifier(
            [('et', ExtraTreeClassifier()),
            ('lsvm', RandomForestClassifier()),
            ('dt', DecisionTreeClassifier())],
            voting='soft')

        clf.fit(X_train, y_train)

        X_test -= a
        X_test /= b

        pred = np.c_[clf.predict(X_test) > 0.5, y_test]
        acc = np.sum(pred[:, 0] == pred[:, 1]) / pred.shape[0]
        print('acc=', acc)

        joblib.dump([a, b, clf, acc], 'model.bin')
        self.clf = clf

    def classify(self, feats):
        feats -= self.a
        feats /= self.b
        return self.clf.predict([feats])