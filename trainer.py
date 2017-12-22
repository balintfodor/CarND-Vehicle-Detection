import numpy as np
from tqdm import tqdm

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread, imsave
from skimage.util import view_as_windows

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.clf = None
        self.a = 0
        self.b = 1
        self.acc = 0

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
        cells_per_block = (1, 1)
        hog_map = hog(gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=cells_per_block, block_norm='L2')
        blocks_row = image.shape[0] // 8 - cells_per_block[0] // 2
        blocks_col = image.shape[1] // 8 - cells_per_block[1] // 2
        hog_map = hog_map.reshape((blocks_row, blocks_col, 1, 1, 9))
        return hog_map

    def _preprocess_list(self, samples):
        images = []
        for image in samples:
            images.append(self.preprocess(image))
        return np.array(images)

    def generate_rois(self, hog_map):
        wins = view_as_windows(hog_map, (8, 8, 1, 1, 9), step=1)
        return wins

    def train(self, vehicle_images, non_vehicle_images):
        v_im = self._load_samples(vehicle_images)
        nv_im = self._load_samples(non_vehicle_images)
        samples = np.r_[v_im, nv_im]
        samples = self._preprocess_list(samples)
        samples = samples.reshape((samples.shape[0], -1))
        print(samples.shape)

        labels = np.hstack((np.ones(v_im.shape[0]), np.zeros(nv_im.shape[0])))
        X_train, X_test, y_train, y_test = train_test_split(samples, labels, shuffle=True)

        a = np.mean(X_train, axis=0)
        X_train = X_train - a
        b = np.std(X_train, axis=0)
        X_train = X_train / b

        n_estimators = 4
        clf = BaggingClassifier(SVC(kernel='rbf', C=1, gamma=0.1, class_weight='balanced', probability=False), 
                max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=4)
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
        return self.clf.predict(feats)