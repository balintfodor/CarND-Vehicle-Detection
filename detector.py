import numpy as np
from tqdm import tqdm

from skimage.io import imsave
from skimage.transform import pyramid_reduce, downscale_local_mean, rescale, resize
from skimage.exposure import rescale_intensity, equalize_adapthist, equalize_hist
from skimage.color import rgb2hsv
import cv2


def uint_im(im):
    return (im * 255).astype(np.uint8)

class Detector(object):
    def __init__(self, trainer, reader, args):
        self.trainer = trainer
        self.reader = reader
        self.args = args

    def process(self, out_dir):
        for idx, image in tqdm(enumerate(self.reader.next_frame()), total=self.reader.total):
            self._process_image(image, out_dir, idx)

    def _detect(self, image):
        feat_map = self.trainer.image_to_feat_map(image)
        blocks = self.trainer.slide_feat_map(feat_map)
        rois = []
        for patch, (ys, ye, xs, xe) in blocks:
            pred = self.trainer.classify(patch.ravel())
            if pred > 0.3:
                s = (ys, xs)
                e = (ye, xe)
                rois.append((s, e, pred))
        return rois

    def _coord_to_tuple(self, s, e):
        return (int(s[1]), int(s[0])), (int(e[1]), int(e[0]))

    def _process_image(self, image, out_dir, idx):
        cropped = image[368:656, :, :]
        roi_list = []
        for scale in np.linspace(1, 4, 8):
            scaled = rescale(cropped, 1 / scale, order=1, mode='constant')
            rois = self._detect(scaled)
            for s, e, pred in rois:
                roi_param = np.array([s[0], s[1], e[0], e[1], pred]).ravel()
                roi_param[:4] *= scale
                roi_list.append(roi_param)

        final_rois = self._select_final_rois(roi_list)
        for roi in final_rois:
            p1, p2 = self._coord_to_tuple(roi[0:2], roi[2:4])
            cv2.rectangle(cropped, p1, p2, (127, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cropped, '{:.4f}'.format(roi[4]), (p1[0]+2, p2[1]-2), font, 0.33, (255, 0, 0))
        imsave('{}/{:02d}.png'.format(out_dir, idx), cropped)

    def _select_final_rois(self, roi_list, overlap_th=0.2, min_sum_score=0.0):
        boxes = np.array(roi_list)

         # code based on: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        if boxes.shape[0] == 0:
            return []
    
        pick = []
    
        x1 = boxes[:, 1]
        y1 = boxes[:, 0]
        x2 = boxes[:, 3]
        y2 = boxes[:, 2]
    
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
    
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
    
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
    
            overlap = (w * h) / area[idxs[:last]]
            above_th = overlap > overlap_th
            if np.any(above_th):
                above_idxs = idxs[:last][above_th]
                boxes[i, 4] = np.sum(boxes[above_idxs, 4])
                # for k in range(4):
                    # boxes[i, k] = np.mean(boxes[above_idxs, k])

            wh = np.where(above_th)[0]
            idxs = np.delete(idxs, np.concatenate(([last], wh)))
    
        finals = boxes[pick]
        return finals[np.where(finals[:, 4] > min_sum_score)]