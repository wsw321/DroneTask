import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import torch
import sys
import gdown
from os.path import exists as file_exists, join
import cv2

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .deep.reid_model_factory import show_downloadeable_models, get_model_url, get_model_name

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import download_url
# from fast_reid.fast_reid_interfece import FastReIDInterface


__all__ = ['StrongSORT']
cls2id = {
    'pedestrian': 0,
    'people': 1,
    'bicycle': 2,
    'car': 3,
    'van': 4,
    'truck': 5,
    'tricycle': 6,
    'awning-tricycle': 7,
    'bus': 8,
    'motor': 9
}

id2cls = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor'
}
def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep

def cv2_letterbox_image(image, expected_size=(128, 256)):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


class StrongSORT(object):
    def __init__(self, 
                 model_weights,
                 device, max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70, n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9,

                 # fastreid_vehicle_cfg='',
                 # fastreid_vehicle_weights='',
                 # fastreid_ped_cfg='',
                 # fastreid_ped_weights=''
                ):
        model_name = get_model_name(model_weights)
        model_url = get_model_url(model_weights)

        if not file_exists(model_weights) and model_url is not None:
            gdown.download(model_url, str(model_weights), quiet=False)
        elif file_exists(model_weights):
            pass
        elif model_url is None:
            print('No URL associated to the chosen DeepSort weights. Choose between:')
            show_downloadeable_models()
            exit()

        self.extractor = FeatureExtractor(
            # get rid of dataset information DeepSort model name
            model_name=model_name,
            model_path=model_weights,
            device=str(device)
        )
        # self.extractor_fastreid_vehicle = FastReIDInterface(fastreid_vehicle_cfg, fastreid_vehicle_weights, device)
        # self.extractor_fastreid_ped = FastReIDInterface(fastreid_ped_cfg, fastreid_ped_weights, device)

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)  # euclidean
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections

        features = self._get_features(bbox_xywh, ori_img, classes)  # Notice:modified,add class for CE
        #features = self._get_features_by_fastreid(bbox_xywh, ori_img, classes)  # support fast-reid
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()


        #######################################################################
        re_find_option_1 = False

        if re_find_option_1:
            """
            when run nms, original detections will be changed,
            so original feature/bbox_tlwh/classes/confidence should change.
            """
            # can optim by matrix computing
            akf_bbox_tlwh = [t.to_tlwh() for t in self.tracker.tracks if (t.time_since_update <= 1 and t.hits >10)]
            akf_classes = [t.class_id for t in self.tracker.tracks if (t.time_since_update <= 1 and t.hits >10)]
            akf_confidence = [0.5 for t in self.tracker.tracks if (t.time_since_update <= 1 and t.hits >10)]  # 0.5
            akf_detections = [Detection(t.to_tlwh(), t.conf, torch.from_numpy(t.last_det_feature[-1])) for t in self.tracker.tracks if (t.time_since_update <= 1 and t.hits >10)]

            # sum the bbox_tlwh/classes/confs/detections
            sum_bboxes = bbox_tlwh.numpy().tolist() + akf_bbox_tlwh
            sum_classes = classes.numpy().tolist() + akf_classes
            sum_confs = confidences.numpy().tolist() + akf_confidence
            sum_detections = detections + akf_detections

            # convert to [y1,x1,y2,x2]
            sum_bboxes = [self._tlwh_to_y1x1y2x2(row) for row in sum_bboxes]

            # re-built
            index = py_cpu_softnms(np.array(sum_bboxes, dtype=np.float32), np.array(sum_confs, dtype=np.float32))
            final_detections = [sum_detections[i] for i in index]
            final_classes = [sum_classes[i] for i in index]
            final_confidence = [sum_confs[i] for i in index]

            detections = final_detections
            classes = torch.from_numpy(np.array(final_classes))
            confidences = torch.from_numpy(np.array(final_confidence))


        ########################################################################


        # self.tracker.update(detections, classes, confidences)  # modified, add a param:ori_img
        self.tracker.update(detections, classes, confidences, ori_img)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _tlwh_to_y1x1y2x2(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return y1, x1, y2, x2

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img, classes):  # Notice: modified,add a parameter "classes"
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            # im = cv2_letterbox_image(im)
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
            # features = self.add_dimentions(features, classes)     # NOTICE: modified
        else:
            features = np.array([])
        return features

    # def _get_features_by_fastreid(self, bbox_xywh, ori_img, classes):
    #     dets = []
    #     for bbox in bbox_xywh:
    #         x1, y1, x2, y2 = self._xywh_to_xyxy(bbox)
    #         dets.append([x1, y1, x2, y2])
    #
    #     dets = np.array(dets)
    #
    #     # TODO: "split and merge"
    #     # pedestrian and people :0,1
    #     # others:2-9
    #     # make a mask list by classes
    #     if dets.any():
    #         mask_list = ( (classes[:] == 0) + (classes[:] == 1) )  # create mask, only 0 or 1 is True
    #         ped_dets, vehicle_dets = self._split_dets(dets, mask_list)
    #         if ped_dets.any():
    #             # ped_features = self.extractor_fastreid_ped.inference(ori_img, ped_dets)
    #             ped_features = (self._get_features(ped_dets, ori_img)).cpu().numpy()
    #         else:
    #             ped_features = np.array([])
    #
    #         if vehicle_dets.any():
    #             vehicle_features = self.extractor_fastreid_vehicle.inference(ori_img, vehicle_dets)
    #         else:
    #             vehicle_features = np.array([])
    #
    #         features = self._merge_features(ped_features, vehicle_features, mask_list)
    #     else:
    #         features = np.array([])
    #     return torch.from_numpy(features)

    def _split_dets(self, dets, mask_list):
        _dets = dets.copy()
        ped_dets, vehicle_dets = [], []
        for i in range(len(mask_list)):
            if mask_list[i]:
                ped_dets.append(_dets[i])
            else:
                vehicle_dets.append(_dets[i])
        return np.array(ped_dets), np.array(vehicle_dets)

    def _merge_features(self, ped_features, vehicle_features, mask_list):
        features = []
        i, j = 0, 0
        if not ped_features.any():
            features = vehicle_features
            return features
        if not vehicle_features.any():
            features = ped_features
            return features
        # impossible both have not

        for mask in mask_list:
            if mask:
                features.append(ped_features[i])
                i += 1
            else:
                features.append(vehicle_features[j])
                j += 1

        return np.array(features)

    def add_dimentions(self, features, classes):
        # for the number of classes, here is hard coded(10),if you want to change it, just make it a parameter
        assert features.shape[0] == classes.shape[0]
        eye = np.eye(10, 10).tolist()

        suffix_matrix = np.array([eye[int(i)] for i in classes.detach().numpy().tolist()])
        suffix_matrix = torch.from_numpy(suffix_matrix)
        suffix_matrix = suffix_matrix.to("cuda:0")

        features = torch.cat((features, suffix_matrix), 1)
        return features

