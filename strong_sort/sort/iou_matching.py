# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        # for every track
        if tracks[track_idx].time_since_update > 35:  # modified, only half life tracks deserve IOU association
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()  # 1-dim,4 coordinate,np
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])  # 2-dim,1st represent the num of detection,2nd is 4 coordinate,np

        #########################
        # make bbox and candidates to B-IOU form(be aware tlwh)
        # bbox = get_biou_bboxes(bbox)
        # candidates = get_biou_bboxes(candidates)

        #########################

        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


def get_biou_bboxes(ori_bbox, buffer_rate=0.6, img_size=(1920, 1080)):
    """
    :param img_size:
    :param buffer_rate:
    :param ori_bbox: tlwh
    :return:
    """
    assert (ori_bbox.ndim == 1 or ori_bbox.ndim == 2)

    if ori_bbox.ndim == 1:
        # 1 * 4: x1,y1,w,h
        n_w = (1. + buffer_rate) * ori_bbox[2]
        n_h = (1. + buffer_rate) * ori_bbox[3]
        n_x = ori_bbox[0] - (n_w - ori_bbox[2]) / 2.
        n_y = ori_bbox[1] - (n_h - ori_bbox[3]) / 2.
        if n_x < 0:
            n_x = 0.
        if n_y < 0:
            n_y = 0.
        #
        return np.array([n_x, n_y, n_w, n_h])
    if ori_bbox.ndim == 2:
        # m * 4:
        n_w = (1. + buffer_rate) * ori_bbox[:, 2]
        n_h = (1. + buffer_rate) * ori_bbox[:, 3]
        n_x = ori_bbox[:, 0] - (n_w - ori_bbox[:, 2]) / 2.
        n_y = ori_bbox[:, 1] - (n_h - ori_bbox[:, 3]) / 2.
        return np.array([n_x, n_y, n_w, n_h]).T
