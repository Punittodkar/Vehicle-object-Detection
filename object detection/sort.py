import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    wh = w * h
    return wh / (
        (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
        (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6
    )


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R *= 10.
        self.kf.P *= 10.
        self.kf.Q *= 0.01

        self.kf.x[:4] = bbox.reshape((4,1))
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(bbox.reshape((4,1)))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))


class Sort:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.trackers = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, detections):
        predicted_boxes = np.array([t.predict() for t in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, predicted_boxes, self.iou_threshold
        )

        for d, t in matched:
            self.trackers[t].update(detections[d])

        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d]))

        results = []
        for t in self.trackers[:]:
            if t.time_since_update <= self.max_age:
                results.append([*t.kf.x[:4].flatten(), t.id])
            else:
                self.trackers.remove(t)

        return np.array(results) if len(results) else np.empty((0,5))


def associate_detections_to_trackers(detections, trackers, iou_threshold):
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d in range(len(detections)):
        for t in range(len(trackers)):
            iou_matrix[d, t] = iou(detections[d], trackers[t])

    row_idx, col_idx = linear_sum_assignment(-iou_matrix)
    matches = []

    for r, c in zip(row_idx, col_idx):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append([r, c])

    matches = np.array(matches, dtype=int) if len(matches) else np.empty((0,2), dtype=int)

    unmatched_dets = [d for d in range(len(detections)) if d not in matches[:,0]] if len(matches) else list(range(len(detections)))
    unmatched_trks = [t for t in range(len(trackers)) if t not in matches[:,1]] if len(matches) else list(range(len(trackers)))

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
