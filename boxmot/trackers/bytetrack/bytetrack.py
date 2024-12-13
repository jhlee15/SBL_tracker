# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from collections import deque
import ultralytics
from boxmot.motion.kalman_filters.xyah_kf import KalmanFilterXYAH
from boxmot.trackers.bytetrack.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment
from boxmot.utils.ops import tlwh2xyah, xywh2tlwh, xywh2xyxy, xyxy2xywh
from boxmot.trackers.basetracker import BaseTracker


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, det, max_obs):
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.tlwh = xywh2tlwh(self.xywh)  # (xc, yc, w, h) --> (t, l, w, h)
        self.xyah = tlwh2xyah(self.tlwh)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs=max_obs
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
        self.history_observations = deque([], maxlen=self.max_obs)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xyah)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, a, h)
            ret[2] *= ret[3]  # (xc, yc, a, h)  -->  (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret


class ByteTrack(BaseTracker):
    """
    BYTETracker: A tracking algorithm based on ByteTrack, which utilizes motion-based tracking.

    Args:
        track_thresh (float, optional): Threshold for detection confidence. Detections above this threshold are considered for tracking in the first association round.
        match_thresh (float, optional): Threshold for the matching step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
        track_buffer (int, optional): Number of frames to keep a track alive after it was last detected. A longer buffer allows for more robust tracking but may increase identity switches.
        frame_rate (int, optional): Frame rate of the video being processed. Used to scale the track buffer size.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
    """
    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.8,
        track_buffer: int = 25,
        frame_rate: int = 30,
        per_class: bool = False,
    ):
        super().__init__(per_class=per_class)
        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.equip_info = {}
        self.reset_id()

        self.frame_id = 0
        self.track_buffer = track_buffer

        self.per_class = per_class
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYAH()

    def categories(self, cat_id) :
        cat = {0:"person",1:"hat",2:"glasses",3:"face_shield",4:"mask",5:"gloves",6:"suit",7:"boots",8:"integ"}
        cats = cat[cat_id]
        return cats
        
    # Track이 생성될때 실행 update 쪽에서 함수를 호출해 주면 된다.     
    def insert_equip_info(self, track_id):
        self.equip_info[track_id] = {"hat":False,"face_shield":False,"suit":False,"gloves":False,
                                      "mask":False,"boots":False,"glasses":False,"integ":False}


    # Track이 remove될때 실행 update 쪽에서 함수를 호출해 주면 된다.
    def remove_equip_info(self, track_id):
        self.equip_info.pop(track_id, None)

    # update 함수 이전이나 이후에 외부에서 함수를 호출해 착용 여부를 줄 수 있다.
    def track_set_equip(self, track_id, equipment):
        if equipment == 'integ' :
            self.equip_info[track_id]['hat'] = True
            self.equip_info[track_id]['face_shield'] = True
        else :
            self.equip_info[track_id][equipment] = True
            
    def overlap_equipment(self, cls, bboxes, xresult):          
        
        iou_matrix = []
        skip = False
        for i in range(len(xresult)):
            for j in range(i + 1, len(xresult)):
                iou = IoU(xresult[i][:4], xresult[j][:4])
                if iou > 0:
                    skip = True
                    break
            if skip:
                break

        if not skip:
            # Calculate IOUs between xresult boxes and all_box excluding cls 0
            for x_box in xresult:
                track_id = int(x_box[4])
                for j, bbox in enumerate(bboxes):
                    if cls[j] != 0:
                        iou = IoU(x_box[:4], bbox)
                        if iou > 0:
                            iou_matrix.append([track_id, cls[j], iou])
            
        for i in iou_matrix :
            i[1] = self.categories(i[1])
            
        result_dict = {}
        for item in iou_matrix:
            if item[2] > 0:
                key = item[0]
                if key not in result_dict:
                    result_dict[key] = []
                result_dict[key].append(item[1])
                
        return result_dict           
        
                   
    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self, dets, img: np.ndarray = None, embs: np.ndarray = None) -> np.ndarray:
        
        if isinstance(dets, ultralytics.engine.results.Boxes):
            dets = dets.data
        else :
            self.check_inputs(dets, img)

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        confs = dets[:, 4]
        
        _bboxes = dets[:,:4]
        _cls = dets[:,5]

        self.det_result = [i[:5] + [self.categories(int(i[5]))] for i in dets.tolist()]
               
        person_idx = [i for i,j in enumerate(_cls) if j == 0]
        confs = [confs[i] for i in person_idx]
        dets = [dets[i] for i in person_idx]
        
        dets = np.array(dets)
        confs = np.array(confs)

        remain_inds = confs > self.track_thresh

        inds_low = confs > 0.1
        inds_high = confs < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = dets[inds_second]
        dets = dets[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(det, max_obs=self.max_obs) for det in dets
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        # association the untrack to the low conf detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack(det_second, max_obs=self.max_obs) for det_second in dets_second]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_count)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                # remove equip info
                track_id = [str(int(i.id)) for i in removed_stracks]
                for t_idx in track_id :
                    self.remove_equip_info(t_idx)

        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )
        
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum
        
        # get confs of lost tracks
        output_stracks = [track for track in self.active_tracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)
        outputs = np.asarray(outputs)

        track_id = [str(int(t.id)) for t in output_stracks if track.is_activated]
        for t_idx in track_id :
            self.insert_equip_info(t_idx)
            
        result_dict = self.overlap_equipment(_cls, _bboxes, outputs)
        if result_dict :
            for e_id, equip in result_dict.items() :
                for eq in equip : 
                    self.track_set_equip(str(e_id), eq) 
                    
        return outputs
    
    def equip_info(self) :
        return self.equip_info
    
    def det_result(self) :
        return self.det_result
    
    @staticmethod
    def reset_id():
        """Resets the ID counter of STrack."""
        STrack.clear_count()

    def rest_equip(self):
        self.equip_info = {}

    def reset(self):
        """Reset tracker."""
        self.tracked_stracks = []
        self.lost_stracks = [] 
        self.removed_stracks = [] 
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYAH()
        self.equip_info = {}
        self.reset_id()

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb

def IoU(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if (inter_x_min < inter_x_max) and (inter_y_min < inter_y_max):
        intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        return 0

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
   
    return iou