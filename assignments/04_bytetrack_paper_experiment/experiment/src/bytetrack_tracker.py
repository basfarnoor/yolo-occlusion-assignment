"""ByteTrack paper reference: Zhang et al., "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box," ECCV 2022 (https://arxiv.org/abs/2110.06864).

Method C: the educational two-stage BYTE association tracker. This is the
paper's central contribution, built on the same predict/associate/correct/
manage-lifecycle loop as sort_tracker.py's Method B baseline (both share
kalman_box_tracker.py and assignment.py), so the *only* difference between
the two methods is this class's second association round:

  for every frame:
    predict every existing track forward (real dt)
    split this frame's detections into high-score and low-score groups
      (below detection_floor was already filtered out upstream -- never seen)
    STAGE 1 -- match high-score detections against every predicted track
    STAGE 2 -- match low-score detections against only the tracks STAGE 1
               left unmatched (the paper's "give unmatched tracks a second,
               harder-evidence chance" idea)
    update every track matched in either stage
    tracks unmatched in both stages remain in "motion_prediction" state and
      are removed once time_since_update exceeds track_buffer
    new tracks are born only from unmatched high-score detections that clear
      new_track_threshold -- an unmatched low-score detection is discarded,
      never starting a track (a weak, never-associated box is too likely to
      be background noise or a false detection to trust as a brand-new object)

This is not a copy of the authors' ByteTracker source file -- it is a small,
from-scratch reimplementation of the paper's algorithm built on this
assignment's own Kalman filter and Hungarian-assignment modules.
"""
from __future__ import annotations

from assignment import associate_detections_to_trackers
from kalman_box_tracker import KalmanBoxTracker
from track import EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE, EVIDENCE_PREDICTION, TrackOutput


class ByteTrackTracker:
    def __init__(self, detection_floor: float = 0.05, high_score_threshold: float = 0.5,
                 new_track_threshold: float = 0.6, first_association_iou_threshold: float = 0.3,
                 second_association_iou_threshold: float = 0.5, track_buffer: int = 5,
                 min_hits: int = 1):
        self.detection_floor = detection_floor
        self.high_score_threshold = high_score_threshold
        self.new_track_threshold = new_track_threshold
        self.first_iou = first_association_iou_threshold
        self.second_iou = second_association_iou_threshold
        self.track_buffer = track_buffer
        self.min_hits = min_hits

        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self.last_timestamp: float | None = None

    def update(self, detections: list[dict], timestamp: float = 0.0) -> list[TrackOutput]:
        """detections: list of {"class","confidence","x1","y1","x2","y2"}, any
        confidence >= detection_floor (weaker boxes should already have been
        dropped upstream by the detector's own floor, but this method also
        defensively ignores anything below detection_floor)."""
        self.frame_count += 1
        dt = 1.0 if self.last_timestamp is None else max(timestamp - self.last_timestamp, 1e-6)
        self.last_timestamp = timestamp

        detections = [d for d in detections if d.get("confidence", 0.0) >= self.detection_floor]
        high_dets = [d for d in detections if d["confidence"] >= self.high_score_threshold]
        low_dets = [d for d in detections if d["confidence"] < self.high_score_threshold]

        predicted_boxes = [t.predict(dt) for t in self.trackers]
        predicted_classes = [t.class_name for t in self.trackers]
        n_trk = len(self.trackers)

        evidence_this_frame: dict[int, str] = {}

        # --- Stage 1: high-confidence detections vs. every predicted track ---
        matches1, unmatched_high, unmatched_trk_idx = associate_detections_to_trackers(
            high_dets, predicted_boxes, predicted_classes, self.first_iou)

        for det_idx, trk_idx in matches1:
            det = high_dets[det_idx]
            self.trackers[trk_idx].update((det["x1"], det["y1"], det["x2"], det["y2"]))
            self.trackers[trk_idx].last_timestamp = timestamp
            self.trackers[trk_idx].last_confidence = det.get("confidence", 0.0)
            evidence_this_frame[trk_idx] = EVIDENCE_HIGH_SCORE

        # --- Stage 2: low-confidence detections vs. tracks stage 1 left unmatched ---
        remaining_boxes = [predicted_boxes[i] for i in unmatched_trk_idx]
        remaining_classes = [predicted_classes[i] for i in unmatched_trk_idx]
        matches2, unmatched_low, unmatched_remaining_local = associate_detections_to_trackers(
            low_dets, remaining_boxes, remaining_classes, self.second_iou)

        for det_idx, local_trk_idx in matches2:
            trk_idx = unmatched_trk_idx[local_trk_idx]
            det = low_dets[det_idx]
            self.trackers[trk_idx].update((det["x1"], det["y1"], det["x2"], det["y2"]))
            self.trackers[trk_idx].last_timestamp = timestamp
            self.trackers[trk_idx].last_confidence = det.get("confidence", 0.0)
            evidence_this_frame[trk_idx] = EVIDENCE_LOW_SCORE
        # unmatched_low (leftover low-score detections) are discarded here --
        # never used to birth a track, per the paper's design.

        # --- Track birth: only from high-score detections still unmatched after stage 1 ---
        for det_idx in unmatched_high:
            det = high_dets[det_idx]
            if det.get("confidence", 0.0) < self.new_track_threshold:
                continue
            new_tracker = KalmanBoxTracker(
                (det["x1"], det["y1"], det["x2"], det["y2"]), det["class"], timestamp)
            new_tracker.hits = 1
            new_tracker.hit_streak = 1
            new_tracker.time_since_update = 0
            new_tracker.last_confidence = det.get("confidence", 0.0)
            evidence_this_frame[len(self.trackers)] = EVIDENCE_HIGH_SCORE
            self.trackers.append(new_tracker)

        # --- Lifecycle: remove tracks unmatched (in either stage) beyond the buffer ---
        keep_indices = [i for i, t in enumerate(self.trackers) if t.time_since_update <= self.track_buffer]
        kept_trackers = [self.trackers[i] for i in keep_indices]
        kept_evidence = {new_i: evidence_this_frame.get(old_i, EVIDENCE_PREDICTION)
                          for new_i, old_i in enumerate(keep_indices)}
        self.trackers = kept_trackers

        outputs = []
        for i, t in enumerate(self.trackers):
            confirmed = t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            outputs.append(TrackOutput(
                track_id=t.id,
                box=t.current_box(),
                class_name=t.class_name,
                evidence_source=kept_evidence[i],
                hits=t.hits,
                hit_streak=t.hit_streak,
                time_since_update=t.time_since_update,
                age=t.age,
                confirmed=confirmed,
                velocity=t.velocity(),
                confidence=t.last_confidence,
                raw_detection_box=t.last_raw_box if kept_evidence[i] != EVIDENCE_PREDICTION else None,
            ))
        return outputs
