from oatm.evaluation.linking import find_track_id_for_reference_box, resolve_instance_token


def test_find_track_id_picks_best_same_class_overlap():
    frame_outputs = [
        {"track_id": 1, "class_name": "car", "x1": 0, "y1": 0, "x2": 40, "y2": 40},
        {"track_id": 2, "class_name": "person", "x1": 1, "y1": 1, "x2": 41, "y2": 41},
        {"track_id": 3, "class_name": "car", "x1": 100, "y1": 100, "x2": 140, "y2": 140},
    ]
    result = find_track_id_for_reference_box(frame_outputs, (0, 0, 40, 40), "car")
    assert result == 1, "must prefer same-class high-overlap track, not the higher-IoU wrong-class one"


def test_find_track_id_returns_none_below_threshold():
    frame_outputs = [{"track_id": 1, "class_name": "car", "x1": 300, "y1": 300, "x2": 340, "y2": 340}]
    assert find_track_id_for_reference_box(frame_outputs, (0, 0, 40, 40), "car") is None


def test_resolve_instance_token_maps_detector_class_to_eval_class():
    frame_gt = [
        {"instance_token": "ped-1", "evaluation_class": "pedestrian", "box": (0, 0, 40, 40)},
        {"instance_token": "car-1", "evaluation_class": "car", "box": (100, 0, 140, 40)},
    ]
    result = resolve_instance_token(frame_gt, (0, 0, 40, 40), detector_class="person")
    assert result == "ped-1"


def test_resolve_instance_token_none_for_unmapped_class():
    frame_gt = [{"instance_token": "car-1", "evaluation_class": "car", "box": (0, 0, 40, 40)}]
    assert resolve_instance_token(frame_gt, (0, 0, 40, 40), detector_class="truck") is None
