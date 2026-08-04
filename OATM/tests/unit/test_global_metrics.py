from oatm.evaluation.global_metrics import compute_ghost_rate, compute_precision_recall


def _out(track_id, cls, box, state="OBSERVED_STRONG", sdt="f1", scene="s1"):
    return {"scene_token": scene, "sample_data_token": sdt, "track_id": track_id, "class_name": cls,
            "state": state, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}


def _gt(instance, cls, box, sdt="f1"):
    return {"sample_data_token": sdt, "instance_token": instance, "evaluation_class": cls, "box": box}


def test_precision_recall_perfect_match():
    outputs = [_out(1, "car", (0, 0, 40, 40))]
    gt_by_frame = {"f1": [_gt("i1", "car", (0, 0, 40, 40))]}
    result = compute_precision_recall(outputs, gt_by_frame)
    assert result["car"]["tp"] == 1
    assert result["car"]["precision"] == 1.0
    assert result["car"]["recall"] == 1.0


def test_precision_recall_false_positive_and_false_negative():
    outputs = [_out(1, "car", (500, 500, 540, 540))]  # nowhere near the real object
    gt_by_frame = {"f1": [_gt("i1", "car", (0, 0, 40, 40))]}
    result = compute_precision_recall(outputs, gt_by_frame)
    assert result["car"]["tp"] == 0
    assert result["car"]["fp"] == 1
    assert result["car"]["fn"] == 1
    assert result["car"]["precision"] == 0.0
    assert result["car"]["recall"] == 0.0


def test_precision_recall_ignores_predicted_hidden_rows():
    outputs = [_out(1, "car", (0, 0, 40, 40), state="PREDICTED_HIDDEN")]
    gt_by_frame = {"f1": [_gt("i1", "car", (0, 0, 40, 40))]}
    result = compute_precision_recall(outputs, gt_by_frame)
    assert result["car"]["tp"] == 0
    assert result["car"]["fn"] == 1, "a memory-only prediction is not a claim of current visibility"


def test_precision_recall_ignores_unmapped_classes():
    outputs = [_out(1, "truck", (0, 0, 40, 40))]
    gt_by_frame = {"f1": [_gt("i1", "car", (0, 0, 40, 40))]}
    result = compute_precision_recall(outputs, gt_by_frame)
    assert result["overall"]["fp"] == 0, "truck has no ground-truth class and must never be scored"


def test_ghost_rate_flags_track_with_zero_real_support():
    outputs = [_out(1, "car", (500, 500, 540, 540), sdt="f1"), _out(1, "car", (500, 500, 540, 540), sdt="f2")]
    gt_by_frame = {"f1": [_gt("i1", "car", (0, 0, 40, 40))], "f2": [_gt("i1", "car", (0, 0, 40, 40))]}
    result = compute_ghost_rate(outputs, gt_by_frame)
    assert result["n_tracks"] == 1
    assert result["n_ghost_tracks"] == 1
    assert result["ghost_rate"] == 1.0
    assert result["mean_ghost_duration_frames"] == 2


def test_ghost_rate_track_supported_even_once_is_not_a_ghost():
    outputs = [_out(1, "car", (0, 0, 40, 40), sdt="f1"), _out(1, "car", (500, 500, 540, 540), sdt="f2")]
    gt_by_frame = {"f1": [_gt("i1", "car", (0, 0, 40, 40))], "f2": [_gt("i1", "car", (0, 0, 40, 40))]}
    result = compute_ghost_rate(outputs, gt_by_frame)
    assert result["n_ghost_tracks"] == 0
