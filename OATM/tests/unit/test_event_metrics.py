from oatm.evaluation.event_metrics import EventMetricsInputs, compute_event_metrics


def _row(track_id, cls, box, state="OBSERVED_STRONG"):
    return {"track_id": track_id, "class_name": cls, "state": state,
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}


def test_target_not_linked_returns_n_a_without_crashing():
    inputs = EventMetricsInputs(
        event_id="e1", method_name="sort", reference_box=(0, 0, 40, 40), reference_class="car",
        pre_frame_index=0, hidden_frame_indices=[1, 2], recovery_search_frame_indices=[3, 4],
        outputs_by_frame={0: [_row(9, "car", (500, 500, 540, 540))]},
    )
    result = compute_event_metrics(inputs)
    assert result.target_linked is False
    assert result.recovery_status == "n/a"
    assert result.hidden_frame_coverage is None


def test_fully_bridged_and_same_id_recovery_with_accurate_boxes():
    outputs_by_frame = {
        0: [_row(1, "car", (0, 0, 40, 40))],
        1: [_row(1, "car", (2, 2, 42, 42), state="PREDICTED_HIDDEN")],
        2: [_row(1, "car", (4, 4, 44, 44), state="PREDICTED_HIDDEN")],
        3: [_row(1, "car", (6, 6, 46, 46))],
    }
    gt_box_by_frame = {1: (2, 2, 42, 42), 2: (4, 4, 44, 44), 3: (6, 6, 46, 46)}
    inputs = EventMetricsInputs(
        event_id="e2", method_name="oatm_mvp", reference_box=(0, 0, 40, 40), reference_class="car",
        pre_frame_index=0, hidden_frame_indices=[1, 2], recovery_search_frame_indices=[3],
        outputs_by_frame=outputs_by_frame, gt_box_by_frame=gt_box_by_frame,
    )
    result = compute_event_metrics(inputs)
    assert result.target_linked is True
    assert result.hidden_frame_coverage == 1.0
    assert result.fully_bridged is True
    assert result.mean_center_error_px == 0.0
    assert result.mean_iou == 1.0
    assert result.recovery_status == "same_id"
    assert result.recovery_latency_frames == 0


def test_partial_bridge_when_track_disappears_mid_window():
    outputs_by_frame = {
        0: [_row(1, "car", (0, 0, 40, 40))],
        1: [_row(1, "car", (2, 2, 42, 42), state="PREDICTED_HIDDEN")],
        # frame 2: track 1 is gone entirely (terminated) -- not alive.
    }
    inputs = EventMetricsInputs(
        event_id="e3", method_name="static_memory", reference_box=(0, 0, 40, 40), reference_class="car",
        pre_frame_index=0, hidden_frame_indices=[1, 2], recovery_search_frame_indices=[3],
        outputs_by_frame=outputs_by_frame,
    )
    result = compute_event_metrics(inputs)
    assert result.n_hidden_frames_alive == 1
    assert result.hidden_frame_coverage == 0.5
    assert result.fully_bridged is False


def test_new_id_reassignment_counts_as_new_id_not_same_id():
    outputs_by_frame = {
        0: [_row(1, "car", (0, 0, 40, 40))],
        # track 1 never reappears; a brand new track 2 claims the object later.
        3: [_row(2, "car", (6, 6, 46, 46))],
    }
    gt_box_by_frame = {3: (6, 6, 46, 46)}
    inputs = EventMetricsInputs(
        event_id="e4", method_name="bytetrack", reference_box=(0, 0, 40, 40), reference_class="car",
        pre_frame_index=0, hidden_frame_indices=[1, 2], recovery_search_frame_indices=[3],
        outputs_by_frame=outputs_by_frame, gt_box_by_frame=gt_box_by_frame,
    )
    result = compute_event_metrics(inputs)
    assert result.recovery_status == "new_id"
    assert result.recovery_latency_frames == 0


def test_not_recovered_when_no_real_detection_reclaims_the_object():
    outputs_by_frame = {0: [_row(1, "car", (0, 0, 40, 40))]}
    gt_box_by_frame = {3: (6, 6, 46, 46), 4: (8, 8, 48, 48)}
    inputs = EventMetricsInputs(
        event_id="e5", method_name="yolo_only", reference_box=(0, 0, 40, 40), reference_class="car",
        pre_frame_index=0, hidden_frame_indices=[1, 2], recovery_search_frame_indices=[3, 4],
        outputs_by_frame=outputs_by_frame, gt_box_by_frame=gt_box_by_frame,
    )
    result = compute_event_metrics(inputs)
    assert result.recovery_status == "not_recovered"
    assert result.recovery_latency_frames is None
