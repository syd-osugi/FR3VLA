"""
Test 33: DINOv2 grounding interface helpers.

These are no-model tests. They verify the request-label matching and mask
post-processing used by Working/vision/grounding_interface.py without importing
torch, transformers, or downloading a checkpoint.
"""

from __future__ import annotations

import numpy as np

from _working_test_utils import require, require_close, run_tests

from vision.grounding_interface import (
    DINOv2SegmentationDetector,
    component_detections_from_mask,
    expand_terms_with_aliases,
    label_matches_terms,
    parse_label_aliases,
    request_terms,
)


def test_request_terms_drop_color_and_action_words():
    terms = request_terms("please locate the red coffee mug")
    require("please" not in terms, "request terms should drop polite/action words")
    require("red" not in terms, "request terms should drop color token terms")
    require("coffee" in terms, "request terms should keep object modifiers")
    require("mug" in terms, "request terms should keep object noun")


def test_label_matching_and_aliases():
    aliases = parse_label_aliases({"mug": ["cup", "coffee cup"]})
    terms = expand_terms_with_aliases(request_terms("red mug"), aliases)
    require("cup" in terms, "aliases should add alternate label words")
    require(label_matches_terms("coffee cup", terms), "coffee cup label should match mug request")
    require(not label_matches_terms("chair", terms), "unrelated label should not match")


def test_component_detections_from_mask():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:10, 6:16] = 1
    confidence = np.zeros((20, 30), dtype=float)
    confidence[mask.astype(bool)] = 0.75

    detections = component_detections_from_mask(
        mask,
        confidence,
        label="block",
        label_id=3,
        requested_class="block",
        min_area_px=5,
        model_name="unit-test-model",
    )

    require(len(detections) == 1, f"expected one component, got {len(detections)}")
    detection = detections[0]
    require(detection["bbox_xywh"] == [6, 4, 10, 6], "bbox_xywh was incorrect")
    require(detection["bbox_xyxy"] == [6, 4, 15, 9], "bbox_xyxy was incorrect")
    require(detection["area_px"] == 60, "area should equal mask pixels")
    require_close(detection["score"], 0.75, "confidence should average over component")


def test_detector_constructs_without_loading_model():
    detector = DINOv2SegmentationDetector(
        model_name="fake-local-checkpoint",
        label_aliases={"mug": ["cup"]},
    )
    require(detector.model_name == "fake-local-checkpoint", "model name should be stored")
    require(detector._runtime is None, "constructor should not load heavy model runtime")


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("request terms", test_request_terms_drop_color_and_action_words),
                ("label matching aliases", test_label_matching_and_aliases),
                ("component detections", test_component_detections_from_mask),
                ("lazy detector construction", test_detector_constructs_without_loading_model),
            ]
        )
    )
