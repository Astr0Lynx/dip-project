import numpy as np
import pytest
from docscan.types import (BinarizedDoc, Detection, annotation_from_dict, annotation_to_dict,
                           detection_from_dict, detection_to_dict)
from docscan.validate import ContractError, validate_binarized, validate_detection, validate_detections


def good_det(i=0):
    q = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    return Detection(det_id=i, quad=q, score=0.9, accepted=True, decided_by="ml", source="contour")


def test_detection_roundtrip():
    d = good_det()
    d2 = detection_from_dict(detection_to_dict(d))
    assert d2.det_id == d.det_id and np.allclose(d2.quad, d.quad) and d2.decided_by == "ml"


def test_validators_reject_bad_objects():
    d = good_det()
    d.quad = d.quad.astype(np.float64)
    with pytest.raises(ContractError):
        validate_detection(d)
    with pytest.raises(ContractError):
        validate_detections([good_det(0), good_det(0)])  # duplicate ids
    b = BinarizedDoc(det_id=0, binary=np.full((4, 4), 128, np.uint8), method="otsu")
    with pytest.raises(ContractError):
        validate_binarized(b)


def test_annotation_roundtrip():
    from docscan.types import SceneAnnotation, SceneObject
    q = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    a = SceneAnnotation("s", "scenes/s.jpg", 100, 80, "T0", 1,
                        [SceneObject(0, "document", "note", q, 0, 1.0, text="hi")])
    b = annotation_from_dict(annotation_to_dict(a))
    assert b.objects[0].text == "hi" and np.allclose(b.objects[0].quad, q)
