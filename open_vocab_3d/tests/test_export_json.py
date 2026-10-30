"""Tests for export_json.py: raw/reviewed schema, stable ids, review logic, validate.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_export_json.py -q
"""
import copy
import datetime as dt
import json
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import export_json as ej  # noqa: E402

REC = {
    "scan_id": "scan_20261007_141500_ab12", "source_file": "room_scan.usdz",
    "uploaded_at": "2026-10-07T14:15:00+05:30", "up_axis": "Y",
    "orientation": {"flip_applied": True, "source": "auto"},
    "scene_bounds": {"min": [-0.5, -1.33, -2.67], "max": [5.45, 1.79, 0.25]},
}


def _lift_obj(cls, conf, c, half=0.1, views=10):
    return {"class_name": cls, "confidence": conf,
            "position_world": {"x": c[0], "y": c[1], "z": c[2]},
            "bounding_box_3d": {"min": [x - half for x in c], "max": [x + half for x in c]},
            "num_views": views, "num_points": 100, "source_detections": ["p1_0000_00"]}


@pytest.fixture
def lift_json(tmp_path):
    objs = [
        _lift_obj("phone", 0.39, [1.7, -0.75, -1.16]),
        _lift_obj("laptop", 0.53751234, [1.10823456, -0.5063, -1.2053], 0.2, 59),
        _lift_obj("phone", 0.41, [1.4, -0.53, -1.63]),
        _lift_obj("decorative plate", 0.53, [1.46, 0.22, -1.81]),
        _lift_obj("table", 0.58, [1.08, -0.53, -1.37], 0.5, 60),
    ]
    p = tmp_path / "scene_objects.json"
    p.write_text(json.dumps({"scene_id": "x", "objects": objs}))
    return p


@pytest.fixture
def raw(lift_json):
    return ej.build_raw(REC, lift_json, "job1", ["table", "phone", "laptop", "decorative plate"],
                        {"focus_centres": [[1.04123456, -0.6, -1.3]]})


def test_raw_schema_and_ids(raw):
    assert ej.validate(raw) == []
    assert raw["schema_version"] == "1.0" and raw["status"] == "raw"
    ids = [o["object_id"] for o in raw["objects"]]
    assert ids == ["obj_001", "obj_002", "obj_003", "obj_004", "obj_005"]
    # sorted by class, then confidence descending
    assert [(o["class_name"], o["confidence"]) for o in raw["objects"]] == [
        ("decorative plate", 0.53), ("laptop", 0.5375), ("phone", 0.41), ("phone", 0.39),
        ("table", 0.58)]
    lap = raw["objects"][1]
    assert set(lap) == {"object_id", "class_name", "position", "bounding_box", "confidence",
                        "num_views"}
    assert lap["position"] == {"x": 1.1082, "y": -0.5063, "z": -1.2053}   # centroid, rounded
    assert lap["bounding_box"]["size"] == [0.4, 0.4, 0.4]
    assert lap["num_views"] == 59
    s = raw["scan"]
    assert s["reviewed_at"] is None and s["job_id"] == "job1"
    assert dt.datetime.fromisoformat(s["processed_at"]).tzinfo is not None
    cf = s["coordinate_frame"]
    assert cf["up_axis"] == "Y" and cf["units"] == "meters" and cf["handedness"] == "right"
    assert cf["orientation_fix"] == "rotate_x_180" and cf["orientation_source"] == "auto"
    assert s["pipeline"]["focus_centres"] == [[1.0412, -0.6, -1.3]]
    assert s["pipeline"]["two_pass"] is True
    assert s["scene_bounds"] == REC["scene_bounds"]
    assert raw["summary"] == {"num_objects": 5, "by_class": {
        "decorative plate": 1, "laptop": 1, "phone": 2, "table": 1}}
    assert "review_summary" not in raw


def test_raw_no_flip_frame(lift_json):
    rec = {**REC, "up_axis": "Z", "orientation": {"flip_applied": False, "source": "manual"}}
    d = ej.build_raw(rec, lift_json, "j", ["a"], {})
    cf = d["scan"]["coordinate_frame"]
    assert cf["orientation_fix"] == "none" and cf["orientation_source"] == "manual"
    assert cf["up_axis"] == "Z" and "+Z is up" in cf["note"]
    assert ej.validate(d) == []


def _decisions(raw, **over):
    out = []
    for o in raw["objects"]:
        d = {"object_id": o["object_id"], "decision": "accept", "class_name": o["class_name"]}
        d.update(over.get(o["object_id"], {}))
        out.append(d)
    return out


def test_reviewed(raw):
    before = copy.deepcopy(raw)
    rv = ej.build_reviewed(raw, _decisions(raw, obj_002={"decision": "reject"},
                                           obj_004={"class_name": "  mobile   phone "},
                                           obj_005={"decision": "reject", "class_name": ""}))
    assert raw == before                                   # input untouched
    assert ej.validate(rv) == []
    assert rv["status"] == "reviewed"
    assert dt.datetime.fromisoformat(rv["scan"]["reviewed_at"]).tzinfo is not None
    assert [o["object_id"] for o in rv["objects"]] == ["obj_001", "obj_003", "obj_004"]
    o4 = rv["objects"][2]
    assert o4["class_name"] == "mobile phone"
    assert o4["review"] == {"decision": "renamed", "original_class_name": "phone"}
    assert rv["objects"][0]["review"] == {"decision": "accepted",
                                          "original_class_name": "decorative plate"}
    assert rv["review_summary"] == {"accepted": 2, "renamed": 1,
                                    "rejected_object_ids": ["obj_002", "obj_005"]}
    assert rv["summary"] == {"num_objects": 3, "by_class": {
        "decorative plate": 1, "mobile phone": 1, "phone": 1}}
    assert ej.validate(raw) == []


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.pop(), "no decision"),
    (lambda d: d.append(dict(d[0])), "duplicate"),
    (lambda d: d[0].update(object_id="obj_999"), "unknown"),
    (lambda d: d[0].update(decision="maybe"), "decision"),
    (lambda d: d[0].update(class_name="   "), "empty"),
    (lambda d: d[0].update(class_name="x" * 101), "longer"),
])
def test_reviewed_rejects_bad_decisions(raw, mutate, match):
    d = _decisions(raw)
    mutate(d)
    with pytest.raises(ValueError, match=match):
        ej.build_reviewed(raw, d)


def test_reviewed_needs_raw(raw):
    rv = ej.build_reviewed(raw, _decisions(raw))
    with pytest.raises(ValueError):
        ej.build_reviewed(rv, _decisions(raw))


def test_validate_catches_problems(raw):
    def bad(fn):
        d = copy.deepcopy(raw)
        fn(d)
        return ej.validate(d)
    assert bad(lambda d: d.update(schema_version="2"))
    assert bad(lambda d: d["objects"][1].update(object_id="obj_001"))       # duplicate id
    assert bad(lambda d: d["objects"][0].update(object_id="7"))
    assert bad(lambda d: d["objects"][0].update(confidence=1.5))
    assert bad(lambda d: d["objects"][0].update(num_views=-1))
    assert bad(lambda d: d["objects"][0].update(extra=1))
    assert bad(lambda d: d["objects"][0]["bounding_box"].update(size=[9, 9, 9]))
    assert bad(lambda d: d["objects"][0]["position"].pop("z"))
    assert bad(lambda d: d["summary"].update(num_objects=99))
    assert bad(lambda d: d["scan"].update(processed_at="yesterday"))
    assert bad(lambda d: d["scan"].update(reviewed_at="2026-10-07T14:15:00+05:30"))
    assert bad(lambda d: d["scan"]["coordinate_frame"].update(units="feet"))
    assert bad(lambda d: d.update(review_summary={}))
    assert bad(lambda d: d["objects"][0].update(review={"decision": "accepted"}))
    assert ej.validate([]) == ["document is not an object"]
    rv = ej.build_reviewed(raw, _decisions(raw, obj_001={"decision": "reject"}))
    d = copy.deepcopy(rv)
    d["review_summary"]["rejected_object_ids"].append("obj_002")             # still present
    assert ej.validate(d)
    d = copy.deepcopy(rv)
    d["objects"][0].pop("review")
    assert ej.validate(d)
    d = copy.deepcopy(rv)
    d["objects"][0]["review"]["decision"] = "renamed"                       # not renamed
    assert ej.validate(d)


def test_write_json_atomic(tmp_path, raw):
    p = tmp_path / "a" / "b.json"
    ej.write_json(p, raw)
    assert json.loads(p.read_text()) == raw
    ej.write_json(p, {"x": 1})
    assert json.loads(p.read_text()) == {"x": 1}
    assert [f.name for f in p.parent.iterdir()] == ["b.json"]                # no temp files left


def test_empty_lift(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"objects": []}))
    d = ej.build_raw(REC, p, "j", ["a"], None)
    assert d["objects"] == [] and d["summary"] == {"num_objects": 0, "by_class": {}}
    assert ej.validate(d) == []
    rv = ej.build_reviewed(d, [])
    assert ej.validate(rv) == [] and rv["review_summary"]["rejected_object_ids"] == []
