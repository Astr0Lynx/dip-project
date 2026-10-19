from docscan.config import load_config


def test_defaults_and_overrides():
    cfg = load_config()
    assert cfg.detect.max_side == 1000
    assert cfg.binarize.methods == ["otsu", "sauvola", "niblack", "tuned"]
    cfg2 = load_config(overrides={"detect.max_side": 800, "overlap.enabled": True})
    assert cfg2.detect.max_side == 800 and cfg2.overlap.enabled is True
    assert load_config().detect.max_side == 1000  # no cross-contamination
