import json, os
try:
    import pytest
except ImportError:   # allow `python -m unittest` without pytest installed
    import contextlib, types
    @contextlib.contextmanager
    def _raises(exc):
        info = types.SimpleNamespace(value=None)
        try:
            yield info
        except exc as e:
            info.value = e
        else:
            raise AssertionError(f"{exc.__name__} not raised")
    pytest = types.SimpleNamespace(raises=_raises)
from src.pipeline import final
from src.planner.llm import make_client


def test_profiles_exist_and_point_to_real_files():
    for n in ("reduced", "full", "pilot"):
        p = final.profile(n)
        assert os.path.isdir(p["scenario_dir"]) and os.path.exists(p["exp_config"])


def test_unknown_profile_rejected():
    with pytest.raises(SystemExit):
        final.profile("nope")


def test_estimate_reduced_is_positive_and_consistent():
    e = final.estimate(final.profile("reduced"))
    assert e["n_scenarios"] == 34 and e["static_worst_case_calls"] > 0
    assert e["total_worst_case_calls"] == e["static_worst_case_calls"] + e["dynamic_worst_case_calls"]


def test_follower_client_is_mock_and_deterministic():
    c = make_client({"provider": "follower", "model": "x"})
    assert c.is_mock


def test_preflight_passes_for_follower():
    msgs = []
    assert final.preflight("follower", "follower-dryrun-v1", "pilot", say=msgs.append)


def test_code_change_is_refused(tmp_path):
    with open(tmp_path / "state.json", "w") as f:
        json.dump({"code_sha256": "0" * 64, "sessions": 1, "static_done": False, "dynamic_done": False, "static_dir": None, "dynamic_dir": None}, f)
    with pytest.raises(SystemExit) as ei:
        final.run_final("p", "m", "pilot", run_dir=str(tmp_path))
    assert "code/config changed" in str(ei.value)


def test_state_dir_is_filesystem_safe():
    assert "/" not in final.state_dir("a/b", "c:d e", "reduced").split("final")[-1].strip("/\\").replace("__", "")
