"""Executes app.py end-to-end against a stub of the streamlit module (streamlit itself cannot be installed offline).
It catches runtime errors in the script (wrong keys, bad calls into src/), not visual problems."""
import runpy, sys, types, unittest
from unittest.mock import MagicMock


class SS(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def make_stub(press: bool):
    st = MagicMock()
    st.session_state = SS()
    st.selectbox.side_effect = lambda label, options, *a, **k: options[k.get("index", 0)] if label != "Provider" else "mock"
    st.text_input.side_effect = lambda label, value="", **k: value
    st.text_area.side_effect = lambda label, value="", **k: value
    st.number_input.side_effect = lambda label, mn=0, mx=0, value=0, **k: value
    st.slider.side_effect = lambda label, lo, hi, value=None, **k: value if value is not None else lo
    st.checkbox.side_effect = lambda label, value=False, **k: value
    st.button.side_effect = lambda *a, **k: press
    st.columns.side_effect = lambda spec, **k: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    st.stop.side_effect = SystemExit
    return st


class TestAppStub(unittest.TestCase):
    def _run(self, press):
        st = make_stub(press)
        class Mod(types.ModuleType):
            def __getattr__(self, k):
                return getattr(st, k)
        mod = Mod("streamlit")
        old = sys.modules.get("streamlit")
        sys.modules["streamlit"] = mod
        try:
            runpy.run_path("app.py", run_name="__main__")
        finally:
            if old is not None:
                sys.modules["streamlit"] = old
            else:
                sys.modules.pop("streamlit", None)
        return st

    def test_renders_idle_and_after_run(self):
        self._run(False)
        st = self._run(True)
        self.assertTrue(st.pyplot.called)                   # map history drawn after a run
        self.assertTrue(st.tabs.called and st.json.called)
        html = " ".join(str(c.args[0]) for c in st.markdown.call_args_list if c.args)
        self.assertIn("MOCK PLANNER", html)                  # mock banner always visible
        self.assertIn("strip", html)                         # revision strips rendered


if __name__ == "__main__":
    unittest.main()
