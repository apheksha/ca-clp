"""Regression test: every analysis tool must start when called as `python tools/<name>.py` from the repo root (and as a module)."""
import os, subprocess, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = ["kinematic_audit", "energy_sensitivity", "dynamic_summary", "diversity_probe", "rebuild_airport_scenarios"]


class ToolsCli(unittest.TestCase):
    def test_help_runs_as_script_and_module(self):
        for t in TOOLS:
            for cmd in ([sys.executable, f"tools/{t}.py", "--help"], [sys.executable, "-m", f"tools.{t}", "--help"]):
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, f"{' '.join(cmd[1:])}: {r.stderr[-300:]}")


if __name__ == "__main__":
    unittest.main()
