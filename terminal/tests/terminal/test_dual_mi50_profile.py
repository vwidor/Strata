"""Exercise both hardware profiles with synthetic model paths."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.dual_mi50_profile import profile

TERMINAL = Path(__file__).resolve().parents[2]


def source(quant="IQ3_S"):
    return {
        "backend": "hip", "gpu": [1, 0],
        "args": ["--pack", "pack", "--native", f"Qwen3.8-Flash-Next-GSQ-RCO-{quant}-00001-of-00002.gguf",
                 "--ple-gguf", "table.gguf", "--mtp", "runtime"],
        "vision": {"gpu": True}, "env": {"EXAMPLE_SETTING": "value"},
    }


class DualMI50(unittest.TestCase):
    def test_mixed_profile_replaces_owned_values_and_keeps_model_paths(self):
        cfg = source()
        cfg["user_args"] = ["--batch-groups", "3", "--spec", "2", "--max-context", "16384",
                            "--conversation-cache-mib", "0", "--spec-min-p", "0.5"]
        out = profile(cfg, Path("trial.json"), split="24", port=18750, three_slot=True)
        self.assertEqual((out["parallel"], out["layer_split"], out["drafting"]), (3, "24", "mtp"))
        self.assertEqual(out["gpu"], [1, 0])
        self.assertEqual(out["vision"], cfg["vision"])
        self.assertEqual(out["env"]["EXAMPLE_SETTING"], "value")
        self.assertEqual(out["env"]["STRATA_HIP_PROMPT_F16"], "1")
        self.assertEqual(out["env"]["STRATA_PW_WITH_BATCH"], "1")
        args = out["args"]
        for flag, value in {"--batch-groups": "1", "--spec": "4", "--pipeline-windows": "2",
                            "--conversation-cache-mib": "4096", "--conversation-cache-slots": "8",
                            "--conversation-cache-min-tokens": "4096", "--max-context": "102400",
                            "--kv": "int8", "--vram-reserve-mib": "1400", "--mtp": "runtime"}.items():
            self.assertEqual(args.count(flag), 1)
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn("--spec-min-p", args)
        self.assertNotIn("user_args", out)

    def test_old_single_slot_modes(self):
        for pipeline in (False, True):
            out = profile(source("UD-IQ4_XS"), Path("test.json"), split="27", port=18750, pipeline=pipeline)
            self.assertEqual(out["parallel"], 1)
            self.assertEqual(out["args"][out["args"].index("--pipeline-windows")+1], "2" if pipeline else "0")
            self.assertNotIn("--conversation-cache-mib", out["args"])

    def test_refuses_wrong_quant_or_missing_runtime(self):
        missing_runtime = source()
        missing_runtime["args"] = missing_runtime["args"][:-2]
        for cfg in (source("IQ3_XXS"), missing_runtime):
            with self.assertRaises(ValueError):
                profile(cfg, Path("test.json"), split="24", port=18750, three_slot=True)

    def test_cli_from_either_layout_and_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "base.json"
            cfg = source()
            cfg["args"] = cfg["args"][:-2]
            base.write_text(json.dumps(cfg))
            for path in (TERMINAL, TERMINAL.parent):
                output = Path(folder) / (path.name + ".json")
                cmd = [sys.executable, str(path / "tools/dual_mi50_profile.py"), str(base), str(output),
                       "--three-slot", "--mtp-dir", folder]
                subprocess.run(cmd, check=True)
                contents = output.read_bytes()
                self.assertEqual(json.loads(contents)["host"], "127.0.0.1")
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
                self.assertEqual(output.read_bytes(), contents)
