#!/usr/bin/env python3
"""Derive an isolated Flash-Next benchmark config for two MI50s."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_control import user_args


PROFILE_FLAGS = {
    "--layer-split", "--prefill", "--ple-io", "--pcie-frac",
    "--pipeline-windows", "--trim-stage-weights", "--no-trim-stage-weights",
    "--remote-expert-opt", "--split-skip-if-fits", "--batch", "--slots",
    "--batch-groups", "--batch-mtp", "--spec",
}


def profile(source, output, *, split, port, pipeline=False, three_slot=False, mtp_dir=None):
    if not isinstance(source, dict) or source.get("backend") != "hip":
        raise ValueError("Start with a HIP run config from the gfx906 build.")
    cards = source.get("gpu")
    if (not isinstance(cards, list) or len(cards) != 2
            or any(type(card) is not int or card < 0 for card in cards)
            or cards[0] == cards[1]):
        raise ValueError("The run config must select exactly two different GPU indices.")
    if source.get("replicas") or source.get("split_serve"):
        raise ValueError("Use a layer split, without replicas or separate prefill and decode engines.")

    args = user_args.merge(source.get("args"), source.get("user_args"))
    groups = user_args.groups(args)
    native = next((group[1] for group in groups if group[0] == "--native" and len(group) == 2), "")
    file_name = Path(native).name.lower()
    if three_slot:
        if pipeline:
            raise ValueError("The three-slot profile already enables the solo pipeline.")
        if not file_name.startswith("qwen3.8-flash-next-") or not (
                file_name.endswith("-iq3_s.gguf") or "-iq3_s-00001-of-" in file_name):
            raise ValueError("The three-slot profile requires a native Flash-Next IQ3_S model.")
        if not all(any(g[0] == flag and len(g) == 2 for g in groups) for flag in ("--pack", "--ple-gguf")):
            raise ValueError("Include the matching native --pack and --ple-gguf in the source config.")
    elif not (file_name.startswith("qwen3.8-flash-next-")
              and any(quant in file_name for quant in ("-ud-iq4_xs-", "-ud-q4_k_xl-"))):
        raise ValueError("Use the native Unsloth Flash-Next UD-IQ4_XS or UD-Q4_K_XL pack.")
    if mtp_dir is not None:
        args = user_args.merge(args, ["--mtp", str(mtp_dir)])
        groups = user_args.groups(args)
    if not any(group[0] == "--mtp" and len(group) == 2 for group in groups):
        raise ValueError("Supply a prepared MTP runtime in the source config or with --mtp-dir.")
    if {"--peer-device", "--split-device", "--draft"} & set(args):
        raise ValueError("Use an MTP layer-split config without custom device or draft flags.")

    flags = PROFILE_FLAGS
    if three_slot:
        flags = flags | {"--kv", "--max-context", "--vram-reserve-mib", "--lookup-max-t",
                         "--conversation-cache-slots", "--conversation-cache-mib",
                         "--conversation-cache-min-tokens", "--spec-min-p"}
    args = user_args.flat(group for group in groups if group[0] not in flags)
    args += ["--prefill", "2048", "--ple-io", "ram", "--pcie-frac", "0",
             "--trim-stage-weights", "--spec", "4" if three_slot else "3",
             "--pipeline-windows", "2" if three_slot or pipeline else "0"]
    env = {**(source.get("env") or {}), "STRATA_SPLIT_OVERLAP": "0"}
    if three_slot:
        args += ["--max-context", "102400", "--kv", "int8", "--vram-reserve-mib", "1400",
                 "--batch-groups", "1", "--lookup-max-t", "0", "--conversation-cache-mib", "4096",
                 "--conversation-cache-slots", "8", "--conversation-cache-min-tokens", "4096"]
        env.update(STRATA_HIP_PROMPT_F16="1", STRATA_PW_WITH_BATCH="1", STRATA_BATCH_YIELD_CHUNKS="1",
                   STRATA_ROUTE_TAIL_SKIP="0", STRATA_PIPED_CONCURRENT="0")
    config = dict(source, args=args, env=env, layer_split=split, gpu_order="as_given",
                  parallel=3 if three_slot else 1, drafting="mtp", mtp=True, remote_expert_opt=False,
                  split_skip_if_fits=False, host="127.0.0.1", port=port, log=str(output.with_suffix(".log")),
                  open_browser=False)
    user_args.stamp(config, [])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing Flash-Next HIP config with both MI50s selected")
    parser.add_argument("output", type=Path, help="new config file, in an existing directory")
    parser.add_argument("--split", choices=("24", "27", "auto"), default="24")
    parser.add_argument("--port", type=int, default=18750)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--three-slot", action="store_true", help="use the three-slot IQ3_S profile")
    parser.add_argument("--mtp-dir", type=Path, help="prepared MTP runtime directory for the target model")
    modes.add_argument("--pipeline", action="store_true", help="overlap two MTP verification windows")
    options = parser.parse_args()
    if not 1024 <= options.port <= 65535:
        parser.error("--port must be between 1024 and 65535")

    output = options.output.resolve()
    try:
        config = profile(json.loads(options.source.read_text(encoding="utf-8")), output,
                         split=options.split, port=options.port, pipeline=options.pipeline,
                         three_slot=options.three_slot,
                         mtp_dir=options.mtp_dir.resolve() if options.mtp_dir is not None else None)
        with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
