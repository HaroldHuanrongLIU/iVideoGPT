from __future__ import annotations

from typing import Optional, Sequence

from benchmark.surgwmbench import BaselineSpec, run_cli


SPEC = BaselineSpec(
    baseline="ivideogpt",
    model="iVideoGPT",
    native_entrypoint="inference/predict.py",
    native_train_entrypoint="train_gpt.py",
    native_frame_predictor=(
        "upstream inference/predict.py consumes iVideoGPT npz episodes; SurgWMBench "
        "manifest-to-npz conversion and checkpoint selection are still model-specific."
    ),
    notes=(
        "iVideoGPT native prediction is available through inference/predict.py after "
        "preparing an action-free or action-conditioned checkpoint. The SurgWMBench "
        "adapter smoke path validates official manifests and metrics with copy-last frames."
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(SPEC, argv)


if __name__ == "__main__":
    main()
