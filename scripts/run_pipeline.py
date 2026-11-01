"""Command-line entry point.

  python scripts/run_pipeline.py all                # REAL data (needs internet for first run)
  python scripts/run_pipeline.py all --synthetic    # code check with computer-generated test data
  python scripts/run_pipeline.py train              # run a single stage
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rasterio.errors import RasterioIOError  # noqa: E402
from src import pipeline  # noqa: E402
from src.utils import get_paths, load_config  # noqa: E402

STAGES = ["acquire", "check", "train", "predict", "analyze"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--synthetic", action="store_true", help="use generated test data (NOT real results)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(synthetic=args.synthetic)
    todo = STAGES if args.stage == "all" else [args.stage]
    for s in todo:
        print(f"=== stage: {s} ===")
        try:
            if s == "acquire":
                pipeline.stage_acquire(cfg, paths, args.synthetic)
            else:
                getattr(pipeline, f"stage_{s}")(cfg, paths)
        except (FileNotFoundError, RasterioIOError) as exc:
            flag = " --synthetic" if args.synthetic else ""
            sys.exit(f"\nStage '{s}' could not find an input file:\n  {exc}\n"
                     f"Run the earlier stages first, e.g.  python scripts/run_pipeline.py acquire{flag}")
        except RuntimeError as exc:  # raised by acquisition for 'no scenes found' etc.
            sys.exit(f"\nStage '{s}' failed: {exc}")
    print("Done. Launch the dashboard with:  streamlit run app/app.py")


if __name__ == "__main__":
    main()
