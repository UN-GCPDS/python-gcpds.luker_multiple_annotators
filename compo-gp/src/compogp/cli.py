from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .pipeline import run_pipeline, predict_from_artifacts, save_artifacts, load_artifacts

def main():
    ap = argparse.ArgumentParser(prog="compogp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train model")
    tr.add_argument("--sensory", required=True)
    tr.add_argument("--ingredients", required=True)
    tr.add_argument("--id-col", default="sample_id")
    tr.add_argument("--out", required=True)
    tr.add_argument("--num-inducing", type=int, default=20)
    tr.add_argument("--tau", type=float, default=100.0)
    tr.add_argument("--max-iters", type=int, default=5000)
    tr.add_argument("--loocv", action="store_true")

    pr = sub.add_parser("predict", help="Predict from CSV using saved artifacts")
    pr.add_argument("--artifacts", required=True)
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        artifacts = run_pipeline(
            sensory_csv=args.sensory,
            ingredients_csv=args.ingredients,
            id_col=args.id_col,
            num_inducing=args.num_inducing,
            tau=args.tau,
            max_iters=args.max_iters,
            do_loocv=args.loocv,
        )
        Path(args.out).mkdir(parents=True, exist_ok=True)
        save_artifacts(artifacts, args.out)
        print("Saved to", args.out)

    elif args.cmd == "predict":
        arts = load_artifacts(args.artifacts)
        X_new = pd.read_csv(args.input)
        Yhat_pct, vars_df = predict_from_artifacts(arts, X_new)
        Yhat_pct.to_csv(args.output, index=True)
        print("Wrote", args.output)

if __name__ == "__main__":
    main()
