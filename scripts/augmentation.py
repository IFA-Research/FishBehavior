import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs"),
        help="Directory that contains one subfolder per run, each with training_history.csv",
    )
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("*/training_history.csv"))
    if not paths:
        raise SystemExit(f"No training_history.csv under {args.run_dir}")

    for path in paths:
        df = pd.read_csv(path)
        print("\n" + "=" * 80)
        print(path.parent.name, "| rows:", len(df))
        print("columns:", list(df.columns))
        print(df.head(2).to_string(index=False))
        print("--- last epoch ---")
        print(df.tail(1).to_string(index=False))
        for col in df.columns:
            if any(k in col.lower() for k in ("acc", "f1", "loss")):
                i = df[col].idxmin() if "loss" in col.lower() else df[col].idxmax()
                print(f"best {col}: {df.loc[i].to_dict()}")


if __name__ == "__main__":
    main()
