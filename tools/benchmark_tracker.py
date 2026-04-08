from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_PATH = REPO_ROOT / "models" / "performance_leaderboard.csv"


def _format_leaderboard(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def log_experiment(model_name, hits1, hits3, hits10, mrr, notes=""):
    """
    Regista os resultados de uma experiência na leaderboard central.
    """
    new_data = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Model": model_name,
        "Hits@1": round(float(hits1), 4),
        "Hits@3": round(float(hits3), 4),
        "Hits@10": round(float(hits10), 4),
        "MRR": round(float(mrr), 4),
        "Notes": notes,
    }

    df_new = pd.DataFrame([new_data])

    if LEADERBOARD_PATH.exists():
        df_old = pd.read_csv(LEADERBOARD_PATH)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_final = df_new

    df_final = df_final.sort_values(by="MRR", ascending=False, kind="mergesort")
    df_final.to_csv(LEADERBOARD_PATH, index=False)

    print("\nLeaderboard updated:")
    print(_format_leaderboard(df_final))


if __name__ == "__main__":
    log_experiment("Example", 0.0, 0.0, 0.0, 0.0, notes="Manual test")