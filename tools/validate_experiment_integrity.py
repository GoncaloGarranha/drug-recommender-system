from __future__ import annotations

import argparse
from pathlib import Path

from models.shared.experiment_integrity import load_and_validate_experiment_integrity


def main() -> None:
	parser = argparse.ArgumentParser(description="Validate experiment split integrity before training")
	parser.add_argument("--config", required=True, help="Path to the experiment YAML config")
	parser.add_argument(
		"--output",
		default=None,
		help="Optional path for the validation report JSON. Defaults to processed_dir/splits/split_integrity_report.json",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Recreate splits before validation if they are missing or stale.",
	)
	args = parser.parse_args()

	report = load_and_validate_experiment_integrity(
		args.config,
		output_path=Path(args.output) if args.output else None,
		force=args.force,
	)

	print(f"[OK] Split integrity validated: {report['status']}")
	print(f"[OK] Report saved to: {report['report_path']}")


if __name__ == "__main__":
	main()
