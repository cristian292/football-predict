"""Desktop launcher and reproducible headless experiments."""
import argparse
from pathlib import Path


def default_data_path():
    return Path(__file__).resolve().parents[2] / "matches.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Football Predictor — GTK4, KNN, Random Forest și Poisson")
    parser.add_argument("--data", type=Path, default=default_data_path(), help="Fișierul CSV de intrare")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--evaluate", action="store_true", help="Rulează experimente fără interfață grafică")
    modes.add_argument("--benchmark", action="store_true", help="Compară configurații istorice și optimizate pe sezoane")
    parser.add_argument("--output", type=Path, default=Path("reports"), help="Directorul rapoartelor experimentale")
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error(f"Fișier inexistent: {args.data}")
    if args.benchmark:
        import pandas as pd
        from .experiments import run_benchmark
        from .diagnostics import export_diagnostics
        report = run_benchmark(args.data, args.output, progress=print)
        predictions = pd.read_csv(args.output / "predictions_by_season.csv")
        export_diagnostics(predictions, args.output / "diagnostics")
        for row in report["metrics"]:
            print(f"{row['season']} / {row['variant']} / {row['model']}: accuracy={row['accuracy']:.4f}, log-loss={row['log_loss']:.4f}")
        print(f"Benchmark și analiza erorilor: {args.output.resolve()}")
        return 0
    if args.evaluate:
        from .service import ProjectService
        service = ProjectService(args.data)
        report = service.train(progress=print)
        service.export_report(args.output)
        for row in report["metrics"]:
            print(f"{row['model']}: accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, log-loss={row['log_loss']:.4f}")
        print(f"Rapoarte: {args.output.resolve()}")
        return 0
    try:
        from .ui.app import FootballApplication
    except (ImportError, ValueError) as exc:
        parser.exit(1, f"GTK4/PyGObject indisponibil: {exc}\nConsultă README.md; folosește ./run.sh pentru mediul local.\n")
    return FootballApplication(args.data).run([])
