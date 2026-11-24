"""Web launcher and reproducible headless experiments."""
import argparse
from pathlib import Path


def default_data_path():
    return Path(__file__).resolve().parents[2] / "matches.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Football Predictor — WEB, KNN, Random Forest și Poisson")
    parser.add_argument("--data", type=Path, default=default_data_path(), help="Fișierul CSV de intrare")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--evaluate", action="store_true", help="Rulează experimente fără interfață grafică")
    modes.add_argument("--benchmark", action="store_true", help="Compară configurații istorice și optimizate pe sezoane")
    modes.add_argument("--doctor", action="store_true", help="Verifică instalarea web și fișierul de date fără antrenare")
    parser.add_argument("--output", type=Path, default=Path("reports"), help="Directorul rapoartelor experimentale")
    parser.add_argument("--host", default="127.0.0.1", help="Adresa serverului web (implicit 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Portul serverului web (implicit 8080)")
    parser.add_argument("--work-dir", type=Path, help="Director uploaduri și exporturi web (implicit data/webapp)")
    parser.add_argument("--model-config", type=Path, help="Rețetă optimizată explicită; se aplică numai CSV-ului cu hash identic")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port trebuie să fie în intervalul 1..65535")
    if args.doctor:
        from .doctor import check_environment, format_report
        report = check_environment(args.data)
        print(format_report(report))
        return 0 if report["ok"] else 1
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
        from .service_factory import create_project_service
        service = create_project_service(args.data, config_path=args.model_config)
        report = service.train(progress=print)
        service.export_report(args.output)
        for row in report["metrics"]:
            print(f"{row['model']}: accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, log-loss={row['log_loss']:.4f}")
        print(f"Rapoarte: {args.output.resolve()}")
        return 0
    from .web import create_app
    from waitress import serve
    web_options = dict(data_path=args.data, work_dir=args.work_dir)
    if args.model_config is not None:
        from functools import partial
        from .service_factory import create_project_service
        web_options['service_factory'] = partial(create_project_service, config_path=args.model_config)
    app = create_app(**web_options)
    print(f"Football Predictor: http://{args.host}:{args.port}", flush=True)
    try:
        serve(app, host=args.host, port=args.port, threads=8)
    finally:
        workspace = getattr(app, 'extensions', {}).get('workspace')
        if workspace is not None:
            workspace.close()
    return 0
