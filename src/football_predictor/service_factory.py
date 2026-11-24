"""Choose a source-compatible, explicitly approved model recipe for the web app."""
from .service import ProjectService


def create_project_service(csv_path, config_path=None):
    """Use frozen settings only when a recipe is explicitly supplied."""
    from pathlib import Path
    import json

    if config_path is None or not Path(config_path).exists():
        return ProjectService(csv_path)
    from .optimized_service import OptimizedProjectService, _validate_config
    from hashlib import sha256
    config = _validate_config(json.loads(Path(config_path).read_text()))
    if sha256(Path(csv_path).read_bytes()).hexdigest() != config['source_sha256']:
        return ProjectService(csv_path)
    return OptimizedProjectService(csv_path, config)
