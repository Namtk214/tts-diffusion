"""Parse source/config syntax only. No project imports, tests, downloads or model loading."""
import ast
import json
from pathlib import Path
import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    count = 0
    for path in root.rglob("*.py"):
        if any(part in (".venvs", ".venv", "models", "external") for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            count += 1
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"{path}: {error}")
    for path in (root / "configs").rglob("*"):
        try:
            if path.suffix in (".yml", ".yaml"):
                yaml.safe_load(path.read_text())
            elif path.suffix == ".json":
                json.loads(path.read_text())
        except (ValueError, yaml.YAMLError) as error:
            errors.append(f"{path}: {error}")
    print(json.dumps({"python_files_parsed": count, "syntax_errors": errors,
                      "scope": "static syntax only; no tests executed"}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
