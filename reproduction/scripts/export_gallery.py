"""Make a portable HTML contact sheet from complete generated groups, never generate images."""
import argparse
import base64
import html
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import read_json, read_manifest, atomic_text
from benchmark.storage import validate_group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=10)
    a = p.parse_args()
    m = read_json(a.run_dir / "run_manifest.json")
    size = (m["config"]["backbone"]["width"], m["config"]["backbone"]["height"])
    sections = []
    for row in read_manifest(a.run_dir / "prompts.jsonl")[:a.limit]:
        folder = a.run_dir / "images" / row["prompt_id"]
        if not validate_group(folder, m["config_hash"], row, size):
            sections.append(f"<p>{html.escape(row['prompt_id'])}: incomplete</p>")
            continue
        images = ''.join(f'<img alt="candidate {i}" src="data:image/png;base64,{base64.b64encode((folder/f"{i:03d}.png").read_bytes()).decode()}">' for i in range(4))
        sections.append(f"<section><p>{html.escape(row['prompt_id'])}: {html.escape(row['prompt'])}</p><div>{images}</div></section>")
    atomic_text(a.output, '<!doctype html><meta charset="utf-8"><title>Generated groups</title><style>body{font:16px sans-serif;max-width:1500px;margin:2em auto}div{display:flex}img{width:25%;object-fit:contain}section{margin:2em 0}</style>'
                + f"<h1>{html.escape(m['method'])}: {html.escape(m['run_id'])}</h1>" + ''.join(sections))


if __name__ == "__main__":
    main()
