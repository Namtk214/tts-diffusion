"""Archive the completed single GPU experiment to the user's HF dataset."""
import hashlib
import csv
import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

ROOT = Path('/workspace/tts-diffusion/reproduction')
WORKSPACE = ROOT.parents[1]
STAGE = Path('/workspace/tts-hf-upload-stage')
BUNDLE_ID = 'satisdive-sgi-flux1dev-768-30x2-20261126'
PREFIX = f'runs/{BUNDLE_ID}'
BUNDLE = STAGE / PREFIX
REPO = 'Namthukhoa214/TTS-FLUX'
RUNS = {
    'pickapic/satisdive': '8b28eda458de67a65d56',
    'pickapic/sgi': '5e63ecb5b186d602d0d2',
    'geneval/satisdive': '39d5f02e5824319955b2',
    'geneval/sgi': '4e78c2b98f280d346da0',
}
METRICS = ('hpsv3', 'image_reward', 'pickscore', 'clip_score', 'dreamsim', 'dino', 'lpips')


def copy_or_link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def ignore(directory, names):
    return {name for name in names if name in {'.git', '__pycache__', '.locks', '.staging'}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_complete():
    state = json.loads(Path('/workspace/tts-pipeline-status.json').read_text())
    if state.get('stage') != 'complete':
        raise RuntimeError(f"Pipeline is not complete: {state.get('stage')}")
    summary = {}
    for key, run_id in RUNS.items():
        dataset, method = key.split('/')
        run = ROOT / 'outputs/single_gpu_768' / dataset / method / run_id
        manifest = json.loads((run / 'run_manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['coverage']['complete'] != 30:
            raise RuntimeError(f'Generation incomplete: {key}')
        prompts = [json.loads(line) for line in (run / 'prompts.jsonl').read_text().splitlines()]
        if len(prompts) != 30:
            raise RuntimeError(f'Prompt count is not 30: {key}')
        for row in prompts:
            if any(not (run / 'images' / row['prompt_id'] / f'{n:03}.png').is_file() for n in range(4)):
                raise RuntimeError(f"Missing image: {key}/{row['prompt_id']}")
        latest = json.loads((run / 'evaluation/latest_report.json').read_text())
        report = json.loads((run / latest['directory'] / 'report_manifest.json').read_text())
        if report['coverage']['fully_evaluated'] != 30:
            raise RuntimeError(f'Evaluation coverage incomplete: {key}')
        expected = set(METRICS) | ({'geneval'} if dataset == 'geneval' else set())
        if not expected.issubset(report['metric_ids']):
            raise RuntimeError(f'Missing metric IDs: {key}')
        for metric in expected:
            metric_id = report['metric_ids'][metric]
            base = run / 'evaluation' / ('geneval' if metric == 'geneval' else f'cache/{metric}') / metric_id
            for row in prompts:
                record = json.loads((base / f"{row['prompt_id']}.json").read_text())
                if record.get('status') != 'complete' or record.get('metric_id') != metric_id:
                    raise RuntimeError(f"Metric incomplete: {key}/{metric}/{row['prompt_id']}")
        summary[key] = {'run_id': run_id, 'report_id': latest['eval_id'], 'metrics': report['metric_ids']}
    return summary


def make_bundle(summary):
    if (BUNDLE / 'EXPORT_READY').is_file():
        return
    if STAGE.exists():
        shutil.rmtree(STAGE)
    temporary = Path('/workspace/tts-hf-upload-building')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    contents = temporary / PREFIX
    contents.mkdir(parents=True)
    for name in ('outputs', 'reports'):
        shutil.copytree(ROOT / name, contents / 'reproduction' / name, copy_function=copy_or_link, ignore=ignore)
    source = contents / 'reproduction'
    for item in ROOT.iterdir():
        if item.name in {'.venvs', 'outputs', 'reports', 'models'}:
            continue
        dest = source / item.name
        if item.is_dir():
            shutil.copytree(item, dest, copy_function=copy_or_link, ignore=ignore)
        elif item.is_file():
            copy_or_link(item, dest)
    model_dir = source / 'models'
    model_dir.mkdir()
    for p in (ROOT / 'models').rglob('*.json'):
        if p.is_file() and not p.is_symlink():
            dest = model_dir / p.relative_to(ROOT / 'models')
            dest.parent.mkdir(parents=True, exist_ok=True)
            copy_or_link(p, dest)
    runtime = contents / 'runtime'
    runtime.mkdir()
    keep = ('tts-main.log', 'tts-resume.log', 'tts-evaluation.log', 'tts-eval-setup.log',
            'tts-pipeline-status.json', 'tts-generation-runs.json', 'tts-eval-after-generation.py',
            'tts-main.sh', 'tts-resume-768.sh', 'tts-eval-setup.sh', 'tts-eval-resume-index')
    for name in keep:
        p = WORKSPACE / name
        if p.is_file():
            if re.search(rb'hf_[A-Za-z0-9]{20,}', p.read_bytes()):
                raise RuntimeError(f'Credential-like value in runtime file: {name}')
            copy_or_link(p, runtime / name)
    (contents / 'canonical_runs.json').write_text(json.dumps(summary, indent=2) + '\n')
    rows = []
    for key, info in summary.items():
        dataset, method = key.split('/')
        table = ROOT / 'outputs/single_gpu_768' / dataset / method / info['run_id'] / 'evaluation/reports' / info['report_id'] / 'summary.csv'
        with open(table, newline='') as stream:
            rows.extend(csv.DictReader(stream))
    with open(contents / 'metrics_summary.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (contents / 'README.md').write_text(
        '# SatisDive and SGI, FLUX.1-dev, 768 px\n\n'
        'Subset experiment: 30 Pick-a-Pic and 30 GenEval prompts per method, four PNGs per prompt '
        '(480 canonical images). Exact canonical run IDs and active metric cache IDs are in '
        '`canonical_runs.json`. Aggregate metric values are in `metrics_summary.csv`. '
        'The `reproduction/` tree contains all run records, generated '
        'images, per-image metric caches, CSV reports, galleries, source, pinned configurations, '
        'prompt subsets, provenance, and GenEval asset manifests. `runtime/` contains VM logs '
        'and continuation scripts. Older pilot/failed outputs are retained for provenance; use '
        'the four run IDs in `canonical_runs.json` for analysis.\n\n'
        'Recreate the Python environments and download the pinned FLUX/HPSv3/other model '
        'weights described in `reproduction/README.md` and `configs/models.lock.json` to run '
        'further inference. Large model caches and virtual environments are not archived. '
        'The original VM paths in manifests are historical; place this `reproduction/` directory '
        'at your chosen path and use the relative run directories.\n\n'
        'This is a 30 prompt subset comparison, not the full paper experiment. '
        'Metric definitions and limitations are in `reproduction/METRICS.md`.\n'
    )
    files = sorted(p for p in contents.rglob('*') if p.is_file())
    with open(contents / 'FILE_MANIFEST.jsonl', 'w') as out:
        for p in files:
            out.write(json.dumps({'path': str(p.relative_to(contents)), 'bytes': p.stat().st_size,
                                  'sha256': sha256(p)}, ensure_ascii=False) + '\n')
    (contents / 'EXPORT_READY').write_text(datetime.now(timezone.utc).isoformat() + '\n')
    temporary.rename(STAGE)


def main():
    summary = check_complete()
    make_bundle(summary)
    token = Path('/dev/shm/tts-hf/token').read_text().strip()
    api = HfApi(token=token)
    if api.whoami().get('name') != 'Namthukhoa214':
        raise RuntimeError('Unexpected Hugging Face account')
    api.upload_large_folder(repo_id=REPO, repo_type='dataset', folder_path=STAGE,
                            num_workers=4, print_report_every=60)
    expected = {str(p.relative_to(STAGE)) for p in BUNDLE.rglob('*') if p.is_file()
                and '.cache' not in p.parts}
    found = {item.path for item in api.list_repo_tree(REPO, path_in_repo=PREFIX,
                                                       recursive=True, repo_type='dataset')
             if isinstance(item, RepoFile)}
    missing = expected - found
    if missing:
        raise RuntimeError(f'Upload verification found {len(missing)} missing files: {sorted(missing)[:5]}')
    result = {'status': 'verified', 'repo': REPO, 'prefix': PREFIX,
              'files': len(expected), 'canonical_images': 480, 'runs': summary,
              'verified_at': datetime.now(timezone.utc).isoformat()}
    Path('/workspace/tts-hf-upload-status.json').write_text(json.dumps(result, indent=2) + '\n')
    print('HF_UPLOAD_VERIFIED', PREFIX, len(expected), flush=True)


if __name__ == '__main__':
    main()
