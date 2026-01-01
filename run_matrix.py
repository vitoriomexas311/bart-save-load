"""Run a real process boundary between training and each reload."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    """Train each legacy environment, then load each archive in both Pythons.

    subprocess.run waits for the previous interpreter to EXIT before launching
    a reload, so a passing test cannot rely on a lingering Python variable.
    Every child log and return code is kept, including failures.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--python-a', default=str(ROOT / '.venv/bin/python'))
    parser.add_argument('--python-b', default=str(ROOT / '.venv-py311/bin/python'))
    parser.add_argument('--skip-training', action='store_true')
    args = parser.parse_args()
    environments = [('py313-single', args.python_a, '1', '1'),
                    ('py311-parallel', args.python_b, '2', '2')]
    jobs = []
    if not args.skip_training:
        for name, python, chains, cores in environments:
            jobs.append((name + '-train', [python, 'demo.py', 'train', 'artifacts/' + name,
                                           '--chains', chains, '--cores', cores]))
    # Form both same-Python and cross-Python reload combinations.
    for name, _, _, _ in environments:
        for label, python in [('py313', args.python_a), ('py311', args.python_b)]:
            jobs.append((name + '-reload-' + label,
                         [python, 'demo.py', 'reload', 'artifacts/' + name, '--label', label]))
    logdir = ROOT / 'logs'
    logdir.mkdir(exist_ok=True)
    outcomes = []
    # Continue independent jobs after a failure, but return a failing exit code.
    for name, cmd in jobs:
        print('Running ' + name, flush=True)
        with (logdir / (name + '.log')).open('w') as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        outcomes.append({'job': name, 'exit_code': proc.returncode})
        print(outcomes[-1], flush=True)
    (ROOT / 'matrix.json').write_text(json.dumps(outcomes, indent=2))
    return int(any(row['exit_code'] != 0 for row in outcomes))


if __name__ == '__main__':
    sys.exit(main())
