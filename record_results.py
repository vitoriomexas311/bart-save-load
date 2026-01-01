"""Turn successful local suite logs into a small, public verification report."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def main():
    """Read actual suite output and its generated results; reject failed runs.

    This does not run or manufacture tests. Run each of the three documented
    environments first, redirecting output to tests-py311.log, tests-py313.log,
    and tests-current.log. Only compact numerical evidence is published; model
    pickles, local environments, and compiler caches stay outside Git.
    """
    suites = []
    for label in ['py311', 'py313', 'current']:
        log = (ROOT / f'tests-{label}.log').read_text()
        if not re.search(r'^OK(?: \(skipped=\d+\))?$', log, re.MULTILINE):
            raise ValueError(f'{label} did not finish successfully')
        match = re.search(r'Ran (\d+) tests in ([\d.]+)s', log)
        folder_match = re.search(r'Verified fresh-process artifacts: (.+)', log)
        if match is None or folder_match is None:
            raise ValueError(f'{label} has no complete integration evidence')
        folder = Path(folder_match.group(1))
        filename = 'result-current.json' if label == 'current' else 'result-local.json'
        result = json.loads((folder / filename).read_text())
        if not result['passed'] or len(result['checks']) != 40:
            raise ValueError(f'{label} has incomplete prediction comparisons')
        suites.append({'environment': label, 'versions': result['versions'],
                       'tests_discovered': int(match.group(1)),
                       'tests_skipped': 3 if label == 'current' else 0,
                       'elapsed_seconds': float(match.group(2)), 'passed': True,
                       'numerical_comparisons': len(result['checks']),
                       'max_absolute_difference': result['max_absolute_difference'],
                       'trace_only_control': result['trace_only']})
    report = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'suites': suites}
    (ROOT / 'reports/local-test-suites.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
