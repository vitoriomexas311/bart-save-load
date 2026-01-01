"""Train in one interpreter, terminate it, and predict in another interpreter.

Unlike a same-process pickle test, this cannot accidentally reuse live trees.
All files are generated from synthetic data; no precomputed predictions are
used to manufacture a pass. Logs and comparison JSON are kept for inspection.
"""

import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProcessRoundTripTests(unittest.TestCase):
    """Exercise the published scripts exactly as a user would run them."""

    def run_phase(self, script, phase, folder):
        """Launch a child, wait for its exit, and preserve its output on failure."""
        command = [sys.executable, str(ROOT / script), phase, str(folder)]
        if script == 'demo.py' and phase == 'train':
            command += ['--chains', '2', '--cores', '2']
        log_path = folder / f'{phase}.log'
        with log_path.open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=600)
        self.assertEqual(result.returncode, 0, log_path.read_text()[-10000:])

    def test_fresh_process_round_trip(self):
        """Both serializers must restore all ensembles and both PyMC outputs."""
        version = md.version('pymc-bart')
        self.assertIn(version, ['0.11.0', '0.13.1'], 'Use one of the published locks')
        script = 'demo.py' if version == '0.11.0' else 'demo_current.py'
        # Keep artifacts on disk so CI can upload them even after a failed test.
        artifact_root = Path(os.environ.get('BART_TEST_OUTPUT', ROOT / '.test-artifacts'))
        artifact_root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f'bart-{version}-', dir=artifact_root))
        self.run_phase(script, 'train', folder)
        self.run_phase(script, 'reload', folder)
        result_file = folder / ('result-local.json' if version == '0.11.0' else 'result-current.json')
        result = json.loads(result_file.read_text())
        meta = json.loads((folder / 'metadata.json').read_text())
        self.assertTrue(result['passed'])
        self.assertEqual(meta['retained_ensembles'], 160)
        self.assertEqual(len(result['checks']), 40)
        self.assertIn('trace_only', result)
        self.assertLessEqual(result['max_absolute_difference'], 1e-12)
        self.assertEqual({row['format'] for row in result['checks']}, {'pickle', 'cloudpickle'})
        self.assertEqual({row['case'] for row in result['checks']},
                         {'new_1', 'new_17', 'new_120', 'new_175', 'training'})
        self.assertEqual({row['kind'] for row in result['checks']}, {'trees', 'all', 'mu', 'y'})
        print(f'\nVerified fresh-process artifacts: {folder}', flush=True)
