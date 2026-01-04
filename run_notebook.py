"""Execute the walkthrough from a clean kernel and verify its final report."""

import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def main():
    """Run every cell, stop on any error, and keep an executed notebook copy.

    The kernel must belong to the environment installed from
    requirements-notebook.txt. The notebook additionally starts its own reload
    subprocess, so executing this file tests the full process boundary too.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / '.test-artifacts/bart_persistence_walkthrough.executed.ipynb')
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / 'bart_persistence_walkthrough.ipynb', as_version=4)
    # Never use saved execution counts or outputs as evidence of a new run.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}})
    executed = client.execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, args.output)
    report = json.loads((ROOT / 'artifacts/notebook-demo/reload_report.json').read_text())
    if not report['passed'] or report['training_pid'] == report['reload_pid']:
        raise AssertionError('The notebook did not verify a separate-process reload')
    if report['retained_ensembles'] != 160:
        raise AssertionError('Not all posterior ensembles were preserved')
    if any(error > 1e-12 for error in report['max_absolute_errors'].values()):
        raise AssertionError('Restored predictions differ from the original model')
    print(json.dumps(report, indent=2))
    print(f'Executed notebook saved to {args.output}')


if __name__ == '__main__':
    main()
