"""Execute both notebooks using this project's Python; retain outputs and HTML."""

import json
import os
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_JUPYTER = ROOT / '.jupyter'
os.environ['JUPYTER_RUNTIME_DIR'] = str(LOCAL_JUPYTER / 'runtime')
os.environ['JUPYTER_CONFIG_DIR'] = str(LOCAL_JUPYTER / 'config')
os.environ['IPYTHONDIR'] = str(LOCAL_JUPYTER / 'ipython')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', help='Optional notebook filenames; default is every project notebook.')
    args = parser.parse_args()
    available = {path.name for path in (ROOT / 'notebooks').glob('*.ipynb')}
    names = args.notebooks or sorted(available)
    if set(names) - available:
        raise ValueError('Select an existing filename within notebooks/.')
    # This is project-local kernel metadata; no user kernels or login tokens change.
    kernel_dir = LOCAL_JUPYTER / 'kernels' / 'rca-project'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    specification = {'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                     'display_name': 'Python (RCA project)', 'language': 'python'}
    (kernel_dir / 'kernel.json').write_text(json.dumps(specification), encoding='utf-8')
    kernel_specs = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    record_path = ROOT / 'results/notebook_execution.json'
    reports = json.loads(record_path.read_text()) if record_path.exists() else []
    reports = [report for report in reports if report['notebook'] not in names]
    for name in names:
        path = ROOT / 'notebooks' / name
        notebook = nbformat.read(path, as_version=4)
        print(f'Starting {name} with {sys.executable}', flush=True)
        client = NotebookClient(notebook, timeout=300, kernel_name='rca-project',
                                resources={'metadata': {'path': str(ROOT)}},
                                on_cell_complete=lambda cell_index, **_: print(f'  cell {cell_index + 1} finished', flush=True))
        from jupyter_client import KernelManager
        client.km = KernelManager(kernel_name='rca-project', kernel_spec_manager=kernel_specs)
        client.owns_km = True
        try:
            client.execute()
        except Exception:
            nbformat.write(notebook, path)  # preserve actual diagnostic output on failure
            raise
        nbformat.write(notebook, path)
        body, _ = HTMLExporter().from_notebook_node(notebook)
        html_path = ROOT / 'results' / (path.stem + '.html')
        html_path.write_text(body, encoding='utf-8')
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        errors = [output for cell in code_cells for output in cell.outputs if output.output_type == 'error']
        reports.append({'notebook': name, 'code_cells': len(code_cells),
                        'executed_cells': sum(cell.execution_count is not None for cell in code_cells),
                        'errors': len(errors), 'html': html_path.name})
        print(f'Executed {name}: {len(code_cells)} code cells, {len(errors)} errors.', flush=True)
    reports.sort(key=lambda report: report['notebook'])
    record_path.write_text(json.dumps(reports, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
