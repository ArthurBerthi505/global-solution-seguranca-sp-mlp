"""Executa células via IPython no mesmo processo, sem servidor Jupyter.

Útil em ambientes sem sockets para kernels. Não simula saídas: cada célula
passa pelo interpretador IPython e seus streams/display são capturados.
"""
from pathlib import Path
import sys
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "notebooks/GlobalSolution_MLP_SP.ipynb"
notebook = nbformat.read(p, as_version=4)
shell = InteractiveShell.instance()
count = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    count += 1
    with capture_output() as captured:
        result = shell.run_cell(cell.source, store_history=True, silent=False)
    cell.execution_count = count
    cell.outputs = []
    if captured.stdout:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
    if captured.stderr:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
    print(f"Célula {count}: {len(cell.outputs)} saídas", flush=True)
    if result.error_before_exec or result.error_in_exec:
        nbformat.write(notebook, p)
        raise RuntimeError(f"Erro na célula {count}: {result.error_before_exec or result.error_in_exec}")
nbformat.validate(notebook)
nbformat.write(notebook, p)
print("Notebook executado e validado:", count, "células de código")
