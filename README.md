# AML Graph

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-informational)](#run-tests)
[![Dataset](https://img.shields.io/badge/dataset-HI--Small%20synthetic-orange)](#dataset)
[![Status](https://img.shields.io/badge/status-research%20prototype-yellow)](#current-limitations)

AML Graph is a prototype for spotting suspicious patterns across connected bank transfers. It groups transfers into temporal graph patterns, shows the accounts and evidence behind each candidate, and lets you replay the transfer sequence. A flagged result is a **candidate to review**; it does not prove that money laundering happened.

The four detector commands map to these patterns: `fan` detects fan-in (many accounts sending to one account) and fan-out (one account sending to many); `scatter_gather` detects scatter-gather and gather-scatter layering; `cycle` detects time-ordered loops; and `pass_through` detects rapid transfer chains.

## Clone and install

```bash
git clone <https://github.com/tanishshadow/kajukatli> kajukatli
cd kajukatli
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

```powershell
git clone <https://github.com/tanishshadow/kajukatli> kajukatli
cd kajukatli
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Dataset

Get the HI-Small dataset from either [Kaggle](https://www.kaggle.com/datasets/berkanoztas/transactions-data) or [IBM's dataset page](https://github.com/IBM/AML-Data). Kaggle may require an account. The current preparation step requires these files in the project root:

```text
HI-Small_Trans.csv
HI-Small_Patterns.txt
```

`HI-Small_accounts.csv` is optional. Raw files remain unchanged. Prepare the processed files with:

```bash
aml prepare --config backend/configs/default.yaml
```

## Run the local demo

Create detector results for the investigation page:

```bash
aml detect --detector fan --config backend/configs/default.yaml
aml detect --detector scatter_gather --config backend/configs/default.yaml
aml detect --detector cycle --config backend/configs/default.yaml
aml detect --detector pass_through --config backend/configs/default.yaml
```

Start the combined backend and website:

```bash
aml web --config backend/configs/default.yaml
```

Open **http://127.0.0.1:8765/** or go directly to **http://127.0.0.1:8765/investigation.html**. The backend is not hosted because the IBM dataset is large and subject to its license. The live site uses precomputed sample cases; the full pipeline runs locally. On Windows, press `Ctrl+C` in PowerShell to stop the server. To use another port, add `--port 8766`.

The investigation page fetches only the selected case's transactions for the graph, not the full dataset. It shows the accounts, transfers, timing, detector evidence, and dataset labels.

## Run tests

After activating the virtual environment, run:

```bash
pytest -q
```

```powershell
python -m pytest -q
```

## Evaluation integrity

The system uses a temporal split: train data sets detector/model parameters, validation data selects the alert cutoff, and the frozen setup is measured on the final test period. `aml report` performs that final evaluation and is guarded against a second run under a changed configuration. Avoid running it casually; use the saved report and the [Evaluation page](http://127.0.0.1:8765/evaluation.html) for existing results.

### Test-period disclosure

The test period was evaluated twice before the one-run safeguard was added. After the first run, scoring and evasion settings changed and score-report formatting was updated. The earlier outputs are kept under `outputs/m11/archived_exploratory/` and are exploratory, not the final result. Those earlier files did not record a configuration hash, so an exact before-and-after hash comparison is unavailable. The current final report was run with the frozen configuration recorded by the project.

## Current limitations

- HI-Small is synthetic; real-world effectiveness has not been established.
- Some evaluation comparisons are unfinished. The saved final report is reduced; missing comparisons should not be interpreted as passing.
- Some benchmark typologies are not covered, and recall varies by pattern.
- Summaries use an offline deterministic template; no outside AI service or API key is connected.
- Very large detected cases may be shortened in the graph view; the page reports when transfers are omitted.

## Data credit

The HI-Small benchmark comes from IBM's synthetic financial-transactions work. See the [research paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html) and IBM's [dataset information and license](https://github.com/IBM/AML-Data/blob/main/README.md) before sharing the dataset or a hosted copy. Raw data and generated results are excluded from Git.
