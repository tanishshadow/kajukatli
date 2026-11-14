# AML Graph

AML Graph is a prototype for spotting suspicious patterns across connected bank
transfers. A transfer that looks ordinary by itself may look different when
viewed with the accounts and transfers around it.

The project checks for four patterns: many accounts sending to one account,
one account sending to many accounts, money moving around a loop of accounts,
and money being passed quickly from one account to another. The website lets
you select a detector result, inspect its accounts and transfers, and replay
the sequence over time.

A flagged result is a **candidate to review**. It does not prove that money
laundering happened.

## Run the website on Windows

The existing `aml web` command starts the backend and serves the frontend pages
together. There is no separate frontend server or Node.js install.

### 1. Set up Python packages

Python 3.12 or 3.13 is suitable; this project does not need a new Python
installation. If `.venv` does not exist yet, create it using the Python already
installed on your computer. Use the version you have:

```powershell
cd C:\Users\tanis\Documents\pyinter
py -3.13 -m venv .venv
# Or, if you have Python 3.12 instead:
# py -3.12 -m venv .venv
```

Install the project packages (run this once, or again after package changes):

```powershell
.\.venv\Scripts\python -m pip install -e .
```

If `.venv` already exists, skip the environment creation command.

### 2. Prepare the dataset (first setup only)

Place the HI-Small files `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in
the project folder. Then run:

```powershell
.\.venv\Scripts\aml prepare --config backend/configs/default.yaml
```

This creates the smaller files the app uses under `data/processed/`. The raw
dataset files are left as they are. You can skip this step if `data/processed/`
already contains the prepared files.
Official dataset sources

Download the dataset from either of these sources:

Option 1 — Kaggle (recommended)

IBM Transactions for Anti-Money Laundering (AML) — Kaggle

Option 2 — IBM original dataset

IBM AML Dataset — Box

Kaggle may require an account to download the files.

Required files

The current preprocessing instructions require:

HI-Small_Trans.csv
HI-Small_Patterns.txt

The HI-Small distribution also includes:

HI-Small_accounts.csv

### 3. Create detector results (first setup, or after preparing new data)

Run each detector once to create the cases shown in the investigation page:

```powershell
.\.venv\Scripts\aml detect --detector fan --config backend/configs/default.yaml
.\.venv\Scripts\aml detect --detector scatter_gather --config backend/configs/default.yaml
.\.venv\Scripts\aml detect --detector cycle --config backend/configs/default.yaml
.\.venv\Scripts\aml detect --detector pass_through --config backend/configs/default.yaml
```

### 4. Start the website and backend

In PowerShell, go to the project folder and start the combined server:

```powershell
cd C:\Users\tanis\Documents\pyinter
.\.venv\Scripts\aml web --config backend/configs/default.yaml
```

Open **http://127.0.0.1:8765/** for the website, or go directly to
**http://127.0.0.1:8765/investigation.html** to explore detector cases. Keep
the PowerShell window open while using the site. Press `Ctrl+C` to stop the
server. If port 8765 is already in use, add `--port 8766` and use that port in
the browser address.

The investigation page loads a list of real cases from the detector results.
When you choose one, the backend fetches only that case's transactions for the
graph; it does not send the full dataset to your browser. It shows the matched
accounts, transfers, timing, detector evidence, and dataset labels. You can
click accounts and transfers, move around the graph, and play or pause the
transfer sequence.

To run the project's checks after making code changes:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## What the results mean

The project uses the **HI-Small synthetic dataset**: computer-generated bank
transactions with example suspicious patterns. This is useful for building
and demonstrating the prototype, but it is not real customer activity. Good
results on this dataset would not by themselves show that the system works
well at a bank.

The system uses earlier data to set its rules, a separate validation period to
choose its alert cutoff, and a final test period to measure the frozen setup.
The final test is protected against being run a second time under a changed
configuration. **Do not run `aml report` casually:** it performs that final
evaluation, and it is intended to run once.

### Test-period disclosure

The test period was evaluated twice before the one-run safeguard was added.
After the first run, scoring and evasion settings changed and score-report
formatting was updated. The earlier outputs are kept under
`outputs/m11/archived_exploratory/` and are exploratory, not the final result.
Those earlier files did not record a configuration hash, so an exact
before-and-after hash comparison is unavailable. The current final report was
run with the frozen configuration recorded by the project.

## Current limitations

- The data is synthetic; real-world effectiveness has not been established.
- The latest M10 evasion matrix is a validation-only experiment with 20 rows:
  amount splitting, bounded delays, benign decoys, and combined changes at
  strengths 0, 0.25, 0.5, 0.75, and 1.0. The earlier CSV and JSON are preserved
  under `outputs/m10/archived_prior/` for comparison. The evaluator now reports
  legacy exact-edge recall, period-censored attempt recall, split-path recovery,
  and detector candidates before the frozen model cutoff.
- That matrix does **not** establish end-to-end robustness: the frozen cutoff
  selected zero validation-only candidate clusters. The highest candidate
  probability was 0.84065, below the frozen cutoff 0.85064. This isolates a
  scoring/context mismatch after detector evidence is found; the cutoff was
  chosen on features from full-timeline clusters, while M10 rebuilds the graph
  from validation rows only. Do not lower the cutoff based on this result.
- 60 of 76 validation-start attempts extend beyond the validation boundary.
  Legacy whole-attempt edge recall is therefore boundary-censored in M10; the
  separately reported period-censored metric restricts expected edges to the
  validation rows and does not replace the legacy metric.
- Detector thresholds, cluster merging, and model training have not yet been
  changed to address the scoring/context mismatch, amount-splitting false
  positives, decoy resistance, or delayed-path recall. The adversarial input
  generator and evaluation diagnostics are improved, but the detection
  robustness objective remains **NOT COMPLETE**. The one-shot test report was
  not rerun.
- Some additional evaluation comparisons are unfinished. The existing final
  report is a reduced report; do not interpret missing comparisons as passing.
- The text summary uses a built-in offline template. No outside AI service or
  API key is connected.
- Very large detected cases may be shortened in the graph view; the page tells
  you when transactions were omitted.

## Data credit

The HI-Small benchmark comes from IBM's synthetic financial-transactions work.
See the [research paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html)
and IBM's [dataset information and license](https://github.com/IBM/AML-Data/blob/main/README.md)
before sharing the dataset or a hosted copy. Raw data and generated results are
excluded from Git.
