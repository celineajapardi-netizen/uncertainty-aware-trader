# Getting started on a Mac

## 1. Install Python (once)

Download **Python 3.12** from <https://www.python.org/downloads/> (the macOS
installer) and run it. The Python that comes with macOS is usually too old.

## 2. Open Terminal inside this folder

In Finder, right-click the `confidence_trader` folder → **New Terminal at Folder**.
(Or open Terminal, type `cd ` with a space, drag the folder onto the window, press Enter.)

## 3. Create an environment and install the libraries (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Check that everything works

```bash
python -m pytest tests -q
```

You should see `16 passed`.

## 5. Launch the dashboard

```bash
streamlit run app.py
```

A browser tab opens at <http://localhost:8501>. Press **Ctrl+C** in Terminal to stop it.

## 6. Regenerate the report and figures

```bash
python run_experiment.py
```

Tables, charts and `REPORT.md` are written to the `results/` folder.

---

## Every later session

Only three things: open Terminal in the folder, then

```bash
source .venv/bin/activate
streamlit run app.py
```

## Suggested reading order

1. `README.md` — the research question, the method, the findings
2. `uncertain_trader/strategies.py` — the three decision rules (shortest, most important)
3. `uncertain_trader/model.py` — where the confidence number comes from
4. `uncertain_trader/backtest.py` and `metrics.py` — how results are measured
5. Then experiment in the app:
   * set **Signal strength** to 0 (a random walk) — confidence filtering should stop helping
   * set **Transaction cost** to 0, then to 25 — where does the ranking change?
   * switch to **Real (Yahoo Finance)** with ticker `SPY`, `AAPL`, `TSLA` …

Edits to the code show up in the app after a refresh (Streamlit offers **Rerun**
when it notices a file changed).

## If something goes wrong

* `command not found: python3` → Python isn't installed yet (step 1), or reopen Terminal.
* `No module named streamlit` → the environment isn't active; run `source .venv/bin/activate`.
* Real-data download fails → no internet or Yahoo is down; the simulated market still works,
  and the CSVs already in `data/` (SPY, AAPL, BTC-USD) work offline.
