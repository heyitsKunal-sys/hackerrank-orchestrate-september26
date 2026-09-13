# Buy or Wait agent

Run from the repository root with `python code/main.py`. The standard-library-only
agent reads `dataset/`, forecasts cash flow for 90 days, evaluates allowed payment
options, and writes `output.csv`. If Tesseract is available, it is used only to
recover amounts from locally supplied images when an event amount is blank.

The `evaluation/` folder contains the reproducible validation workflow and token-use report.
