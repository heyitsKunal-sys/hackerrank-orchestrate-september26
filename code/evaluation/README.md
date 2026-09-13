# Evaluation workflow

1. Run `python code/main.py` from the repository root.
2. The program validates one output row per request, output schema, allowed decision
   values, and the safe-amount bounds before writing the file.
3. Check solved public examples by temporarily substituting `sample_requests.csv`
   for `requests.csv`; this is a development calibration step only and is not used
   by the final dataset run.
4. Confirm `output.csv` has 251 lines (header plus 250 requests) before submitting.
