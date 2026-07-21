# Evaluation results

Saved benchmark reports. Every benchmark script tees its terminal report here via
`evaluation.capture()`.

There is exactly one file per benchmark — `<benchmark>_latest.txt` — overwritten on
every run, so it always holds the most recent result (no timestamped history pile).

Each file starts with a header (run time + the exact command) followed by the verbatim
report. Runs are reproducible: every script samples with a fixed `random_state`, so the
same `--n` on the same dataset yields the same numbers.

| Benchmark | What it measures (vs Ground Truth.csv) |
|---|---|
| `evaluate_ocr`     | Field recall (Brand/MRP/NetQty/FSSAI/UPC/Manufacturer), preprocessed vs raw, escalation rate |
| `ceiling`          | What is even recoverable — how much was actually photographed |
| `benchmark_models` | All free engines side by side (rapidocr / easyocr / local_vlm / union) |
| `compare_engines`  | RapidOCR vs local VLM on the hard subset |
| `tune_ocr`         | FSSAI-recovery probe across thresholds / tiling |
| `benchmark_barcode`| Barcode pipeline: decode → checksum → lookup → exact GT UPC match |
