# Swiss Parliament — Le Temps slides replication

Rebuild the deck end-to-end:

```bash
pip install -r requirements.txt
./run_all.sh
```

Output: `slides_letemps_parlacap.pdf`. Compilation requires `tectonic`.

## Speech-level dataset

The pipeline needs `data/frozen/final_dataset_capstone_with_parlacap.csv`
(~600 MB, speech-level, not in this repo because of GitHub's file-size
limit). Place it at exactly that path before running `./run_all.sh`. Ask
the maintainer for a copy, or rebuild it from the official Swiss
Parliament OData service at <https://ws.parlament.ch/odata.svc/> (Python
wrapper `swissparlpy`) plus the LLM emotion labels and the ParlaCAP topic
classifier.

`output/figure_data/` is committed: the 86 chart-level CSVs that back the
figures can already be browsed without running the pipeline.

## Layout

- `slides/slides_letemps_parlacap.tex` — Beamer source. Figures are picked up from `data/frozen/approved_figures/`.
- `data/frozen/final_dataset_capstone_with_parlacap.csv` — speech-level dataset (not in repo, see above).
- `data/frozen/source_csv/` — small upstream tables used as-is when a figure relies on outputs from an external step (Gennaro–Ash speech-level scores, country comparison, decomposition drivers, gender summary).
- `data/frozen/emotions_ahef_data.csv` — chart input for the AHEF-by-party figure.
- `data/frozen/approved_figures/` — figures embedded in the slides.
- `src/build.py` — for every figure in the deck, writes a compact CSV with the values plotted, plus `manifest.csv`.
- `src/compile.py` — runs `tectonic` and copies the PDF to the project root.
- `output/figure_data/` — chart CSVs written by `build.py` (one per figure + `manifest.csv`).
