# Swiss Parliament replication

Rebuild the deck end-to-end from the official API:

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
./run_all.sh                  # ~18 h, ~$30 at gpt-4o-mini prices
```

Output: `slides_letemps_parlacap.pdf`. Compilation requires `tectonic`.

To skip the API steps and use the committed frozen dataset instead:

```bash
./run_all.sh --skip-api       # fast (~2 min)
```

## Pipeline

| Script | Input | Output | Notes |
|---|---|---|---|
| `src/00_fetch.py` | parlament.ch OData API | `data/raw/speeches.csv` | ~1 h, no key needed |
| `src/01_label_rhetoric.py` | speeches.csv | `data/processed/rhetoric_labels.csv` | OPENAI_API_KEY required |
| `src/02_label_emotions.py` | speeches.csv | `data/processed/emotion_labels.csv` | OPENAI_API_KEY required |
| `src/03_label_topics.py` | speeches.csv | `data/processed/topic_labels.csv` | OPENAI_API_KEY required |
| `src/05_ga_score.py` | speeches.csv | `data/processed/ga_scores.csv` | dictionary-based, no key needed |
| `src/04_merge.py` | raw + processed | `data/processed/final_dataset.csv` | |
| `src/build.py` | final_dataset.csv | `output/figure_data/*.csv` | |
| `src/compile.py` | slides + figures | `slides_letemps_parlacap.pdf` | |

Each labelling script supports `--smoke N` (test on N speeches) and `--resume` (continue after interruption).

## Layout

- `slides/slides_letemps_parlacap.tex` — Beamer source.
- `data/frozen/approved_figures/` — pre-generated figures embedded in the slides.
- `data/frozen/source_csv/` — small upstream tables (Gennaro–Ash scores, country comparison, decomposition drivers, gender summary).
- `data/frozen/emotions_ahef_data.csv` — chart input for the AHEF-by-party figure.
- `output/figure_data/` — chart-level CSVs (one per figure + `manifest.csv`), committed for browsing without running the pipeline.
- `.env.example` — template for API credentials.

## Data sources

- **Speeches**: Swiss Parliament OData service at <https://ws.parlament.ch/odata.svc/>
- **Rhetoric and emotion labels**: OpenAI `gpt-4o-mini` via the prompts in `src/01_label_rhetoric.py` and `src/02_label_emotions.py`
- **Topic labels**: CAP codebook (Comparative Agendas Project, <https://www.comparativeagendas.net/>) via `src/03_label_topics.py`
- **Gennaro–Ash emotionality scores**: Gennaro & Ash (2022), *Emotion and Reason in Political Language*, Economic Journal. Dictionaries in `data/frozen/dictionary_*.json` (EN + FR stemmed word lists). Encoded with [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2); score formula: `Yi = (sim(speech, Affect) + 1) / (sim(speech, Cognition) + 1)`.
