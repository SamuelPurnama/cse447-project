# Phase 2: Data acquisition and preprocessing

Phase 2 downloads **100,000 conversational and interactive multilingual samples** from HPLT v3, normalizes them to **Unicode NFC**, and stores them as **UTF-8** in one text file (one sample per line).

## What is implemented

- **Source**: HPLT Monolingual Datasets v3.0 (https://data.hplt-project.org/three/sorted).
- **Filtering**: Uses HPLT’s Turku web-register labels when present (e.g. **ID** = Interactive discussion, **IP**, **IN**) to prefer conversational/interactive documents.
- **Language balance**: Samples from ~20 languages by default; each language is capped so the total is balanced.
- **Normalization**: Every line is normalized with `unicodedata.normalize("NFC", text)` so byte–character mapping is consistent for byte-level models.
- **Output**: UTF-8 text file, one sample per line.

## Requirements

Install Phase 2 dependencies (if not already installed):

```bash
pip install -r requirements.txt
```

This adds `requests` and `zstandard` for downloading and reading HPLT `.jsonl.zst` shards.

## Quick start

From the repository root (with your environment activated):

```bash
python src/download_phase2_data.py
```

This will:

1. Fetch HPLT map files for the default language set.
2. Download the first 2 shards per language (highest-quality bins) and filter by register.
3. Normalize all text to NFC and write up to **100,000** lines to:
   - **`work/data/phase2/corpus_utf8_nfc.txt`**

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output_dir` | `work/data/phase2` | Directory for the output file. |
| `--max_samples` | `100000` | Target number of samples (lines). |
| `--languages` | (20 languages) | Space-separated list of HPLT language codes (e.g. `eng_Latn spa_Latn fra_Latn`). |
| `--max_shards_per_lang` | `2` | Max shards to download per language (limits download size). |

### Examples

- **Small test run (e.g. 1000 samples, 2 languages):**
  ```bash
  python src/download_phase2_data.py --max_samples 1000 --languages eng_Latn spa_Latn --max_shards_per_lang 1 --output_dir work/data/phase2_test
  ```

- **Custom output path and 50k samples:**
  ```bash
  python src/download_phase2_data.py --output_dir /path/to/data --max_samples 50000
  ```

- **More shards per language (larger download, more data per lang):**
  ```bash
  python src/download_phase2_data.py --max_shards_per_lang 4
  ```

## Default languages

The script uses a fixed set of ~20 languages for balance, including for example:  
`eng_Latn`, `spa_Latn`, `fra_Latn`, `deu_Latn`, `cmn_Hans`, `jpn_Jpan`, `arb_Arab`, `hin_Deva`, `por_Latn`, `rus_Cyrl`, `ita_Latn`, `kor_Hang`, `tha_Thai`, `vie_Latn`, `tur_Latn`, `pol_Latn`, `ind_Latn`, `nld_Latn`, `ell_Grek`, `heb_Hebr`.

Override with `--languages` to use a different set of HPLT codes.

## Output format

- **Path**: `work/data/phase2/corpus_utf8_nfc.txt` (or `{output_dir}/corpus_utf8_nfc.txt`).
- **Encoding**: UTF-8.
- **Content**: One text sample per line; all text is NFC-normalized.
- **Use**: Suitable for Phase 3 training or evaluation; compatible with byte-level models (e.g. Bolmo) and consistent Unicode handling.

## Notes

- HPLT data is hosted at Sigma2 NIRD; downloads may be slow depending on your network. Use `--max_shards_per_lang 1` for a faster, smaller test.
- If a language or shard fails (e.g. network error), the script skips it and continues with the next; check stderr for warnings.
- Documents without register metadata are kept by default; documents with register labels are filtered by conversational/interactive scores (see `CONVERSATIONAL_REGISTER_KEYS` and `REGISTER_THRESHOLD` in `src/download_phase2_data.py`).
