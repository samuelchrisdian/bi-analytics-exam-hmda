# Triage KPR: Prediksi Penolakan Aplikasi KPR (HMDA Colorado 2024–2025)

Kode, *notebook*, dan prototipe *dashboard* untuk Final Exam ISYS8036042 *Business Intelligence and Analytics*,
MMSI BINUS Graduate Program (Samuel Chrisdian, 2802939083). Branch ini hanya berisi aplikasi dan kode;
naskah laporan dan materi presentasi ada di branch lain.

Model memprediksi apakah aplikasi KPR akan **ditolak lender** (`action_taken` 3 = 1; 1 atau 2 = 0), lalu
menerjemahkan probabilitas ke tiga *risk tier* untuk *triage underwriting*. Target adalah keputusan lender,
bukan *default* aktual.

## Struktur

```
app/dashboard.py        Streamlit: scoring satu aplikasi, antrean triage (batch), kinerja model
src/                    config, data_prep, features, train, evaluate, explain, score, monitor, reporting
notebooks/              02–10 (format jupytext .py + .ipynb hasil eksekusi), urut tahap CRISP-DM
tests/                  pytest untuk modul src/ dan dashboard
reports/figures|tables  keluaran notebook (PNG 300 dpi, CSV + MD)
reports/evidence        tangkapan layar HMDA Data Browser (rantai dokumentasi data)
reports/diagrams        sumber Mermaid diagram (make diagrams)
models/                 final_threshold.json, shap_background.parquet (model .joblib dibuat ulang)
```

## Menjalankan

1. Python 3.11:

   ```bash
   python3.11 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   .venv/bin/pip install -e .
   ```

2. Unduh data dari HMDA Data Browser (https://ffiec.cfpb.gov/data-browser/): negara bagian Colorado,
   *action taken* 1–3, *loan purpose* home purchase; tahun 2024 (dataset *One Year*) dan 2025 (*Snapshot*).
   Simpan sebagai `data/raw/hmda_CO_2024.csv` dan `data/raw/hmda_CO_2025.csv`. Data tidak disertakan di repositori.

3. Jalankan seluruh *pipeline* (notebook 02 sampai 10 berurutan; membuat data olahan, model, tabel, dan gambar):

   ```bash
   make pipeline          # atau satu per satu: make nb-05_training
   ```

4. Buka *dashboard*:

   ```bash
   make dashboard
   ```

5. Test: `make test`.

## Catatan

- `RANDOM_STATE = 42`; semua *preprocessing* di-*fit* hanya pada data latih di dalam `sklearn.Pipeline`.
- Atribut *protected* (ras, etnis, jenis kelamin, usia) tidak dipakai sebagai prediktor, hanya untuk uji *fairness*.
- Data 2025 dipakai sebagai *out-of-time* dan tidak dipakai untuk *tuning* maupun pemilihan *threshold*.
