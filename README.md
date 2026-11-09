# Triage KPR: Prediksi Penolakan Aplikasi KPR (HMDA Colorado 2024–2025)

Kode program, *notebook*, dan prototipe *dashboard* untuk Final Exam ISYS8036042 *Business Intelligence
and Analytics*, MMSI BINUS Graduate Program. Penulis: Samuel Chrisdian (2802939083).

Model memprediksi apakah aplikasi KPR akan **ditolak lender** (`action_taken` 3 → 1; 1 atau 2 → 0),
lalu menerjemahkan probabilitas menjadi tiga *risk tier* untuk *triage underwriting*. Target adalah
keputusan lender, bukan *default* aktual.

Berkas ini berisi kode saja. Data HMDA tidak disertakan karena ukurannya besar dan dapat diunduh ulang
secara publik (langkah 2). Tabel dan gambar hasil penelitian sudah tersedia di `reports/`, sehingga hasil
dapat diperiksa tanpa menjalankan ulang apa pun.

## Isi

| Folder / berkas | Isi |
|---|---|
| `src/` | Modul: `config` (konstanta, `RANDOM_STATE = 42`), `data_prep`, `features`, `train`, `evaluate`, `explain`, `score`, `monitor`, `reporting`, `dashboard_data` |
| `notebooks/` | Notebook per tahap CRISP-DM: sumber `.py` (format jupytext) dan `.ipynb` beserta keluaran eksekusi terakhir |
| `app/dashboard.py` | *Dashboard* Streamlit: *scoring* satu aplikasi, antrean *triage* (batch), kinerja model |
| `tests/` | Test `pytest` untuk modul `src/` dan *dashboard* |
| `reports/tables/`, `reports/figures/` | Keluaran notebook: tabel `t{tahap}_*.csv/.md` dan gambar `f{tahap}_*.png` |
| `reports/evidence/` | Tangkapan layar HMDA Data Browser (bukti asal data) beserta tautan filternya |
| `reports/diagrams/` | Sumber Mermaid diagram (dirender dengan `make diagrams`, butuh Node.js) |
| `models/` | `final_threshold.json` dan `shap_background.parquet`; berkas model `.joblib` dibuat ulang oleh notebook 05–06 |
| `requirements.txt` | Versi seluruh paket Python yang dipakai (terkunci) |

| Notebook | Tahap CRISP-DM | Keluaran utama |
|---|---|---|
| `02_sourcing` | 2. *Sourcing of raw data* | Filter *scope*, profil kolom, kamus data (`t2_*`) |
| `03_preprocessing_eda` | 3. *Preprocessing & EDA* | Audit *leakage*, data latih/uji/OOT di `data/processed/` (`t3_*`, `f3_*`) |
| `05_training` | 4–5. Algoritma, *training*, *cross-validation* | `models/{dummy,logreg,rf,xgb}.joblib`, hasil CV, OOF (`t5_*`) |
| `06_model_selection` | 6. Pemilihan model dan *threshold* biaya | `models/final_model.joblib`, `final_threshold.json` (`t6_*`, `f6_*`) |
| `07_validation` | 7. Validasi, OOT 2025, SHAP, *fairness* | `t7_*`, `f7_*` |
| `08_deployment` | 8. *Risk tier*, aturan bisnis, contoh *scoring* | `t8_*`, `shap_background.parquet` |
| `09_monitoring` | 9. PSI dan KPI *monitoring* | `t9_*`, `f9_psi.png` |
| `10_recommendations` | 10. Dampak bisnis | `t10_impact` |

Tahap 1 (identifikasi masalah) dan sebagian Tahap 4 (kajian algoritma) berupa kajian tertulis di laporan,
sehingga tidak memiliki notebook.

## Reproduksi

### 1. Prasyarat

- Python **3.11** (paket dikunci untuk 3.11; `pyproject.toml` membatasi `>=3.11,<3.12`).
- RAM minimal 8 GB dan ruang disk kosong sekitar 3 GB (paket Python ± 2 GB, data ± 70 MB).
- `make` (Linux/macOS). Di Windows, jalankan perintah setara pada bagian "Tanpa make" di bawah.

### 2. Unduh data HMDA

Data diunduh dari HMDA Data Browser (FFIEC/CFPB), publik dan tanpa login. Buka tautan berikut, klik
**Download Dataset**, lalu simpan berkas CSV dengan nama persis seperti kolom terakhir:

| Tahun | Dataset | Tautan filter (Colorado, *action taken* 1–3, *loan purpose* home purchase) | Simpan sebagai |
|---|---|---|---|
| 2024 | *One Year* (final) | https://ffiec.cfpb.gov/data-browser/data/2024?category=states&items=CO&actions_taken=1,2,3&loan_purposes=1 | `data/raw/hmda_CO_2024.csv` |
| 2025 | *Snapshot* | https://ffiec.cfpb.gov/data-browser/data/2025?category=states&items=CO&actions_taken=1,2,3&loan_purposes=1 | `data/raw/hmda_CO_2025.csv` |

Berkas yang dipakai penelitian (diunduh 6 November 2026):

| Berkas | Baris data | Kolom | SHA-256 |
|---|---|---|---|
| `hmda_CO_2024.csv` | 93.655 | 99 | `199fb01b086ded03fee186fd8b24781845777839187d6c9f24dbe766f26aac0b` |
| `hmda_CO_2025.csv` | 93.893 | 99 | `6b2b1da2d6defe3ff42883331111d36335c7b4ed6068b619d8504175779fa8e9` |

Periksa dengan `sha256sum data/raw/*.csv`. Dataset 2024 bersifat final; dataset 2025 adalah *Snapshot* yang
dapat direvisi FFIEC, sehingga unduhan baru bisa sedikit berbeda dan angka OOT 2025 dapat bergeser tipis.

### 3. Instal

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

### 4. Jalankan *pipeline*

```bash
make pipeline
```

Perintah ini mengeksekusi notebook 02 sampai 10 berurutan (sekitar 14 menit pada laptop Intel Core i7-13620H
16 *thread*, RAM 16 GB; notebook 05 paling lama, sekitar 9 menit; instalasi paket sekitar 1 menit). Setiap notebook ditulis ulang beserta keluarannya, dan
`data/`, `models/`, `reports/tables/`, serta `reports/figures/` dibuat ulang. Notebook juga dapat dijalankan
satu per satu, misalnya `make nb-05_training`, asalkan urutannya dipertahankan.

### 5. Cocokkan hasil

Dengan data di langkah 2, angka berikut harus sama (`reports/tables/t6_threshold.md`,
`reports/tables/t7_oot_performance.md`, `models/final_threshold.json`):

| Besaran | Nilai |
|---|---|
| Data setelah filter *scope* | 55.002 aplikasi (2024), 53.241 (2025) |
| Model final, *threshold* | XGBoost terkalibrasi isotonik, *threshold* 0,17 (C_FN = 5, C_FP = 1) |
| Biaya OOF per 1.000 aplikasi | XGBoost 180,93; Random Forest 181,20; Logistic Regression 242,20; Dummy 623,03 |
| Data uji 2024 | ROC-AUC 0,907; PR-AUC 0,790; *recall* 0,759; *precision* 0,692; biaya 181,07 per 1.000 |
| OOT 2025 | ROC-AUC 0,891; PR-AUC 0,729; *recall* 0,718; biaya 179,15 per 1.000 |

Uji reproduksi dari salinan bersih (9 November 2026) menghasilkan 47 dari 48 tabel `reports/tables/*.csv` yang
identik byte-per-byte; satu-satunya perbedaan adalah kolom waktu *tuning* di `t5_best_params.csv`.

### 6. *Dashboard* dan test

```bash
make dashboard   # Streamlit di http://localhost:8501 (butuh langkah 4 selesai)
make test        # pytest
```

### Tanpa `make` (Windows)

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
.venv\Scripts\jupytext --to ipynb notebooks\02_sourcing.py
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\02_sourcing.ipynb
```

Ulangi dua perintah terakhir untuk `03_preprocessing_eda`, `05_training`, `06_model_selection`,
`07_validation`, `08_deployment`, `09_monitoring`, dan `10_recommendations`. *Dashboard*:
`.venv\Scripts\streamlit run app\dashboard.py`; test: `.venv\Scripts\python -m pytest -q`.

## Catatan metodologi

- `RANDOM_STATE = 42` dipakai di semua komponen acak; *tuning* memakai `n_jobs = 4`.
- Semua *preprocessing* (imputasi, *capping*, *scaling*, *encoding*) berada di dalam `sklearn.Pipeline`
  dan di-*fit* hanya pada data latih.
- Data uji 2024 dan OOT 2025 tidak dipakai untuk *tuning* maupun pemilihan *threshold*.
- Atribut *protected* (ras, etnis, jenis kelamin, usia) tidak dipakai sebagai prediktor; disimpan terpisah
  hanya untuk uji *fairness*.
