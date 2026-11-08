# Evidence Tahap 2 — Sourcing Data HMDA

Diambil 6 November 2026 dengan Chrome *headless* (Playwright), viewport 1440×900, skala 2×.
Hasil unduhan lewat tombol **Download Dataset** di web identik byte-per-byte dengan `data/raw/hmda_CO_{2024,2025}.csv`.

| File | Usulan caption | Sumber |
|---|---|---|
| e2_01_ffiec_home.png | Halaman utama situs HMDA FFIEC | https://ffiec.cfpb.gov/ |
| e2_02_data_browser.png | Halaman HMDA Data Browser | https://ffiec.cfpb.gov/data-browser/ |
| e2_03_filter_CO_2024.png | Filter Data Browser: Colorado 2024, *action taken* 1–3, *home purchase* | https://ffiec.cfpb.gov/data-browser/data/2024?category=states&items=CO&actions_taken=1,2,3&loan_purposes=1 |
| e2_04_summary_CO_2024.png | Ringkasan dan tombol unduh data Colorado 2024 (93.655 baris, 99 kolom) | idem |
| e2_05_filter_CO_2025.png | Filter Data Browser: Colorado 2025, *action taken* 1–3, *home purchase* | https://ffiec.cfpb.gov/data-browser/data/2025?category=states&items=CO&actions_taken=1,2,3&loan_purposes=1 |
| e2_06_summary_CO_2025.png | Ringkasan dan tombol unduh data Colorado 2025 (93.893 baris, 99 kolom) | idem |
| e2_07_dict_halaman.png | Halaman *Public HMDA – LAR Data Fields* | https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields |
| e2_08_dict_action_taken.png | Definisi kode `action_taken` (variabel target) | idem |
| e2_09–e2_15_dict_*.png | Definisi kode kolom filter scope: `loan_type`, `loan_purpose`, `lien_status`, `reverse_mortgage`, `open-end_line_of_credit`, `business_or_commercial_purpose`, `occupancy_type` | idem |

Catatan: data 2024 berasal dari dataset *One Year*, data 2025 dari dataset *Snapshot* (tertulis di e2_03 dan e2_05).
