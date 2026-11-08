"""Prototipe dashboard triage KPR. Jalankan: make dashboard"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.dashboard_data import app_to_form, form_to_app, load_metrics, sample_oot
from src.reporting import fmt_id
from src.score import DTI_LIMIT, LTV_LIMIT, TIERS, contributions, load_bundle, score_application, score_batch

TIER_ACTION = {
    "Rendah": "Fast-track (verifikasi dokumen ringan)",
    "Sedang": "Review standar",
    "Tinggi": "Review senior underwriter",
}
TIER_ICON = {"Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴"}
RULE_TEXT = {
    "melewati_batas_kebijakan": f"Melewati batas kebijakan (DTI > {DTI_LIMIT}% atau LTV > {LTV_LIMIT}%) → tier dinaikkan ke Tinggi",
    "data_tidak_lengkap": "Data inti kosong (pendapatan, nilai properti, DTI, atau LTV) → tier minimal Sedang",
}
SUBMISSION_LABELS = {"1": "1 — langsung ke lender", "2": "2 — tidak langsung (lewat pihak lain)"}
CONFORMING_LABELS = {"C": "C — conforming", "NC": "NC — nonconforming", "U": "U — tidak ditentukan"}
RULE_SHORT = {"": "-", "melewati_batas_kebijakan": "Batas DTI/LTV", "data_tidak_lengkap": "Data tidak lengkap"}
COLUMN_LABELS = {
    "data": "Data", "model": "Model", "n": "n", "proporsi_ditolak": "Proporsi ditolak", "threshold": "Threshold",
    "jenis_threshold": "Jenis threshold", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "specificity": "Specificity", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier",
    "cost_per_1000": "Biaya/1.000", "tier": "Tier", "objek": "Objek", "pembanding": "Pembanding", "psi": "PSI",
    "status": "Status", "atribut": "Atribut", "grup": "Grup", "approval_rate_pred": "Approval prediksi",
    "approval_rate_aktual": "Approval aktual", "dir": "DIR prediksi", "dir_aktual": "DIR aktual", "eod": "EOD",
    "catatan": "Catatan",
}
ID_LOCALE = {"number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["", ""]}}
RISK_UP, RISK_DOWN = "#e34948", "#2a78d6"
DEFAULT_FORM = {
    "loan_amount": 300000.0, "income": 90.0, "property_value": 400000.0, "debt_to_income_ratio": 33.0,
    "loan_term": 360.0, "tract_to_msa_income_percentage": 100.0, "ffiec_msa_md_median_family_income": 110000.0,
    "submission_of_application": "1", "conforming_loan_limit": "C", "has_coapplicant": False,
}
REQUIRED_FILES = {
    config.MODELS / "final_model.joblib": "make nb-06_model_selection",
    config.MODELS / "final_threshold.json": "make nb-06_model_selection",
    config.DATA_INTERIM / f"hmda_filtered_{config.YEAR_OOT}.parquet": "make nb-02_sourcing",
}


def pct(x: float, decimals: int = 2) -> str:
    return f"{fmt_id(x, decimals)}%"


@st.cache_resource
def get_bundle() -> dict:
    return load_bundle()


@st.cache_data
def get_metrics() -> dict[str, pd.DataFrame]:
    return load_metrics()


@st.cache_data
def get_batch(n: int) -> pd.DataFrame:
    raw = sample_oot(n)
    return raw.join(score_batch(raw, get_bundle()))


def check_files() -> None:
    missing = [f"`{p.relative_to(config.ROOT)}` → jalankan `{cmd}`" for p, cmd in REQUIRED_FILES.items() if not p.exists()]
    try:
        get_metrics()
    except FileNotFoundError as e:
        missing.append(str(e))
    if missing:
        st.error("File yang dibutuhkan dashboard belum ada:\n\n" + "\n".join(f"- {m}" for m in missing))
        st.stop()


def init_state() -> None:
    for field, value in DEFAULT_FORM.items():
        st.session_state.setdefault(f"f_{field}", value)
    st.session_state.setdefault("random_draws", 0)


def score_current_form(actual: str | None = None) -> None:
    form = {field: st.session_state[f"f_{field}"] for field in DEFAULT_FORM}
    app = form_to_app(form)
    bundle = get_bundle()
    st.session_state["result"] = {
        **score_application(app, bundle),
        "kontribusi": contributions(app, bundle, top=8),
        "form": form,
        "aktual": actual,
    }


def load_random_example() -> None:
    seed = config.RANDOM_STATE + st.session_state["random_draws"]
    st.session_state["random_draws"] += 1
    row = sample_oot(1, seed=seed).iloc[0]
    for field, value in app_to_form(row.to_dict()).items():
        st.session_state[f"f_{field}"] = value
    score_current_form(actual=row["aktual"])


def contribution_chart(contrib: pd.DataFrame) -> alt.Chart:
    data = contrib.assign(
        arah=lambda d: d["shap"].gt(0).map({True: "Menaikkan risiko", False: "Menurunkan risiko"}),
        nilai=lambda d: d["shap"].map(lambda v: fmt_id(v, 3)),
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height=16)
        .encode(
            x=alt.X("shap:Q", title="Kontribusi SHAP (log-odds penolakan)"),
            y=alt.Y("fitur:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("arah:N", title=None,
                            scale=alt.Scale(domain=["Menaikkan risiko", "Menurunkan risiko"], range=[RISK_UP, RISK_DOWN]),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("fitur:N", title="Fitur"), alt.Tooltip("nilai:N", title="SHAP"),
                     alt.Tooltip("arah:N", title="Arah")],
        )
        .properties(height=36 * len(data) + 40)
        .configure(locale=ID_LOCALE)
    )


def render_form() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Jumlah pinjaman (USD)", min_value=0.0, step=5000.0, format="%.0f", key="f_loan_amount")
        st.number_input("Pendapatan tahunan (ribu USD)", min_value=0.0, step=5.0, format="%.0f", key="f_income")
        st.number_input("Nilai properti (USD)", min_value=0.0, step=5000.0, format="%.0f", key="f_property_value")
    with c2:
        st.number_input("Rasio utang terhadap pendapatan / DTI (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.0f",
                        key="f_debt_to_income_ratio", help="Dikonversi otomatis ke rentang DTI HMDA.")
        st.number_input("Tenor (bulan)", min_value=0.0, step=12.0, format="%.0f", key="f_loan_term")
        st.checkbox("Ada co-applicant", key="f_has_coapplicant")
    with c3:
        st.number_input("Pendapatan tract terhadap MSA (%)", min_value=0.0, step=1.0, format="%.2f", key="f_tract_to_msa_income_percentage")
        st.number_input("Median pendapatan keluarga MSA (USD)", min_value=0.0, step=1000.0, format="%.0f",
                        key="f_ffiec_msa_md_median_family_income")
        st.selectbox("Jalur pengajuan", list(SUBMISSION_LABELS), format_func=SUBMISSION_LABELS.get,
                     key="f_submission_of_application", index=None, placeholder="Tidak diketahui")
        st.selectbox("Batas conforming loan", list(CONFORMING_LABELS), format_func=CONFORMING_LABELS.get,
                     key="f_conforming_loan_limit", index=None, placeholder="Tidak diketahui")


def render_result(r: dict) -> None:
    form = r["form"]
    loan, prop = form["loan_amount"], form["property_value"]
    ltv = loan / prop * 100 if loan and prop else None
    tier = r["tier"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilitas penolakan", pct(r["proba"] * 100, 1))
    m2.metric("Risk tier", f"{TIER_ICON[tier]} {tier}")
    m3.metric("LTV hitungan", "-" if ltv is None else pct(ltv, 1))
    m4.metric("Keputusan lender aktual", r["aktual"] or "-",
              help="Hanya untuk contoh acak dari data OOT 2025; tidak dipakai model.")
    st.markdown(f"**Aksi:** {TIER_ACTION[tier]}")
    if r["aturan"]:
        st.warning(f"Aturan bisnis terpicu: {RULE_TEXT[r['aturan']]}")
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Tiga reason codes teratas** (faktor yang paling menaikkan risiko)")
        if r["reason_codes"]:
            st.markdown("\n".join(f"{i}. {code}" for i, code in enumerate(r["reason_codes"], 1)))
        else:
            st.markdown("Tidak ada faktor yang menaikkan risiko.")
        st.caption("Keputusan akhir tetap di tangan underwriter; tidak ada auto-reject.")
    with right:
        st.altair_chart(contribution_chart(r["kontribusi"]), width="stretch")


def tab_scoring() -> None:
    st.subheader("Scoring satu aplikasi")
    st.caption("Isi data aplikasi, atau ambil contoh acak dari data OOT 2025, lalu klik Skor.")
    b1, b2, _ = st.columns([1, 2, 3])
    b1.button("Skor aplikasi", key="btn_score", type="primary", on_click=score_current_form)
    b2.button("Ambil contoh acak OOT 2025", key="btn_random", on_click=load_random_example)
    render_form()
    if "result" in st.session_state:
        st.divider()
        render_result(st.session_state["result"])


def tab_batch() -> None:
    st.subheader("Antrean triage (batch)")
    st.caption("Simulasi satu periode: aplikasi OOT 2025 acak diskor sekaligus dan dibagi ke antrean per tier.")
    n = st.slider("Jumlah aplikasi", min_value=100, max_value=2000, value=500, step=100, key="batch_n")
    df = get_batch(n)
    cols = st.columns(3)
    for col, tier in zip(cols, TIERS):
        sub = df[df["tier"] == tier]
        rate = (sub["aktual"] == "ditolak").mean() * 100 if len(sub) else 0.0
        col.metric(f"{TIER_ICON[tier]} {tier} — {TIER_ACTION[tier]}", f"{fmt_id(len(sub))} aplikasi",
                   help="Persentase antrean dan tingkat tolak aktual di antrean ini.")
        col.caption(f"{pct(len(sub) / len(df) * 100, 1)} dari antrean · tolak aktual {pct(rate, 1)}")
    st.caption(f"Tingkat tolak aktual seluruh sampel: {pct((df['aktual'] == 'ditolak').mean() * 100, 1)}")

    chosen = st.multiselect("Filter tier", TIERS, default=["Tinggi"], key="batch_filter")
    queue = df[df["tier"].isin(chosen)].sort_values("proba", ascending=False)
    view = pd.DataFrame({
        "Probabilitas (%)": queue["proba"].map(lambda v: fmt_id(v * 100, 1)),
        "Tier": queue["tier"],
        "Aturan": queue["aturan"].map(RULE_SHORT),
        "Pinjaman (USD)": pd.to_numeric(queue["loan_amount"], errors="coerce").map(lambda v: fmt_id(v, 0)),
        "Pendapatan (ribu USD)": queue["income"],
        "DTI": queue["debt_to_income_ratio"],
        "Aktual": queue["aktual"],
    })
    st.dataframe(view, hide_index=True, width="stretch", height=360)


def kpi_row(perf: pd.DataFrame) -> None:
    test, oot = perf.iloc[0], perf.iloc[1]
    cols = st.columns(4)
    for col, (key, label) in zip(cols, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall", "Recall"),
                                        ("precision", "Precision")]):
        delta = oot[key] - test[key]
        col.metric(f"{label} OOT 2025", fmt_id(oot[key], 3), delta=f"{fmt_id(delta, 3)} vs test 2024")


def fmt_table(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    def fmt_col(c: pd.Series) -> pd.Series:
        if pd.api.types.is_integer_dtype(c):
            return c.map(lambda v: fmt_id(int(v)))
        if pd.api.types.is_float_dtype(c):
            return c.map(lambda v: fmt_id(v, decimals))
        return c
    return df.apply(fmt_col).rename(columns=COLUMN_LABELS)


def tab_performance() -> None:
    m = get_metrics()
    st.subheader("Kinerja model final (XGBoost terkalibrasi)")
    kpi_row(m["oot_performance"])

    st.markdown("**Test 2024 vs OOT 2025**")
    st.dataframe(fmt_table(m["oot_performance"]), hide_index=True, width="stretch")
    st.markdown("**Perbandingan model (test 2024)**")
    st.dataframe(fmt_table(m["model_comparison"]), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Distribusi tier OOT 2025**")
        tiers = m["tier_distribution"][["tier", "pct_setelah_aturan", "denial_rate_setelah_aturan"]]
        tiers = tiers.rename(columns={"pct_setelah_aturan": "% aplikasi", "denial_rate_setelah_aturan": "Tolak aktual (%)"})
        st.dataframe(fmt_table(tiers, 2), hide_index=True, width="stretch")
    with right:
        st.markdown("**Stabilitas skor (PSI)**")
        st.dataframe(fmt_table(m["psi_score"], 4), hide_index=True, width="stretch")

    st.markdown("**Fairness (test 2024)** — grup dengan disparate impact ratio < 0,8 disorot; grup n kecil tidak dinilai")
    fair = m["fairness"]
    # Sama dengan laporan: penanda diambil dari kolom catatan, bukan dir < 0,8 mentah (grup n kecil dikecualikan).
    flagged = fair["catatan"].eq("DIR < 0,8")
    shown = fmt_table(fair[["atribut", "grup", "n", "approval_rate_pred", "approval_rate_aktual", "dir", "dir_aktual", "eod",
                            "catatan"]].fillna({"catatan": "-"}))
    st.dataframe(shown.style.apply(lambda row: ["background-color: rgba(227,73,72,0.18)" if flagged[row.name] else ""] * len(row),
                                   axis=1), hide_index=True, width="stretch")

    st.markdown("**Gambar dari laporan**")
    figs = [("f6_cost_curve.png", "Biaya per 1.000 aplikasi terhadap threshold"),
            ("f7_shap_summary.png", "Ringkasan SHAP"), ("f9_psi.png", "PSI fitur train vs OOT")]
    for col, (name, caption) in zip(st.columns(3), figs):
        path = config.FIGURES / name
        if path.exists():
            col.image(str(path), caption=caption)


def main() -> None:
    st.set_page_config(page_title="Triage KPR — Prototipe", page_icon="🏠", layout="wide")
    st.title("Triage KPR — Prototipe")
    st.info("Prototipe akademik (Bank X fiktif). Model memprediksi **keputusan lender** pada data HMDA Colorado, "
            "bukan gagal bayar aktual. Skor hanya untuk prioritas review; **tidak ada auto-reject**.")
    check_files()
    init_state()
    t1, t2, t3 = st.tabs(["Scoring aplikasi", "Antrean triage (batch)", "Kinerja model"])
    with t1:
        tab_scoring()
    with t2:
        tab_batch()
    with t3:
        tab_performance()


main()
