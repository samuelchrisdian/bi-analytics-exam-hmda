import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import config
from src.reporting import fmt_id, save_figure, save_table


@pytest.mark.parametrize("x,d,exp", [
    (0.8734, 3, "0,873"), (11000, 3, "11.000"), (1234567.891, 2, "1.234.567,89"),
    (-0.5, 1, "-0,5"), (np.int64(1500), 3, "1.500"), (float("nan"), 3, "-"), (None, 3, "-"),
])
def test_fmt_id(x, d, exp):
    assert fmt_id(x, d) == exp


def test_save_table_writes_csv_and_md(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TABLES", tmp_path)
    csv, md = save_table(pd.DataFrame({"m": ["auc"], "v": [0.8734], "n": [12000]}), "t6_demo")
    assert pd.read_csv(csv)["v"][0] == 0.8734
    text = md.read_text()
    assert "0,873" in text and "12.000" in text


def test_save_table_year_column_not_thousands(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TABLES", tmp_path)
    _, md = save_table(pd.DataFrame({"tahun": [2024], "n": [55002]}), "t2_demo")
    text = md.read_text()
    assert "2024" in text and "2.024" not in text and "55.002" in text


def test_save_table_rejects_bad_name(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TABLES", tmp_path)
    with pytest.raises(ValueError):
        save_table(pd.DataFrame({"a": [1]}), "hasil")


def test_save_figure_300dpi_no_title(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    ax.set_title("Judul")
    fig.suptitle("X")
    p = save_figure(fig, "f3_demo")
    assert round(Image.open(p).info["dpi"][0]) == 300
    assert ax.get_title() == ""
    assert fig._suptitle is None or fig._suptitle.get_text() == ""


def test_save_figure_indonesian_tick_format(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 0.5], [0, 200000])
    save_figure(fig, "f6_demo")
    fx, fy = ax.xaxis.get_major_formatter(), ax.yaxis.get_major_formatter()
    assert fx(0.25, 0) == "0,25" and fx(0.5, 0) == "0,5" and fx(0, 0) == "0"
    assert fy(100000, 0) == "100.000" and fy(-0.125, 0) == "-0,125"


def test_save_figure_rejects_bad_name(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES", tmp_path)
    fig, _ = plt.subplots()
    with pytest.raises(ValueError):
        save_figure(fig, "grafik")


def test_fmt_auto_decimals():
    from src.reporting import fmt_auto
    assert [fmt_auto(v) for v in (0.2, 0.001, 100.0, 0.0374, 15000)] == ["0,2", "0,001", "100", "0,037", "15.000"]


@pytest.mark.parametrize("raw,exp", [
    ("6.176 = Rasio pinjaman", "6,176 = Rasio pinjaman"), ("115000 = Nilai properti (USD)", "115.000 = Nilai properti (USD)"),
    ("+4.23", "+4,23"), ("−0.09", "−0,09"), ("f(x) = 6.522", "f(x) = 6,522"),
    ("Sum of 10 other features", "Jumlah 10 fitur lainnya"), ("15 other features", "15 fitur lainnya"),
    ("65 = Rasio utang", "65 = Rasio utang"),
])
def test_localize_text(raw, exp):
    from src.reporting import localize_text
    assert localize_text(raw) == exp


def test_localize_figure_updates_fixed_ticks_and_texts():
    from src.reporting import localize_figure
    fig, ax = plt.subplots()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["6.176 = A", "Sum of 3 other features"])
    t = ax.text(0, 0, "+4.23")
    localize_figure(fig)
    assert [l.get_text() for l in ax.get_yticklabels()] == ["6,176 = A", "Jumlah 3 fitur lainnya"]
    assert t.get_text() == "+4,23"


def test_localize_text_mathtext_comma():
    from src.reporting import localize_text
    assert localize_text("$ = 0.012$") == "$ = 0{,}012$"


def test_localize_figure_shared_axes_localized_once():
    from src.reporting import localize_figure
    fig, ax = plt.subplots()
    ax.set_yticks([0])
    ax.set_yticklabels(["115000 = A"])
    ax.twiny()
    ax.twiny()
    localize_figure(fig)
    assert ax.get_yticklabels()[0].get_text() == "115.000 = A"
