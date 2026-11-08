import numbers
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from src import config

_TABLE_NAME = re.compile(r"^t\d+_[a-z0-9_]+$")
_FIGURE_NAME = re.compile(r"^f\d+_[a-z0-9_]+$")
# Kolom berisi tahun/kode: angka tanpa titik ribuan.
PLAIN_INT_COLUMNS = {"tahun", "activity_year", "kode_action_taken"}


def fmt_id(x, decimals: int = 3) -> str:
    """Format angka gaya Indonesia: titik ribuan, koma desimal; NaN/None menjadi "-"."""
    if x is None or (isinstance(x, numbers.Number) and not isinstance(x, numbers.Integral) and np.isnan(x)):
        return "-"
    if isinstance(x, numbers.Integral):
        s = f"{int(x):,}"
    else:
        s = f"{float(x):,.{decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def _format_column(col: pd.Series, decimals: int) -> pd.Series:
    if pd.api.types.is_bool_dtype(col):
        return col
    if pd.api.types.is_integer_dtype(col):
        return col.map(lambda v: fmt_id(int(v)))
    if pd.api.types.is_float_dtype(col):
        return col.map(lambda v: fmt_id(v, decimals))
    return col


def save_table(df: pd.DataFrame, name: str, decimals: int = 3, index: bool = False) -> tuple[Path, Path]:
    if not _TABLE_NAME.match(name):
        raise ValueError(f"Nama tabel harus berpola t{{tahap}}_{{deskripsi}}: {name}")
    config.TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = config.TABLES / f"{name}.csv"
    md_path = config.TABLES / f"{name}.md"
    df.to_csv(csv_path, index=index)
    shown = df.reset_index() if index else df.copy()
    shown = shown.apply(lambda c: c.astype(str) if c.name in PLAIN_INT_COLUMNS else _format_column(c, decimals))
    # disable_numparse: tabulate akan membaca "12.000" sebagai 12.
    md_path.write_text(shown.to_markdown(index=False, disable_numparse=True) + "\n")
    return csv_path, md_path


def fmt_auto(v: float, _pos=None) -> str:
    """Format Indonesia dengan jumlah desimal secukupnya (maks. 3)."""
    if float(v).is_integer():
        return fmt_id(int(v))
    decimals = len(f"{v:.3f}".rstrip("0").split(".")[1])
    return fmt_id(v, decimals)


_DECIMAL_EN = re.compile(r"(?<![\w.,])(\d+)\.(\d+)")
_BIG_INT = re.compile(r"(?<![\w.,])\d{4,}(?![\w.,])")


def localize_text(s: str) -> str:
    """Ubah teks bawaan library (mis. SHAP) ke format Indonesia; anggap titik selalu desimal."""
    s = re.sub(r"Sum of (\d+) other features", r"Jumlah \1 fitur lainnya", s)
    s = re.sub(r"(\d+) other features", r"\1 fitur lainnya", s)
    s = _DECIMAL_EN.sub(r"\1,\2", s)
    s = _BIG_INT.sub(lambda m: fmt_id(int(m.group(0))), s)
    # Di mathtext, koma diberi spasi kecuali dibungkus {,}.
    return re.sub(r"\$[^$]*\$", lambda m: re.sub(r"(?<=\d),(?=\d)", "{,}", m.group(0)), s)


def localize_figure(fig: Figure) -> Figure:
    """Lokalisasi teks tetap pada gambar; dipanggil sekali per gambar (tidak idempoten)."""
    done = set()
    for ax in fig.axes:
        for axis, getter, setter in ((ax.yaxis, ax.get_yticklabels, ax.set_yticklabels),
                                     (ax.xaxis, ax.get_xticklabels, ax.set_xticklabels)):
            # Axes kembar berbagi formatter; proses sekali agar "115.000" tidak dibaca ulang sebagai desimal.
            fmt = axis.get_major_formatter()
            labels = [t.get_text() for t in getter()]
            if isinstance(fmt, ScalarFormatter) or not any(labels) or id(fmt) in done:
                continue
            setter([localize_text(s) for s in labels])
            done.update({id(fmt), id(axis.get_major_formatter())})
        for t in ax.texts:
            t.set_text(localize_text(t.get_text()))
    return fig


def save_figure(fig: Figure, name: str) -> Path:
    """Simpan PNG 300 dpi; judul dihapus karena caption ditulis di laporan."""
    if not _FIGURE_NAME.match(name):
        raise ValueError(f"Nama gambar harus berpola f{{tahap}}_{{deskripsi}}: {name}")
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(fmt_auto))
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", font="sans-serif", font_scale=10 / 12)
