import argparse
import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

# ---------- stałe ----------
FILL_ALPHA_ZERO = 0.25  # przezroczystość wypełnienia do zera
FILL_ALPHA_BETWEEN = 0.15  # przezroczystość wypełnienia między seriami
LINE_WIDTH = 2
MARKER_SIZE = 6
ANIMATION_INTERVAL = 80
MARGIN_FACTOR = 0.05

# ---------- logowanie ----------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------- narzędzia pomocnicze ----------
def find_column(df, candidates):
    """Zwraca pierwszą kolumnę z listy kandydatów, która istnieje w DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def set_ax_ylim(ax, series_df, cols):
    """Ustawia zakres osi Y na podstawie danych z wybranych kolumn."""
    if not cols:
        return
    data = series_df[cols].values
    ymin, ymax = np.nanmin(data), np.nanmax(data)
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    margin = MARGIN_FACTOR * (ymax - ymin)
    low, high = ymin - margin, ymax + margin
    if low >= high:
        low, high = min(ymin, ymax) - 1, max(ymin, ymax) + 1
    ax.set_ylim(low, high)


# ---------- wczytywanie pliku ----------
def load_data(file_path):
    """Wczytuje arkusz Excela, identyfikuje kolumnę czasu oraz kolumny z wartościami."""
    logger.info("Wczytywanie pliku: %s", file_path)
    df = pd.read_excel(file_path, sheet_name=0)

    # Kolumna czasu
    time_candidates = ["Time", "time", "Date", "date", "Data", "data"]
    time_col = find_column(df, time_candidates)
    if time_col is None:
        # Heurystyka – sprawdzamy, czy pierwsza kolumna wygląda na datę
        first_col = df.columns[0]
        sample = df[first_col].dropna().astype(str).iloc[0]
        try:
            pd.to_datetime(sample, dayfirst=True)
            time_col = first_col
            logger.info("Kolumna czasu wykryta heurystycznie: '%s'", time_col)
        except Exception:
            pass

    # Kolumny wartości – wszystkie numeryczne oprócz czasu
    value_cols = [c for c in df.columns if c != time_col]
    series_data = {}
    for col in value_cols:
        s = df[col].astype(str).str.replace(",", ".").str.strip()
        numeric = pd.to_numeric(s, errors="coerce")
        series_data[col] = numeric

    if time_col:
        time_series = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
        data = pd.DataFrame({"__time__": time_series, **series_data})
        data = (
            data.dropna(subset=["__time__"])
            .dropna(how="all", subset=value_cols)
            .reset_index(drop=True)
        )
        x = data["__time__"]
        series_df = data[value_cols]
    else:
        data = pd.DataFrame(series_data).dropna(how="all").reset_index(drop=True)
        x = np.arange(len(data))
        series_df = data

    if series_df.empty:
        raise ValueError("Brak poprawnych danych liczbowych.")

    return x, series_df, value_cols


# ---------- wybór kolumn ----------
def choose_columns(series_df):
    """Wyznacza domyślne kolumny dla obu wykresów."""
    all_cols = list(series_df.columns)
    logger.info("Dostępne kolumny: %s", all_cols)

    # Górny wykres – preferowane pary
    top_candidates = [
        ["PV po regulacji [kW]", "PV bez regulacji [kW]"],
        ["PV po regulacji", "PV bez regulacji"],
    ]
    top_cols = None
    for cand in top_candidates:
        top_cols = [c for c in cand if c in all_cols]
        if top_cols:
            break
    if not top_cols and len(all_cols) >= 2:
        top_cols = all_cols[:2]  # fallback: dwie pierwsze kolumny

    # Dolny wykres – preferowane pary
    bot_candidates = [
        ["Moc po regulacji [kW]", "Moc bez regulacji [kW]"],
        ["Moc po regulacji", "Moc bez regulacji"],
    ]
    bottom_cols = None
    for cand in bot_candidates:
        bottom_cols = [c for c in cand if c in all_cols]
        if bottom_cols:
            break
    if not bottom_cols and len(all_cols) >= 4:
        bottom_cols = all_cols[2:4]  # fallback: kolejne dwie kolumny
    elif not bottom_cols and len(all_cols) == 2:
        bottom_cols = all_cols  # tylko dwie kolumny, pokaż tę samą parę na dole

    return top_cols, bottom_cols


# ---------- konfiguracja wykresów ----------
def setup_axes():
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1]}
    )
    ax_top.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax_bot.axhline(0, color="gray", linewidth=1, linestyle="--")
    return fig, ax_top, ax_bot


# ---------- klasa animatora ----------
class DualAnimator:
    def __init__(self, fig, ax_top, ax_bot, x, series_df, top_cols, bottom_cols):
        self.fig = fig
        self.ax_top = ax_top
        self.ax_bot = ax_bot
        self.x = x
        self.series_df = series_df
        self.top_cols = top_cols
        self.bottom_cols = bottom_cols
        self.is_time = isinstance(x, pd.Series)

        if self.is_time:
            fmt = mdates.DateFormatter("%H:%M")
            self.ax_top.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            self.ax_top.xaxis.set_major_formatter(fmt)

        self.colors = itertools.cycle(plt.cm.tab10.colors)

        # Tworzenie linii i kropek
        self.lines_top, self.dots_top = self._create_artists(ax_top, top_cols)
        self.lines_bot, self.dots_bot = self._create_artists(ax_bot, bottom_cols)

        # Opisy osi
        ax_top.set_ylabel("Wartość [kW]")
        ax_bot.set_ylabel("Wartość [kW]")
        ax_bot.set_xlabel("Czas" if self.is_time else "Indeks")
        ax_top.set_title("Produkcja PV: " + ", ".join(top_cols))
        if bottom_cols:
            ax_bot.set_title("Moc czynna obiektu: " + ", ".join(bottom_cols))
        ax_top.legend(loc="upper left")
        if bottom_cols:
            ax_bot.legend(loc="lower left")

        # Zakresy osi
        set_ax_ylim(ax_top, series_df, top_cols)
        set_ax_ylim(ax_bot, series_df, bottom_cols)

        if self.is_time:
            ax_top.set_xlim(self.x.iloc[0], self.x.iloc[-1])
            self.fig.autofmt_xdate()
        else:
            ax_top.set_xlim(0, len(series_df) - 1)

        # Kolumna bazowa dla wypełnienia do zera (pierwsza seria każdego wykresu)
        self.actual_top = top_cols[0] if top_cols else None
        self.actual_bot = bottom_cols[0] if bottom_cols else None

        # Przechowalnia aktualnych wypełnień
        self.fills = {
            "top_zero": {"pos": None, "neg": None},
            "top_between": {"pos": None, "neg": None},
            "bot_zero": {"pos": None, "neg": None},
            "bot_between": {"pos": None, "neg": None},
        }

    def _create_artists(self, ax, cols):
        lines, dots = [], []
        for col in cols:
            color = next(self.colors)
            (ln,) = ax.plot([], [], lw=LINE_WIDTH, color=color, label=col, zorder=3)
            (dt,) = ax.plot([], [], "o", color=color, markersize=MARKER_SIZE, zorder=4)
            lines.append(ln)
            dots.append(dt)
        return lines, dots

    def init_anim(self):
        all_artists = self.lines_top + self.lines_bot + self.dots_top + self.dots_bot
        for artist in all_artists:
            artist.set_data([], [])
        return all_artists

    def update_anim(self, i):
        # Przygotowanie wycinka danych do bieżącego indeksu
        if self.is_time:
            xi = self.x.iloc[: i + 1]
            x_point = self.x.iloc[i]
        else:
            xi = np.arange(i + 1)
            x_point = i

        # Aktualizacja linii i punktów
        self._update_series(
            self.top_cols, self.lines_top, self.dots_top, xi, x_point, i
        )
        self._update_series(
            self.bottom_cols, self.lines_bot, self.dots_bot, xi, x_point, i
        )

        # Usuwanie starych wypełnień
        for grp in self.fills.values():
            for key in ("pos", "neg"):
                if grp[key] is not None:
                    grp[key].remove()
                    grp[key] = None

        # ---------- górny wykres ----------
        if self.actual_top is not None:
            y_act = self.series_df[self.actual_top].iloc[: i + 1].to_numpy(float)
            self._fill_vs_zero(self.ax_top, xi, y_act, "top_zero")
            # Wypełnienie między dwiema seriami (o ile istnieją)
            if len(self.top_cols) >= 2:
                y_a = self.series_df[self.top_cols[0]].iloc[: i + 1].to_numpy(float)
                y_b = self.series_df[self.top_cols[1]].iloc[: i + 1].to_numpy(float)
                self._fill_between_series(self.ax_top, xi, y_a, y_b, "top_between")

        # ---------- dolny wykres ----------
        if self.actual_bot is not None:
            y_act = self.series_df[self.actual_bot].iloc[: i + 1].to_numpy(float)
            self._fill_vs_zero(self.ax_bot, xi, y_act, "bot_zero")
            # To samo wypełnienie między seriami – PRZENIESIONE POZA WARUNEK PREDYKCJI
            if len(self.bottom_cols) >= 2:
                y_a = self.series_df[self.bottom_cols[0]].iloc[: i + 1].to_numpy(float)
                y_b = self.series_df[self.bottom_cols[1]].iloc[: i + 1].to_numpy(float)
                self._fill_between_series(self.ax_bot, xi, y_b, y_a, "bot_between")

        return self.lines_top + self.lines_bot + self.dots_top + self.dots_bot

    def _update_series(self, cols, lines, dots, xi, x_point, i):
        for idx, col in enumerate(cols):
            yi = self.series_df[col].iloc[: i + 1].to_numpy(float)
            lines[idx].set_data(xi, yi)
            dots[idx].set_data([x_point], [self.series_df[col].iloc[i]])

    def _fill_vs_zero(self, ax, xi, y_series, fill_key):
        self.fills[fill_key]["pos"] = ax.fill_between(
            xi,
            y_series,
            0,
            where=(y_series >= 0),
            interpolate=True,
            color="green",
            alpha=FILL_ALPHA_ZERO,
            zorder=1,
        )
        self.fills[fill_key]["neg"] = ax.fill_between(
            xi,
            y_series,
            0,
            where=(y_series <= 0),
            interpolate=True,
            color="red",
            alpha=FILL_ALPHA_ZERO,
            zorder=1,
        )

    def _fill_between_series(self, ax, xi, y_a, y_b, fill_key):
        """Wypełnienie obszaru między dwiema seriami: zielone, gdy pierwsza >= druga, czerwone w przeciwnym razie."""
        self.fills[fill_key]["pos"] = ax.fill_between(
            xi,
            y_a,
            y_b,
            where=(y_a >= y_b),
            interpolate=True,
            color="green",
            alpha=FILL_ALPHA_BETWEEN,
            zorder=1,
        )
        self.fills[fill_key]["neg"] = ax.fill_between(
            xi,
            y_a,
            y_b,
            where=(y_a <= y_b),
            interpolate=True,
            color="red",
            alpha=FILL_ALPHA_BETWEEN,
            zorder=1,
        )


# ---------- program główny ----------
def main():
    parser = argparse.ArgumentParser(
        description="Animacja danych PV z dwoma wykresami."
    )
    parser.add_argument(
        "file", nargs="?", help="Plik Excel (domyślnie: dane_pv.xlsx lub dane.xlsx)"
    )
    args = parser.parse_args()

    if args.file:
        fname = args.file
    else:
        for candidate in ["dane.xlsx", "dane_pv.xlsx"]:
            if os.path.exists(candidate):
                fname = candidate
                break
        else:
            raise FileNotFoundError(
                "Nie znaleziono pliku 'dane_pv.xlsx' ani 'dane.xlsx'. Podaj nazwę pliku jako argument."
            )

    x, series_df, _ = load_data(fname)
    top_cols, bottom_cols = choose_columns(series_df)

    fig, ax_top, ax_bot = setup_axes()
    animator = DualAnimator(fig, ax_top, ax_bot, x, series_df, top_cols, bottom_cols)

    ani = FuncAnimation(
        fig,
        animator.update_anim,
        frames=len(series_df),
        init_func=animator.init_anim,
        blit=False,
        interval=ANIMATION_INTERVAL,
    )
    # Zapis MP4 – ffmpeg jest teraz dostępny globalnie
    # ani.save("animacja.mp4", writer="ffmpeg", fps=12, dpi=150)
    # ani.save("animacja.gif", writer="pillow", fps=12, dpi=100)

    logger.info("Animacja gotowa – zamknij okno, aby zakończyć.")
    plt.show()


if __name__ == "__main__":
    main()
