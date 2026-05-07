import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- stałe konfiguracyjne ----------
FILL_ALPHA = 0.25
LINE_WIDTH = 2
MARKER_SIZE = 6
ANIMATION_INTERVAL = 80
MARGIN_FACTOR = 0.05

# ---------- logowanie ----------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------- wczytywanie danych ----------
def load_data(file_path):
    """Wczytuje plik Excel i zwraca (x, series_df) gdzie x to indeksy/daty,
    a series_df zawiera kolumny z danymi liczbowymi."""
    logger.info("Wczytywanie pliku: %s", file_path)
    df = pd.read_excel(file_path, sheet_name=0)
    logger.info("Dostępne kolumny: %s", list(df.columns))

    # Szukamy kolumny czasu
    time_candidates = ['Time', 'time', 'Date', 'date', 'Data', 'data']
    time_col = None
    for col in time_candidates:
        if col in df.columns:
            time_col = col
            break

    # Heurystyka: czy pierwsza kolumna to data?
    if time_col is None:
        try:
            sample = df.iloc[:, 0].dropna().astype(str).iloc[0]
            pd.to_datetime(sample, dayfirst=True)
            time_col = df.columns[0]
            logger.info("Kolumna czasu wykryta heurystycznie: '%s'", time_col)
        except Exception:
            pass

    # Wszystkie kolumny poza czasem traktujemy jako serie danych
    value_cols = [c for c in df.columns if c != time_col]

    # Konwersja na liczby (zamiana przecinków na kropki)
    series_data = {}
    for col in value_cols:
        s = df[col].astype(str).str.replace(',', '.').str.strip()
        series_data[col] = pd.to_numeric(s, errors='coerce')

    # Budowa końcowego DataFrame
    if time_col:
        time_series = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
        data = pd.DataFrame({'__time__': time_series, **series_data})
        data = data.dropna(subset=['__time__'])
        data = data.dropna(how='all', subset=value_cols)
        data = data.reset_index(drop=True)
        x = data['__time__']
        series_df = data[value_cols]
    else:
        data = pd.DataFrame(series_data)
        data = data.dropna(how='all')
        data = data.reset_index(drop=True)
        x = np.arange(len(data))
        series_df = data

    if series_df.empty:
        raise ValueError("Brak poprawnych danych liczbowych.")

    return x, series_df, value_cols


# ---------- klasa animatora ----------
class SingleAnimator:
    """Zarządza animacją pojedynczego wykresu z wieloma seriami."""

    def __init__(self, fig, ax, x, series_df, columns_to_show=None):
        self.ax = ax
        self.x = x
        self.series_df = series_df
        self.is_time = hasattr(x, 'iloc')

        # Które kolumny rysować (domyślnie wszystkie)
        if columns_to_show is None:
            self.columns = list(series_df.columns)
        else:
            self.columns = [c for c in columns_to_show if c in series_df.columns]

        if not self.columns:
            raise ValueError("Brak kolumn do wyświetlenia.")

        # Kolory
        colors = plt.cm.tab10.colors

        # Tworzenie linii i punktów
        self.lines = []
        self.dots = []
        for idx, col in enumerate(self.columns):
            color = colors[idx % len(colors)]
            ln, = ax.plot([], [], lw=LINE_WIDTH, color=color, label=col, zorder=3)
            dt, = ax.plot([], [], 'o', color=color, markersize=MARKER_SIZE, zorder=4)
            self.lines.append(ln)
            self.dots.append(dt)

        # Etykiety i legenda
        ax.set_ylabel('Wartość')
        ax.set_xlabel('Czas' if self.is_time else 'Indeks')
        ax.set_title('Animacja danych')
        ax.legend(loc='upper left')
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')

        # Limity osi Y
        data_vals = series_df[self.columns].values
        ymin, ymax = np.nanmin(data_vals), np.nanmax(data_vals)
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        margin = MARGIN_FACTOR * (ymax - ymin)
        ax.set_ylim(ymin - margin, ymax + margin)

        # Zakres osi X
        if self.is_time:
            ax.set_xlim(self.x.iloc[0], self.x.iloc[-1])
            fig.autofmt_xdate()
        else:
            ax.set_xlim(0, len(series_df) - 1)

        # Wypełnienia (dla pierwszej serii jako domyślnej)
        self.fill_pos = None
        self.fill_neg = None

    def init_anim(self):
        for ln in self.lines:
            ln.set_data([], [])
        for dt in self.dots:
            dt.set_data([], [])
        return self.lines + self.dots

    def update_anim(self, i):
        # Dane do i-tej klatki
        if self.is_time:
            xi = self.x.iloc[:i+1]
            x_point = self.x.iloc[i]
        else:
            xi = np.arange(i+1)
            x_point = i

        # Aktualizacja linii i punktów
        for idx, col in enumerate(self.columns):
            yi = self.series_df[col].iloc[:i+1].to_numpy(dtype=float)
            self.lines[idx].set_data(xi, yi)
            y_point = self.series_df[col].iloc[i]
            self.dots[idx].set_data([x_point], [y_point])

        # Usuwanie starych wypełnień
        for fill in (self.fill_pos, self.fill_neg):
            if fill is not None:
                fill.remove()
                fill = None

        # Nowe wypełnienia względem zera (dla pierwszej serii)
        first_col = self.columns[0]
        yi_first = self.series_df[first_col].iloc[:i+1].to_numpy(dtype=float)
        self.fill_pos = self.ax.fill_between(
            xi, yi_first, 0,
            where=(yi_first >= 0), interpolate=True,
            color='green', alpha=FILL_ALPHA, zorder=1
        )
        self.fill_neg = self.ax.fill_between(
            xi, yi_first, 0,
            where=(yi_first <= 0), interpolate=True,
            color='red', alpha=FILL_ALPHA, zorder=1
        )

        return self.lines + self.dots


# ---------- program główny ----------
def main():
    parser = argparse.ArgumentParser(description='Animacja serii danych z pliku Excel.')
    parser.add_argument('file', nargs='?', help='Plik Excel')
    parser.add_argument('--columns', '-c', nargs='+',
                        help='Kolumny do wyświetlenia (domyślnie: wszystkie numeryczne)')
    args = parser.parse_args()

    # Wybór pliku
    if args.file:
        fname = args.file
    else:
        for candidate in ['dane_pv.xlsx', 'dane.xlsx']:
            if os.path.exists(candidate):
                fname = candidate
                break
        else:
            raise FileNotFoundError("Nie znaleziono pliku. Podaj nazwę jako argument.")

    # Wczytanie danych
    x, series_df, available_cols = load_data(fname)
    logger.info("Znalezione serie: %s", available_cols)

    # Wybór kolumn do wyświetlenia
    if args.columns:
        columns_to_show = args.columns
    else:
        columns_to_show = available_cols  # domyślnie wszystkie

    logger.info("Wyświetlane kolumny: %s", columns_to_show)

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(12, 6))
    animator = SingleAnimator(fig, ax, x, series_df, columns_to_show)

    # Animacja
    ani = FuncAnimation(
        fig, animator.update_anim,
        frames=len(series_df),
        init_func=animator.init_anim,
        blit=False,
        interval=ANIMATION_INTERVAL
    )

    logger.info("Animacja gotowa – zamknij okno, aby zakończyć.")
    plt.show()


if __name__ == '__main__':
    main()
