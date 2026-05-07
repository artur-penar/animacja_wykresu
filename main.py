import os
import itertools
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- stałe konfiguracyjne ----------
FILL_ALPHA_ZERO = 0.25
FILL_ALPHA_PRED = 0.15
LINE_WIDTH = 2
MARKER_SIZE = 6
ANIMATION_INTERVAL = 80
MARGIN_FACTOR = 0.05

# ---------- logowanie ----------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------- narzędzia ----------
def find_column(df, candidates, fallback_heuristic=None):
    """Zwraca pierwszą kolumnę z listy kandydatów, która istnieje w df."""
    for col in candidates:
        if col in df.columns:
            return col
    if fallback_heuristic:
        try:
            return fallback_heuristic(df)
        except Exception:
            pass
    return None


def set_ax_ylim(ax, series_df, cols):
    """Ustawia limity osi Y na podstawie kolumn `cols` z ramki `series_df`."""
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


# ---------- wczytywanie danych ----------
def load_data(file_path):
    """Wczytuje plik Excel, identyfikuje kolumnę czasu i kolumny numeryczne,
    zwraca (x, series_df), gdzie x to indeksy lub daty, a series_df zawiera
    wyłącznie kolumny z danymi liczbowymi (bez NaN w czasie)."""
    logger.info("Wczytywanie pliku: %s", file_path)
    df = pd.read_excel(file_path, sheet_name=0)

    # Szukamy kolumny czasu
    time_candidates = ['Time', 'time', 'Date', 'date', 'Data', 'data']
    time_col = find_column(df, time_candidates)
    if time_col is None:
        # heurystyka: czy pierwsza kolumna to data?
        first_col = df.columns[0]
        sample = df[first_col].dropna().astype(str).iloc[0]
        try:
            pd.to_datetime(sample, dayfirst=True)
            time_col = first_col
            logger.info("Kolumna czasu wykryta heurystycznie: '%s'", time_col)
        except Exception:
            pass

    # Szukamy kolumny/kolumn wartości (wszystkie poza czasem, które są numeryczne)
    value_cols = [c for c in df.columns if c != time_col]
    # Dla bezpieczeństwa konwertujemy wszystkie kolumny wartości na liczby,
    # zamieniając przecinki na kropki
    series_data = {}
    for col in value_cols:
        s = df[col].astype(str).str.replace(',', '.').str.strip()
        numeric = pd.to_numeric(s, errors='coerce')
        series_data[col] = numeric

    # Budujemy DataFrame z czasem i wartościami
    if time_col:
        time_series = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
        data = pd.DataFrame({'__time__': time_series, **series_data})
        # Usuwamy wiersze z brakującym czasem
        data = data.dropna(subset=['__time__'])
        # Usuwamy wiersze, w których wszystkie wartości są NaN
        data = data.dropna(how='all', subset=value_cols)
        data = data.reset_index(drop=True)
        x = data['__time__']
        series_df = data[value_cols]
    else:
        # Brak kolumny czasu – używamy indeksu
        data = pd.DataFrame(series_data)
        data = data.dropna(how='all')
        data = data.reset_index(drop=True)
        x = np.arange(len(data))
        series_df = data

    if series_df.empty:
        raise ValueError("Brak poprawnych danych liczbowych.")

    return x, series_df, value_cols


# ---------- konfiguracja wykresu ----------
def setup_axes():
    """Tworzy figurę z dwoma wykresami góra/dół."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    ax_top.axhline(0, color='gray', linewidth=1, linestyle='--')
    ax_bot.axhline(0, color='gray', linewidth=1, linestyle='--')
    return fig, ax_top, ax_bot


def choose_columns(series_df):
    """Wybiera domyślne kolumny dla górnego i dolnego wykresu."""
    top_candidates = [['Value [W]', 'Pred [W]'], ['Value', 'Pred']]
    bot_candidates = [['Moc [W]', 'Moc Pred [W]'], ['Moc', 'Moc Pred']]

    # Górny wykres
    top_cols = None
    for cand in top_candidates:
        top_cols = [c for c in cand if c in series_df.columns]
        if top_cols:
            break
    if not top_cols:  # fallback: dwie pierwsze kolumny, jeśli są
        top_cols = list(series_df.columns[:2]) if len(series_df.columns) >= 2 else list(series_df.columns[:1])

    # Dolny wykres
    bottom_cols = None
    for cand in bot_candidates:
        bottom_cols = [c for c in cand if c in series_df.columns]
        if bottom_cols:
            break
    if not bottom_cols:
        # fallback: druga kolumna (jeśli istnieje) lub pusta
        bottom_cols = [series_df.columns[1]] if len(series_df.columns) >= 2 else []

    return top_cols, bottom_cols


def detect_pred_column(series_df, top_cols, bottom_cols):
    """Wykrywa kolumny predykcji osobno dla górnego i dolnego wykresu."""
    # Lista wszystkich kolumn zawierających 'pred' (case-insensitive)
    pred_like = [c for c in series_df.columns if 'pred' in c.lower()]

    pred_top = None
    if 'Pred [W]' in series_df.columns:
        pred_top = 'Pred [W]'
    else:
        # wybierz pierwszą kolumnę pred, która nie zawiera 'moc'
        for c in pred_like:
            if 'moc' not in c.lower():
                pred_top = c
                break
        if pred_top is None and pred_like:
            pred_top = pred_like[0]

    pred_bot = None
    if 'Moc Pred [W]' in series_df.columns:
        pred_bot = 'Moc Pred [W]'
    else:
        # kolumna pred zawierająca 'moc'
        for c in pred_like:
            if 'moc' in c.lower():
                pred_bot = c
                break
        if pred_bot is None and len(pred_like) > 1:
            # weź inną niż pred_top, jeśli istnieje
            for c in pred_like:
                if c != pred_top:
                    pred_bot = c
                    break
        if pred_bot is None and pred_like:
            pred_bot = pred_like[0]

    return pred_top, pred_bot


# ---------- klasa zarządzająca animacją ----------
class DualAnimator:
    def __init__(self, fig, ax_top, ax_bot, x, series_df, top_cols, bottom_cols):
        self.fig = fig
        self.ax_top = ax_top
        self.ax_bot = ax_bot
        self.x = x
        self.series_df = series_df
        self.top_cols = top_cols
        self.bottom_cols = bottom_cols
        self.is_time = hasattr(x, 'iloc')  # True jeśli x to Series z datami

        # Wykrywanie kolumn predykcji
        self.pred_top, self.pred_bot = detect_pred_column(series_df, top_cols, bottom_cols)
        # Dodajemy je do wykresów, jeśli jeszcze nie ma
        for pred_col, ax, col_list, attr_name in [
            (self.pred_top, ax_top, self.top_cols, '_top'),
            (self.pred_bot, ax_bot, self.bottom_cols, '_bot')
        ]:
            if pred_col and pred_col not in col_list:
                col_list.append(pred_col)

        # Kolory dla linii
        self.colors = itertools.cycle(plt.cm.tab10.colors)

        # Tworzenie artystów
        self.lines_top, self.dots_top = self._create_artists(ax_top, top_cols)
        self.lines_bot, self.dots_bot = self._create_artists(ax_bot, bottom_cols)

        # Etykiety i legendy
        ax_top.set_ylabel('Wartość [W]')
        ax_bot.set_ylabel('Wartość [W]')
        ax_bot.set_xlabel('Czas' if self.is_time else 'Indeks')
        ax_top.set_title('Górny wykres: ' + ', '.join(self.top_cols))
        if self.bottom_cols:
            ax_bot.set_title('Dolny wykres: ' + ', '.join(self.bottom_cols))
        ax_top.legend(loc='upper left')
        if self.bottom_cols:
            ax_bot.legend(loc='upper left')

        # Limity osi
        set_ax_ylim(ax_top, series_df, self.top_cols)
        set_ax_ylim(ax_bot, series_df, self.bottom_cols)

        # Zakres osi X
        if self.is_time:
            ax_top.set_xlim(self.x.iloc[0], self.x.iloc[-1])
            self.fig.autofmt_xdate()
        else:
            ax_top.set_xlim(0, len(series_df) - 1)

        # Kolumny użyte do wypełnień względem zera
        self.actual_top = self.top_cols[0] if self.top_cols else None
        self.actual_bot = self.bottom_cols[0] if self.bottom_cols else None

        # Przechowywanie wypełnień
        self.fills = {
            'top_zero': {'pos': None, 'neg': None},
            'top_pred': {'pos': None, 'neg': None},
            'bot_zero': {'pos': None, 'neg': None},
            'bot_pred': {'pos': None, 'neg': None},
        }

    def _create_artists(self, ax, cols):
        lines, dots = [], []
        for col in cols:
            color = next(self.colors)
            ln, = ax.plot([], [], lw=LINE_WIDTH, color=color, label=col, zorder=3)
            dt, = ax.plot([], [], 'o', color=color, markersize=MARKER_SIZE, zorder=4)
            lines.append(ln)
            dots.append(dt)
        return lines, dots

    def init_anim(self):
        all_artists = []
        for lst in (self.lines_top, self.lines_bot, self.dots_top, self.dots_bot):
            all_artists.extend(lst)
        for artist in all_artists:
            artist.set_data([], [])
        return all_artists

    def update_anim(self, i):
        # Przygotowanie danych do i-tej klatki
        if self.is_time:
            xi = self.x.iloc[:i+1]
            x_point = self.x.iloc[i]
        else:
            xi = np.arange(i+1)
            x_point = i

        # Aktualizacja linii i punktów
        self._update_series(self.ax_top, self.top_cols, self.lines_top, self.dots_top, xi, x_point, i)
        self._update_series(self.ax_bot, self.bottom_cols, self.lines_bot, self.dots_bot, xi, x_point, i)

        # Usuwanie starych wypełnień
        for grp in self.fills.values():
            for key in ('pos', 'neg'):
                if grp[key] is not None:
                    grp[key].remove()
                    grp[key] = None

        # Nowe wypełnienia – górny wykres
        if self.actual_top is not None:
            yi_actual = self.series_df[self.actual_top].iloc[:i+1].to_numpy(dtype=float)
            self.fills['top_zero']['pos'] = self.ax_top.fill_between(
                xi, yi_actual, 0, where=(yi_actual >= 0), interpolate=True,
                color='green', alpha=FILL_ALPHA_ZERO, zorder=1
            )
            self.fills['top_zero']['neg'] = self.ax_top.fill_between(
                xi, yi_actual, 0, where=(yi_actual <= 0), interpolate=True,
                color='red', alpha=FILL_ALPHA_ZERO, zorder=1
            )
            if self.pred_top is not None and self.pred_top in self.series_df.columns:
                yi_pred = self.series_df[self.pred_top].iloc[:i+1].to_numpy(dtype=float)
                self.fills['top_pred']['pos'] = self.ax_top.fill_between(
                    xi, yi_actual, yi_pred, where=(yi_actual >= yi_pred), interpolate=True,
                    color='green', alpha=FILL_ALPHA_PRED, zorder=1
                )
                self.fills['top_pred']['neg'] = self.ax_top.fill_between(
                    xi, yi_actual, yi_pred, where=(yi_actual <= yi_pred), interpolate=True,
                    color='red', alpha=FILL_ALPHA_PRED, zorder=1
                )

        # Dolny wykres
        if self.actual_bot is not None:
            yi_actual = self.series_df[self.actual_bot].iloc[:i+1].to_numpy(dtype=float)
            self.fills['bot_zero']['pos'] = self.ax_bot.fill_between(
                xi, yi_actual, 0, where=(yi_actual >= 0), interpolate=True,
                color='green', alpha=FILL_ALPHA_ZERO, zorder=1
            )
            self.fills['bot_zero']['neg'] = self.ax_bot.fill_between(
                xi, yi_actual, 0, where=(yi_actual <= 0), interpolate=True,
                color='red', alpha=FILL_ALPHA_ZERO, zorder=1
            )
            if self.pred_bot is not None and self.pred_bot in self.series_df.columns:
                yi_pred = self.series_df[self.pred_bot].iloc[:i+1].to_numpy(dtype=float)
                self.fills['bot_pred']['pos'] = self.ax_bot.fill_between(
                    xi, yi_actual, yi_pred, where=(yi_actual >= yi_pred), interpolate=True,
                    color='red', alpha=FILL_ALPHA_PRED, zorder=1
                )
                self.fills['bot_pred']['neg'] = self.ax_bot.fill_between(
                    xi, yi_actual, yi_pred, where=(yi_actual <= yi_pred), interpolate=True,
                    color='green', alpha=FILL_ALPHA_PRED, zorder=1
                )

        # Zwracamy wszystkich widocznych artystów, animacja je odświeży
        all_artists = []
        for lst in (self.lines_top, self.lines_bot, self.dots_top, self.dots_bot):
            all_artists.extend(lst)
        return all_artists

    def _update_series(self, ax, cols, lines, dots, xi, x_point, i):
        for idx, col in enumerate(cols):
            yi = self.series_df[col].iloc[:i+1].to_numpy(dtype=float)
            lines[idx].set_data(xi, yi)
            y_point = self.series_df[col].iloc[i]
            dots[idx].set_data([x_point], [y_point])


# ---------- program główny ----------
def main():
    parser = argparse.ArgumentParser(description='Animacja danych PV z dwoma wykresami.')
    parser.add_argument('file', nargs='?', help='Plik Excel (domyślnie: dane_pv.xlsx lub dane.xlsx)')
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
            raise FileNotFoundError(
                "Nie znaleziono pliku 'dane_pv.xlsx' ani 'dane.xlsx'. "
                "Podaj nazwę pliku jako argument."
            )

    # Wczytanie danych
    x, series_df, _ = load_data(fname)

    # Wybór kolumn na wykresy
    top_cols, bottom_cols = choose_columns(series_df)

    # Tworzenie wykresu i animatora
    fig, ax_top, ax_bot = setup_axes()
    animator = DualAnimator(fig, ax_top, ax_bot, x, series_df, top_cols, bottom_cols)

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
