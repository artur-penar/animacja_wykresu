import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# 1) Wczytaj dane: oczekujemy pliku 'data.xlsx' z kolumną 'Cena'
# Jeśli wolisz CSV, zmień pd.read_excel -> pd.read_csv
df = pd.read_excel('dane.xlsx', sheet_name=0)  # plik w tym samym katalogu co skrypt

# Jeśli kolumna 'Cena' nie istnieje (np. brak nagłówka), sprawdź kolumny i wybierz pierwszą
print('Kolumny w pliku:', list(df.columns))
if 'Cena' in df.columns:
    series = df['Cena']
else:
    # jeśli DataFrame ma tylko jedną kolumnę, użyj jej
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        print("Używam pierwszej kolumny jako serii (brak nagłówka 'Cena').")
    else:
        # spróbuj znaleźć kolumnę numeryczną
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            series = df[numeric_cols[0]]
            print(f"Znalazłem kolumnę liczbową '{numeric_cols[0]}' i jej używam.")
        else:
            raise KeyError("Nie znaleziono kolumny 'Cena' ani żadnej kolumny numerycznej. Sprawdź plik 'dane.xlsx'.")

# Upewnij się, że wartości są liczbami: zamień przecinki dziesiętne na kropki, usuń spacje
series_str = series.astype(str).str.replace(',', '.').str.strip()
y = pd.to_numeric(series_str, errors='coerce')
if y.isna().all():
    raise ValueError("Wybrane dane zawierają tylko wartości nienumeryczne.")
# Usuń NaN (jeśli jakieś są)
if y.isna().any():
    print(f"Usuwam {y.isna().sum()} wierszy z brakującymi/nienumerycznymi wartościami.")
    y = y.dropna().reset_index(drop=True)
else:
    y = y.reset_index(drop=True)

x = np.arange(len(y))  # proste indeksy 0..n-1 (brak kolumny czasu na razie)

# 2) Przygotowanie rysunku
fig, ax = plt.subplots(figsize=(8, 4.5))
line, = ax.plot([], [], lw=2, color='tab:blue', label='Cena')
dot, = ax.plot([], [], 'o', color='tab:red')  # poruszająca się kropka
ax.set_xlabel('Indeks')
ax.set_ylabel('Cena')
ax.set_title('Animacja serii "Cena"')
ax.legend(loc='upper left')

# Ustaw limity osi Y z niewielkim marginesem
ymin, ymax = y.min(), y.max()
if ymin == ymax:
    ymin -= 1
    ymax += 1
margin = 0.05 * (ymax - ymin)
ax.set_xlim(0, len(x) - 1)
ax.set_ylim(ymin - margin, ymax + margin)

# 3) Funkcje animacji
def init():
    line.set_data([], [])
    dot.set_data([], [])
    return line, dot

def update(i):
    # rysujemy linię do punktu i oraz kropkę w punkcie i
    line.set_data(x[: i + 1], y[: i + 1])
    # set_data oczekuje sekwencji dla x i y, więc opakowujemy pojedynczy punkt w listę
    dot.set_data([x[i]], [y[i]])
    return line, dot

# 4) Stworzenie animacji i wyświetlenie (nie zapisujemy jeszcze do pliku)
frames = len(x)
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=80)
print('Animacja gotowa — otwieram okno z wykresem. Zamknij okno, aby zakończyć program.')
plt.show()
