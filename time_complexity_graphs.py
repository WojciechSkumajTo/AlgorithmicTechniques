import pandas as pd
import matplotlib.pyplot as plt

def plot_algorithm_complexity(csv_file):
    # Wczytanie danych
    df = pd.read_csv(csv_file)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))

    # Rysowanie wykresów dla każdego algorytmu
    plt.plot(df['Number of Nodes'], df["Yen's Execution Time (s)"], label="Yen's Algorithm", marker='o')
    plt.plot(df['Number of Nodes'], df['GRASP Execution Time (s)'], label='GRASP Algorithm', marker='x')

    # Dodawanie etykiet i tytułu
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Algorithm Time Complexity Analysis')
    plt.legend()

    # Wyświetlanie wykresu
    plt.grid(True)
    plt.show()

# Ścieżka do pliku CSV z wynikami
csv_file = 'benchmark_results.csv'

# Generowanie wykresu
plot_algorithm_complexity(csv_file)
