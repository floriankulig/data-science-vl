import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


# Daten einlesen
def load_and_clean_data():
    # Daten einlesen
    df = pd.read_csv("data_raw/rad_2022_tage_19_06_23_r.csv")

    # Datum in datetime umwandeln
    df["datum"] = pd.to_datetime(df["datum"])

    # Fehlende Werte identifizieren
    print("\nFehlende Werte pro Spalte:")
    print(df.isnull().sum())

    # Grundlegende Statistiken
    print("\nGrundlegende statistische Kennzahlen:")
    print(df.describe())

    # Verteilung der kategorischen Variablen
    print("\nVerteilung der Zählstellen:")
    print(df["zaehlstelle"].value_counts())

    # Korrelationsmatrix für numerische Variablen
    numeric_cols = [
        "richtung_1",
        "richtung_2",
        "gesamt",
        "min.temp",
        "max.temp",
        "niederschlag",
        "bewoelkung",
        "sonnenstunden",
    ]
    correlation_matrix = df[numeric_cols].corr()

    # Visualisierung der Korrelationsmatrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Korrelationsmatrix der numerischen Variablen")
    plt.show()

    # Zeitliche Analyse
    df["monat"] = df["datum"].dt.month
    df["jahr"] = df["datum"].dt.year

    # Monatliche Durchschnittswerte
    monthly_avg = df.groupby("monat")["gesamt"].mean()
    plt.figure(figsize=(10, 6))
    monthly_avg.plot(kind="bar")
    plt.title("Durchschnittliche Fahrradfahrer pro Monat")
    plt.xlabel("Monat")
    plt.ylabel("Durchschnittliche Anzahl")
    plt.show()

    return df


# Funktion ausführen
df = load_and_clean_data()
