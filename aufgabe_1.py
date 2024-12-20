import pandas as pd
import numpy as np
from datetime import datetime


def load_all_years():
    dfs = []
    for year in range(2008, 2024):
        file_path = f"data_raw/rad_{year}_tage_19_06_23_r.csv"
        df = pd.read_csv(file_path)
        # Datum in datetime umwandeln
        df["datum"] = pd.to_datetime(df["datum"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("datum", inplace=True)
    df.to_csv("data_raw.csv", index=False)

    return df


def interpolate_plausible_values(df):
    pass


def clean_data(df):
    # Datum in datetime umwandeln
    df["datum"] = pd.to_datetime(df["datum"])

    # NA bei Kommentaren durch Leerstring ersetzen
    df["kommentar"] = df["kommentar"].fillna("Kein Kommentar")

    # Negative Werte in relevanten Spalten durch NaN ersetzen
    numeric_columns = [
        "richtung_1",
        "richtung_2",
        "gesamt",
        "niederschlag",
        "bewoelkung",
        "sonnenstunden",
    ]
    for col in numeric_columns:
        # Bereinigung: Ersetze negative Werte durch NaN in relevanten Spalten
        df.loc[df[col] < 0, col] = np.nan

    # SPÄTER UMÄNDERN: Fehlende Werte in Wetterdaten durch Mittelwert ersetzen
    # JETZT: Fehlende Werte rauswerfen
    df = df.dropna()

    df.to_csv("data_cleaned.csv", index=False)

    return df


def create_descriptive_statistics(df):
    # 0. Fehlende Werte berechnen
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_stats = pd.DataFrame(
        {"Fehlende Werte": missing_values, "Prozent": missing_percentage.round(2)}
    )

    df = clean_data(df)

    # 1. Grundlegende Informationen
    print("=== GRUNDLEGENDE INFORMATIONEN ===")
    print(f"Zeitraum der Daten: {df['datum'].min()} bis {df['datum'].max()}")
    print(f"Anzahl der Datensätze: {len(df)}")
    print(f"Anzahl der Zählstellen: {df['zaehlstelle'].nunique()}")
    print("Zählstellen:", ", ".join(df["zaehlstelle"].unique()))

    # 2. Fehlende Werte ausgeben
    print("\n=== FEHLENDE WERTE ===")
    print(missing_stats)

    # 3. Statistische Kennzahlen für numerische Spalten
    print("\n=== STATISTISCHE KENNZAHLEN ===")
    numeric_stats = df[
        [
            # Richtungen sind für die Gesamtstatistik nicht relevant, da diese sehr stark von den jeweiligen Zählstellen abhängen
            # "richtung_1",
            # "richtung_2",
            "gesamt",
            "min.temp",
            "max.temp",
            "niederschlag",
            "bewoelkung",
            "sonnenstunden",
        ]
    ].describe()
    print("\nZählungen und Wetterdaten:")
    print(numeric_stats)

    # 4. Aggregierte Statistiken pro Zählstelle
    print("\n=== STATISTIKEN PRO ZÄHLSTELLE ===")
    zahlstellen_stats = (
        df.groupby("zaehlstelle")["gesamt"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(2)
    )
    print(zahlstellen_stats)

    # 6. Wetter-Korrelationen
    print("\n=== KORRELATIONEN GESAMTFAHRTEN MIT WETTERDATEN ===")
    weather_correlations = (
        df[
            [
                "gesamt",
                "min.temp",
                "max.temp",
                "niederschlag",
                "bewoelkung",
                "sonnenstunden",
            ]
        ]
        .corr()["gesamt"]
        .round(2)
    )
    print(weather_correlations)


def main():
    # Daten laden und bereinigen
    df = load_all_years()

    # Deskriptive Statistik erstellen
    create_descriptive_statistics(df)


if __name__ == "__main__":
    main()
