import pandas as pd
import numpy as np
from datetime import datetime

# TODO: Default-Wert für year_range ändern
DEFAULT_YEAR_RANGE = [2008, 2024]

dtypes = {
    "datum": str,
    "zaehlstelle": str,
    "uhrzeit_start": str,
    "uhrzeit_ende": str,
    "richtung_1": "Int32",
    "richtung_2": "Int32",
    "gesamt": "Int32",
    "min.temp": float,
    "max.temp": float,
    "niederschlag": float,
    "bewoelkung": "Int32",
    "sonnenstunden": float,
    "kommentar": str,
}


def load_all_years(year_range=DEFAULT_YEAR_RANGE):
    """Lädt die Tagesdaten für mehrere Jahre"""
    dfs = []
    years = range(year_range[0], year_range[1])
    for year in years:
        file_path = f"data_raw/rad_{year}_tage_19_06_23_r.csv"
        df = pd.read_csv(file_path, dtype=dtypes)
        df["datum"] = pd.to_datetime(df["datum"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("datum", inplace=True)
    df.to_csv("data_raw.csv", index=False)

    return df


def load_quarterly_data(year_range=DEFAULT_YEAR_RANGE):
    """Lädt die 15-Minuten-Daten für mehrere Jahre"""
    dfs = []
    years = range(year_range[0], year_range[1])
    for year in years:
        file_path = f"data_raw/rad_{year}_15min_06_06_23_r.csv"
        df = pd.read_csv(file_path, dtype=dtypes)

        # Zeit-Spalten erstellen
        df["time_start"] = pd.to_datetime(df["datum"] + " " + df["uhrzeit_start"])
        df["time_end"] = pd.to_datetime(df["datum"] + " " + df["uhrzeit_ende"])
        df = df.drop(["datum", "uhrzeit_start", "uhrzeit_ende"], axis=1)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("time_start", inplace=True)
    # df.to_csv("data_raw_quarterly.csv", index=False)

    return df


# Anhand von Kommentar richtung1 und richtung2 bestimmen
def interpolate_plausible_values(df: pd.DataFrame):
    """Bereinigt und interpoliert Werte basierend auf Kommentaren"""
    # Berechne Mediane pro Zählstelle
    medians = df.groupby("zaehlstelle").agg(
        {"richtung_1": "median", "richtung_2": "median"}
    )
    print(medians)
    for i in df.index:
        if df.at[i, "kommentar"] in {
            "Baustelle",
            "Radweg vereist / nach Schneefall nicht geräumt / keine Messung möglich",
            "Zählstelle noch nicht in Betrieb",
        }:
            df.at[i, "richtung_1"] = df.at[i, "richtung_2"] = df.at[i, "gesamt"] = 0
        elif df.at[i, "kommentar"] in {
            "Ausfall",
            "Austausch Sensor",
            "Ausfall, baustellenbedingte Demontage  ",
            "Ausfall nach Beschädigung",
        }:
            median1 = medians.loc[df.at[i, "zaehlstelle"], "richtung_1"]
            median2 = medians.loc[df.at[i, "zaehlstelle"], "richtung_2"]
            df.at[i, "richtung_1"] = median1
            df.at[i, "richtung_2"] = median2
            df.at[i, "gesamt"] = median1 + median2
    return df


def clean_data(df):
    # Datum in datetime umwandeln
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"])

    # NA bei Kommentaren durch Leerstring ersetzen
    df["kommentar"] = df["kommentar"].fillna("Kein Kommentar")

    # Negative Werte in relevanten Spalten durch NaN ersetzen
    numeric_columns = [
        col
        for col in [
            "richtung_1",
            "richtung_2",
            "gesamt",
            "niederschlag",
            "bewoelkung",
            "sonnenstunden",
        ]
        if col in df.columns
    ]
    for col in numeric_columns:
        # Bereinigung: Ersetze negative Werte durch NaN in relevanten Spalten
        df.loc[df[col] < 0, col] = np.nan

    # SPÄTER UMÄNDERN: Fehlende Werte in ricchtungen und gesamt durch Mittelwert ersetzen
    # df = interpolate_plausible_values(df)

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
    numeric_stats = (
        df[
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
        ]
        .describe()
        .round(2)
    )
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
    weather_correlations_df = weather_correlations.to_frame(name="Korrelation")
    weather_correlations_df["Erklärung"] = weather_correlations_df["Korrelation"].apply(
        lambda x: (
            "positiver Zusammenhang"
            if x > 0
            else ("negativer Zusammenhang" if x < 0 else "kein Zusammenhang")
        )
    )

    print(weather_correlations_df)
    # print(weather_correlations)


def main():
    # Daten laden
    print("Lade Tagesdaten...")
    daily_data = load_all_years()
    print("Lade Viertelstundendaten...")
    quarterly_data = load_quarterly_data()

    # Deskriptive Statistik erstellen
    create_descriptive_statistics(daily_data)

    # Daten bereinigen
    print("Bereinige Daten...")
    daily_data = clean_data(daily_data)
    quarterly_data = clean_data(quarterly_data)  #


if __name__ == "__main__":
    main()
