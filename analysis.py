import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import calendar

# TODO: Default-Wert für year_range ändern
DEFAULT_YEAR_RANGE = [2008, 2016]
PLOT_PATH = "plots/"

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
    # df.to_csv("data_raw.csv", index=False)

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
    df = interpolate_plausible_values(df)

    # JETZT: Fehlende Werte rauswerfen
    # df = df.dropna()

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


def format_number(x, p):
    """Formatiert Zahlen in lesbare Strings (K = Tausend, M = Million)"""
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    elif x >= 1e3:
        return f"{x/1e3:.0f}K"
    else:
        return f"{x:.0f}"


def analyze_outages(daily_data):
    """Analysiert die Ausfallzeiten mit detaillierten Visualisierungen"""
    # plt.style.use("whitegrid")

    # Ausfälle nach Typ gruppieren
    outage_types = (
        daily_data[daily_data["kommentar"] != "Kein Kommentar"]
        .groupby("kommentar")
        .size()
    )

    # Visualisierung der Ausfalltypen
    plt.figure(figsize=(15, 6))
    outage_types.plot(kind="bar")
    plt.title("Verteilung der Ausfalltypen")
    plt.xlabel("Ausfallgrund")
    plt.ylabel("Anzahl")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_PATH + "outage_types.png")
    plt.close()

    # Zeitliche Entwicklung der Ausfälle
    daily_data["year"] = daily_data["datum"].dt.year
    daily_data["month"] = daily_data["datum"].dt.month

    monthly_outages = (
        daily_data[daily_data["kommentar"] != "Kein Kommentar"]
        .groupby(["year", "month"])
        .size()
        .reset_index(name="outages")
    )

    plt.figure(figsize=(15, 6))
    sns.lineplot(data=monthly_outages, x="month", y="outages", hue="year")
    plt.title("Entwicklung der Ausfälle über die Zeit")
    plt.xlabel("Monat")
    plt.ylabel("Anzahl Ausfälle")
    plt.legend(title="Jahr", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOT_PATH + "outages_timeline.png")
    plt.close()

    return monthly_outages


def analyze_usage_trends(daily_data):
    """Analysiert die Nutzungstrends mit verschiedenen Zeitfenstern"""
    # Jahrestrend
    yearly_usage = (
        daily_data.groupby(daily_data["datum"].dt.year)["gesamt"]
        .agg(["sum", "mean", "std"])
        .round(2)
    )

    # Visualisierung Jahrestrend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    yearly_usage["sum"].plot(kind="bar", ax=ax1)
    ax1.set_title("Jährliche Gesamtnutzung")
    ax1.set_xlabel("Jahr")
    ax1.set_ylabel("Fahrten (jährlich)")
    ax1.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax1.grid(axis="y", linestyle="-", alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    line = ax2.plot(
        yearly_usage.index,
        yearly_usage["mean"],
        marker="o",
        color="steelblue",
        label="Durchschnittliche Nutzung",
    )
    std_area = ax2.fill_between(
        yearly_usage.index,
        yearly_usage["mean"] - yearly_usage["std"],
        yearly_usage["mean"] + yearly_usage["std"],
        alpha=0.2,
        color="steelblue",
        label="Standardabweichung",
    )
    ax2.set_title("Durchschnittliche tägliche Nutzung pro Jahr")
    ax2.set_xlabel("Jahr")
    ax2.set_ylabel("Durchschnittliche Nutzung")
    ax2.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax2.grid(axis="y", linestyle="-", alpha=0.7)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(PLOT_PATH + "yearly_usage_trends.png", dpi=300, bbox_inches="tight")
    plt.close()

    return yearly_usage


def analyze_weather_impact(daily_data):
    """Erweiterte Wettereinfluss-Analyse mit Schwellenwerten"""
    # Korrelationsmatrix
    weather_cols = ["max.temp", "niederschlag", "sonnenstunden", "bewoelkung", "gesamt"]
    corr_matrix = daily_data[weather_cols].corr()

    # Visualisierung der Korrelationsmatrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Korrelation zwischen Wetter und Fahrradnutzung")
    plt.tight_layout()
    plt.savefig(PLOT_PATH + "weather_correlation.png")

    # Wetterabhängige Nutzungsanalyse
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Temperatur
    sns.regplot(data=daily_data, x="max.temp", y="gesamt", ax=axes[0, 0])
    axes[0, 0].set_title("Temperatureinfluss auf Nutzung")

    # Niederschlag mit Schwellenwert
    sns.regplot(data=daily_data, x="niederschlag", y="gesamt", ax=axes[0, 1])
    axes[0, 1].set_title("Niederschlagseinfluss auf Nutzung")

    # Sonnenstunden
    sns.regplot(data=daily_data, x="sonnenstunden", y="gesamt", ax=axes[1, 0])
    axes[1, 0].set_title("Einfluss der Sonnenstunden")

    # Bewölkung
    sns.regplot(data=daily_data, x="bewoelkung", y="gesamt", ax=axes[1, 1])
    axes[1, 1].set_title("Bewölkungseinfluss auf Nutzung")

    plt.tight_layout()
    plt.savefig(PLOT_PATH + "weather_detailed_analysis.png")
    plt.close()

    return corr_matrix


def main():
    # Daten laden
    print("Lade Tagesdaten...")
    daily_data = load_all_years()
    print("Lade Viertelstundendaten...")
    # quarterly_data = load_quarterly_data()

    # Deskriptive Statistik erstellen
    create_descriptive_statistics(daily_data)

    # Daten bereinigen
    print("Bereinige Daten...")
    daily_data = clean_data(daily_data)
    # quarterly_data = clean_data(quarterly_data)

    # Analysen durchführen
    print("\nFühre Analysen durch...")
    outage_analysis = analyze_outages(daily_data)
    usage_trends = analyze_usage_trends(daily_data)
    weather_impact = analyze_weather_impact(daily_data)

    print("\nAnalysen abgeschlossen. Visualisierungen wurden gespeichert.")


if __name__ == "__main__":
    main()
