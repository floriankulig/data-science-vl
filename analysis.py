import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import calendar

# TODO: Default-Wert für year_range ändern
DEFAULT_YEAR_RANGE = [2008, 2024]
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


def aggregate_hourly(df):
    """Optimierte Version mit numpy"""
    # Gruppiere zuerst nach den Schlüsseln
    grouped = df.groupby(["datum", "zaehlstelle", "hour"])

    # Berechne Summen mit numpy
    sums = grouped.agg({"richtung_1": "sum", "richtung_2": "sum", "gesamt": "sum"})

    # Berechne Kommentar-Aggregation separat
    comments = grouped["kommentar"].agg(
        lambda x: "; ".join(set(x.dropna())) if x.notna().any() else "NA"
    )

    # Finde Gruppen mit Kommentaren und 0-Summen
    has_comments = grouped["kommentar"].apply(lambda x: x.notna().any())
    zero_sums = (sums == 0).all(axis=1)

    # Setze NA mit numpy
    mask = has_comments & zero_sums
    sums[mask] = pd.NA

    result = pd.concat([sums, comments], axis=1)
    return result.reset_index()


def load_hourly_data(year_range=DEFAULT_YEAR_RANGE):
    """Lädt die 15-Minuten-Daten für mehrere Jahre"""
    dfs = []
    years = range(year_range[0], year_range[1])

    def convert_to_hourly(df: pd.DataFrame):
        """Konvertiert 15-Minuten-Daten in stündliche Daten"""
        df["time_start"] = pd.to_datetime(df["datum"] + " " + df["uhrzeit_start"])
        df["hour"] = df["time_start"].dt.hour
        df["gesamt"] = df["richtung_1"] + df["richtung_2"]
        hourly_data = aggregate_hourly(df)
        hourly_data["time_start"] = pd.to_datetime(
            hourly_data["datum"] + " " + hourly_data["hour"].astype(str) + ":00"
        )
        hourly_data = hourly_data.drop(["datum", "hour"], axis=1)
        return hourly_data

    for year in years:
        file_path = f"data_raw/rad_{year}_15min_06_06_23_r.csv"
        df = pd.read_csv(file_path, dtype=dtypes)
        print(f"\tLade Daten für {year}...")

        df_hourly_aggregated = convert_to_hourly(df)
        dfs.append(df_hourly_aggregated)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("time_start", inplace=True)
    # df.to_csv("data_raw_quarterly.csv", index=False)

    return df


# Anhand von Kommentar richtung1 und richtung2 bestimmen
def interpolate_plausible_values(df: pd.DataFrame):
    """Bereinigt und interpoliert Werte basierend auf Kommentaren"""
    # Berechne Mediane pro Zählstelle
    medians = (
        df.groupby("zaehlstelle")
        .agg({"richtung_1": "median", "richtung_2": "median"})
        .round()
        .astype(int)
    )
    for i in df.index:
        if df.at[i, "kommentar"] in {
            "Baustelle",
            "Eis/Schnee",
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

    # Kommentare vereinheitlichen
    df["kommentar"] = df["kommentar"].replace(
        "Radweg vereist / nach Schneefall nicht geräumt / keine Messung möglich",
        "Eis/Schnee",
    )

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

    # df.to_csv("data_cleaned.csv", index=False)

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
    interesting_columns = [
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
    numeric_stats = df[interesting_columns].describe().round(2)
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
    weather_correlations = df[interesting_columns].corr()["gesamt"].round(2)
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


# Visuelle Analysen für 1. Fragestellung: Ausfallzeiten
def analyze_outages(daily_data):
    """Analysiert die Ausfallzeiten mit einer Heatmap und Balkendiagramm"""

    # Zeitliche Entwicklung der Ausfälle
    daily_data["year"] = daily_data["datum"].dt.year
    daily_data["month"] = daily_data["datum"].dt.month

    # Filtere "Zählstelle noch nicht in Betrieb" aus der Analyse aus
    filtered_data = daily_data[
        daily_data["kommentar"] != "Zählstelle noch nicht in Betrieb"
    ]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 0.5])

    # 1. Heatmap der Ausfälle
    all_years = sorted(daily_data["year"].unique())
    all_months = range(1, 13)

    pivot_data = filtered_data[
        filtered_data["kommentar"] != "Kein Kommentar"
    ].pivot_table(
        index="year", columns="month", values="kommentar", aggfunc="count", fill_value=0
    )
    # Füge fehlende Jahre Monate hinzu (, die keine Ausfälle hatten)
    complete_index = pd.Index(all_years, name="year")
    complete_columns = pd.Index(all_months, name="month")
    pivot_data = pivot_data.reindex(
        index=complete_index, columns=complete_columns, fill_value=0
    )

    sns.heatmap(
        pivot_data,
        ax=ax1,
        cmap="Reds",
        fmt="d",
        annot=True,
        cbar_kws={"label": "Anzahl Ausfälle"},
        xticklabels=[calendar.month_abbr[i] for i in range(1, 13)],
    )

    for year in all_years:
        for month in all_months:
            mask = (
                (filtered_data["year"] == year)
                & (filtered_data["month"] == month)
                & (filtered_data["kommentar"] != "Kein Kommentar")
            )
            reasons = filtered_data[mask]["kommentar"].value_counts()

            # Calculate position in the heatmap
            i = all_years.index(year)
            j = month - 1

            if not reasons.empty:
                # Create compact reason string
                reason_str = "\n".join(
                    f"{reason[:10]}: {count}" for reason, count in reasons.items()
                )

                # Add text with small font
                ax1.text(
                    j + 0.5,
                    i + 0.8,
                    reason_str,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "black"
                        if pivot_data.iloc[i, j] < pivot_data.max().max() / 2
                        else "white"
                    ),
                )

    ax1.set_title("Heatmap der Ausfalltage pro Monat und Jahr", pad=20)
    ax1.set_xlabel("Monat")
    ax1.set_ylabel("Jahr")

    # 2. Balkendiagramm der Ausfalltypen (ohne "Zählstelle noch nicht in Betrieb")
    outage_types = (
        filtered_data[filtered_data["kommentar"] != "Kein Kommentar"]
        .groupby("kommentar")
        .size()
    )

    # Plot horizontal bar chart
    outage_types.plot(kind="barh", ax=ax2, color="steelblue")

    ax2.set_title("Verteilung der Ausfalltypen")
    ax2.set_xlabel("Anzahl")
    ax2.set_ylabel("Ausfallgrund")

    # Add value labels to bars
    for i, v in enumerate(outage_types):
        ax2.text(v, i, f" {v}", va="center")

    plt.tight_layout()
    plt.savefig(PLOT_PATH + "outage_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    return pivot_data, outage_types


# Visuelle Analysen für 2. Fragestellung: Nutzungstrends
def analyze_usage_trends(daily_data):
    """Analysiert die Nutzungstrends mit verschiedenen Zeitfenstern"""
    # Jahrestrend
    yearly_usage = (
        daily_data.groupby(daily_data["datum"].dt.year)["gesamt"]
        .agg(["sum", "mean", "std"])
        .round(2)
    )

    # Calculate CAGR
    first_year = yearly_usage.index[0]
    last_year = yearly_usage.index[-1]
    first_value = yearly_usage.loc[first_year, "sum"]
    last_value = yearly_usage.loc[last_year, "sum"]
    # Passe den Wert für 2008 an (nur 6 Monate Daten)
    first_value = first_value * 2 if first_year == 2008 else first_value
    years = last_year - first_year
    cagr = pow(last_value / first_value, 1 / years) - 1

    # Visualisierung Jahrestrend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    yearly_usage["sum"].plot(
        kind="bar",
        ax=ax1,
    )
    ax1.set_title("Jährliche Gesamtnutzung")
    ax1.set_xlabel("Jahr")
    ax1.set_ylabel("Fahrten (jährlich)")
    ax1.yaxis.set_major_formatter(FuncFormatter(format_number))
    ax1.grid(axis="y", linestyle="-", alpha=0.3)
    # CAGR als Text in der Ecke
    legend_text = f"CAGR: {cagr:.1%}"
    ax1.text(
        0.02,
        0.95,
        legend_text,
        transform=ax1.transAxes,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="lightgrey",
            pad=0.5,
        ),
        verticalalignment="top",
    )
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


# Visuelle Analysen für 3. Fragestellung: Wettereinfluss
def analyze_usage_patterns(quarterly_data):
    """
    Analysiert die Nutzungsmuster über Tageszeit und Jahreszeiten.

    Parameters:
    quarterly_data (pd.DataFrame): DataFrame mit 15-Minuten-Daten
    """

    quarterly_data["gesamt"] = pd.to_numeric(quarterly_data["gesamt"], errors="coerce")

    # Extrahiere Zeitinformationen
    quarterly_data["hour"] = quarterly_data["time_start"].dt.hour
    quarterly_data["month"] = quarterly_data["time_start"].dt.month

    # Erstelle Figure mit Subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    # 1. Tagesverlauf (gemittelt über alle Jahre)
    ax1 = fig.add_subplot(gs[0, :])
    hourly_avg = quarterly_data.groupby("hour")["gesamt"].mean()
    hourly_std = quarterly_data.groupby("hour")["gesamt"].std()

    ax1.plot(
        hourly_avg.index, hourly_avg.values, "b-", linewidth=2, label="Durchschnitt"
    )
    ax1.fill_between(
        hourly_avg.index,
        hourly_avg - hourly_std,
        hourly_avg + hourly_std,
        alpha=0.2,
        color="blue",
        label="Standardabweichung",
    )

    ax1.set_title("Durchschnittlicher Tagesverlauf der Fahrradnutzung", pad=20)
    ax1.set_xlabel("Stunde")
    ax1.set_ylabel("Durchschnittliche Anzahl Fahrräder")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    # 2. Heatmap: Stunde x Monat
    ax2 = fig.add_subplot(gs[1, 0])

    # Erstelle Pivot-Tabelle für Heatmap
    pivot_data = quarterly_data.pivot_table(
        values="gesamt", index="month", columns="hour", aggfunc="mean"
    ).astype(float)

    # Colormap (weiß zu dunkelblau)
    colors = ["#ffffff", "#d4e6f1", "#7fb3d5", "#2980b9", "#1a5276"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Plotte Heatmap
    sns.heatmap(
        pivot_data,
        cmap=cmap,
        ax=ax2,
        cbar_kws={"label": "Durchschnittliche Anzahl Fahrräder"},
        yticklabels=[calendar.month_abbr[i] for i in range(1, 13)],
        xticklabels=[f"{h:02d}:00" for h in range(24)],  # Stunden auf X-Achse
        robust=True,
    )

    ax2.set_title("Fahrradnutzung nach Tageszeit und Monat", pad=20)
    ax2.set_xlabel("Stunde")
    ax2.set_ylabel("Monat")
    # Labels rotieren
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    # Layout optimieren
    plt.tight_layout()
    plt.savefig(PLOT_PATH + "usage_patterns.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Daten laden
    print("Lade Tagesdaten...")
    daily_data = load_all_years()
    print("Lade Stundendaten...")
    hourly_data = load_hourly_data()

    # Deskriptive Statistik erstellen
    create_descriptive_statistics(daily_data)

    # Daten bereinigen
    print("Bereinige Daten...")
    daily_data = clean_data(daily_data)
    hourly_data = clean_data(hourly_data)

    # Analysen durchführen
    print("\nFühre Analysen durch...")
    outage_analysis = analyze_outages(daily_data)
    usage_trends = analyze_usage_trends(daily_data)
    # weather_impact = analyze_weather_impact(daily_data)
    analyze_usage_patterns(hourly_data)

    print("\nAnalysen abgeschlossen. Visualisierungen wurden gespeichert.")


if __name__ == "__main__":
    main()
