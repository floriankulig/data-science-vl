# Data Science Vorlesung

## Einrichtung

1. Erstelle eine virtuelle Umgebung:

   ```bash
   python -m venv venv   # Windows/Mac
   ```

2. Aktiviere die virtuelle Umgebung:

   ```bash
   source venv/bin/activate  # Linux/Mac
   # oder
   .\venv\Scripts\activate   # Windows
   ```

3. Installiere die Anforderungen:
   ```bash
   pip install -r requirements.txt
   ```

## Benutzung

1. Aufgabe a) & c):

```bash
   python analysis.py
```

- Daten werden geladen und bereinigt
- **Deskriptive Statstik** wird in der Konsole ausgegeben
- **Visualisierungen zu den Fragestellungen** werden im Ordner _plots_ gespeichert

2. Aufgabe d):

3. Aufgabe e):

```bash
   python app_dash.py
```

- Dash-App unter [http://127.0.0.1:8050](http://127.0.0.1:8050) aufrufen

## Fragestellungen

1.⁠ ⁠Wie hat sich die Ausfallzeit an den Radwegen über die Jahre entwickelt?
2.⁠ ⁠Nimmt die Nutzung der Radwege zu?
3.⁠ ⁠Wie verteilt sich das tägliche Fahrradaufkommen über den Tag / die Saisonalitäten?
