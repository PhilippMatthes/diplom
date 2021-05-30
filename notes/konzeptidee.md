## Ideenerarbeitung zum Konzept und zur Evaluation

`Stand: 29. Mai 2021 (Projektbearbeitungsphase Verwandte Arbeiten)`

- Datenerfassung: prototypische App -> Experimentelle Bedingungen sind kontrollierbar
- GNSS-Daten werden nicht erfasst -> Um Energie zu sparen
- Smartwatch zum Input der Labels statt Smartphone -> Elimination des "Shakings" bei VM-Wechsel
- Festes Zeitfenster um VM-Wechsel -> Löschen oder Label als "VM-Wechsel" für bessere Labels
- Verschiedene Smartphones, Personen und Tragepositionen -> Variation in Input-Daten
- Planung von Fahrten (Fahrrad, Auto, Zu Fuß, Bus, Bahn, Zug)
- Gleichgewichtung der Daten -> Vermeidung von Bias

- Klassifikationskonzept
- Vorverarbeitung durch Segmentierung und Gleitfenstermethode
- Fensterlänge weniger als 2 Sekunden -> Echtzeit-Klassifikation ohne Warten möglich
- Abtastrate auf 50Hz (Shannon-Nyquist-Theorem) -> Lineare Interpolation
- Variante 1:
    - Akzelerometer-Signal - Betrag errechnen -> Gravitation kein Problem mehr
    - Direkte Konvertierung in Spektrogramm -> Max. Informationsgehalt aber möglicherweise Störsignale
    - Transfer Learning z.B. von MobileNet auf Spektrogrammen
    - Modelle können dann aber nicht mehr einfach ausgetauscht werden -> Limitiert auf CNN
- Variante 2:
    - Minimale Rauschunterdrückung durch Glättung, Hoch- und Tiefpass
    - Madgwick-AHRS -> Quaternion und Gravitationselimination
    - Feature-Set erzeugen
    - Klassifikation über RNN (Zeitlinie), FFN oder SVM oder RF oder HMM

- Postprocessing durch Glättung möglich, aber unerheblich für Evaluation des Tradeoffs -> Geschieht *nach* Klassifikation

- Evaluation des Konzeptes -> Ziel: Tradeoff analysieren!
- Lösung: Modelle und Optimierungen variieren
- Variation von Quantisierung, Anzahl Labels, Anzahl Hidden Layer, Pruning (Prozent entfernter Parameter)
- Gegenüberstellung der F1-Scores oder der Accuracy / Confusion matrices
