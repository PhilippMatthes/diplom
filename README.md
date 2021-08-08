# Diplomarbeit

In der STADTRADELN-App werden GPS- und Sensordaten von Radfahrenden gespeichert und für die Weiterverarbeitung an einen Server übermittelt. Die aufgezeichneten Datensätze werden unter anderem durch das Forschungsprojekt Movebis genutzt, mit dem Ziel, die Planung der Radverkehrsinfrastruktur zu verbessern. Bei Analysen der Datensätze konnte festgestellt werden, dass diese auch Datenpakete mit für Radfahrer untypischen Parameterwerten beinhalten. Daher wurden im Movebis-Projekt Machine-Learning-Ansätze entwickelt, um den aufgezeichneten Datensätzen Verkehrsmittel zuzuordnen und nur Radfahrten in die Auswertung und Visualisierung einzubeziehen. Da die Verkehrsmittelerkennung und anschließende Filterung der Daten in der Cloud erfolgt, werden dennoch auch die unplausiblen Daten vom Smartphone aufgezeichnet und zunächst an den Server übermittelt. Damit werden durch die aktive Internet- und GPS-Nutzung sowohl Bandbreite, als auch Energie verbraucht.

Ziel dieser Arbeit ist es, anhand der bestehenden Machine-Learning-Ansätze zu untersuchen, inwiefern diese auf Smartphones portiert werden können. Hierzu müssen die bestehenden Ansätze hinsichtlich ihrer Portierbarkeit und Anforderungen an Verarbeitungszeit, Energie und Speicher analysiert und verglichen werden. Die am besten geeignete Lösung soll im Rahmen einer Smartphone-App portiert und bezüglich der Ergebnisgüte mit den bestehenden Machine-Learning-Ansätzen verglichen werden. Einen wesentlichen Schwerpunkt der Untersuchungen soll der Trade-off zwischen Ressourcenbedarf und Ergebnisqualität bilden.

Schwerpunkte:

- Recherche existierender Ansätze die Aktivitätserkennung bzw. Verkehrsmittelerkennung mit maschinellen Lernverfahren auf Smartphones
- Analyse und Vergleich der existierenden serverseitigen Machine-Learning Lösungen hinsichtlich Portierbarkeit und Ressourcenbedarf einer möglichen Smartphone-Umsetzung
- Portierung der am besten geeigneten Lösung zur Verkehrsmittelerkennung auf iOS in Form einer prototypischen Smartphone-App
- Experimentelle Evaluation mit Untersuchung des Trade-offs zwischen Ressourcenbedarf und Ergebnisqualität
