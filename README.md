# AdRob
Framework zur Bewerkstelligung und Auswertung von Evasion-Blackbox-Angriffen auf Modelle, welche per Webservice bereitgestellt werden.

![image](https://github.com/MareenDev/AdRob/assets/115465960/4b0b8d64-6e78-4e09-a5b5-47cd2c466c9a)

Für einen speziellen Angriffstyp (bspw. nicht-zielgerichteter HopSkipJump-Angriff) können verschiedene Angriffskonfigurationen durch einen Konsolenaufruf gestartet werden. Eine Erweiterung um neue Angriffstypen ist möglich. 

Die bei Angriffsausführung berechneten Daten werden für eine anschließende Auswertung gespeichert. 

Die Implementierung sieht desweiteren eine Erweiterung um Maße für die Bewertung eines Modells vor.
Zudem können Metriken definiert werden, mit welchen sich verschiedene Modelle vergleichen lassen.

Für jeden anzugreifenden Webservice ist ein Adapter, der die Daten der Webservice-Schnittstellen entsprechend aufarbeitet, zu implementieren.

## Was sind Evasion-Blackbox-Angriffe?
Evasion-Angriffe (auch Adversarialer Angriff) sind Angriffe auf die Integrität von ML-Systemen zur Klassifikation. Diese Angriffe erfolgen während der Testphase eines Klassifikators, oder nach dessen Bereitstellung im Produktivbetrieb. Die Angriffsdurchführung beruht darauf, dass die Daten vor Eingabe in das System mit einer leichten, für den Menschen gegebenenfalls nicht wahrnehmbaren, Störung (perturbation) versehen werden, um eine (für den Menschen) nicht erwartungsgemäße Ausgabe zu provozieren. Steht einem Angreifer ausschließlich ein Anfrage-Zugriff auf das System zur Verfügung, so spricht man von einem Blackbox-Evasion Angriff. Das anzugreifende System stellt in diesem Fall eine Blackbox für den Angreifer dar, dessen interne Entscheidungswege nicht einsehbar sind.

Der Angriff auf einen Webservice, welcher Eingaben entgegennimmt und mittels ML-System eine Klassifikation zurück gibt, stellt ein Beispiel für einen Evasion-Blackbox-Angriff dar.

Für weitere Informationen siehe auch [Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations
](https://csrc.nist.gov/pubs/ai/100/2/e2023/final), oder [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572). Hinweis zum  Zusammenhang von Evasion-Angriffe und adversarialen Beispielen: Evasion-Angriffe basieren auf dem Vorhandensein adversarialer Beispiele und stellen eine gezielte Suche nach diesen dar.

## Verwendete Bibliotheken
Diese Implementierung basiert auf der Nutzung verschiedener Bibliotheken.
Insbesondere werden 
* [**Adversarial Robustness Toolbox**](https://adversarial-robustness-toolbox.readthedocs.io/) (ART) - für die Durchführung verschiedener Angriffe
* [**numpy**](https://numpy.org/)  - zur Abspeicherung von Bilddaten
* [**torchvision**](https://pytorch.org/vision/stable/index.html) - Zur Datenbereitstellung und Bildtransformation
* [**matplotlib**](https://matplotlib.org/stable/) - Zur Erstellung von Graphen bei der Auswertung
verwendet.

## Implementierte Angriffe
|Angriff|Paper|Bemerkung|
| :-------------: |:-------------:| :-----:|
| HopSkipJump| https://arxiv.org/abs/1904.02144|basierend auf ART-Implementierung|
| GeoDa|https://arxiv.org/abs/2003.06468|basierend auf ART-Implementierung|
| SignOPT|https://arxiv.org/pdf/1909.10773.pdf|basierend auf ART-Implementierung|

## Implementierte Dateninterpreter (Maße und Metriken)
|Interpreter|Bemerkung|
| :-------------: | :-----:|
| Robuste Akkuranz| |
| lp- Störungsgröße| |
| DeepFoolMeasure| |
| Robuste Akkuranz in Abhängigkeit der Queryanzahl| |
| Robuste Akkuranz in Abhängigkeit der Störungsgröße| |
| RobustnessCurve| in Anlehnung an https://arxiv.org/abs/1908.00096 |

## Installation
1. Conda-Umgebung erzeugen (per Anaconda Prompt)
   ```
   conda create -n adRob python==3.10.9
   ```

2. Conda-Umgebung aktivieren (per Anaconda Prompt)
   ```
   conda activate adRob
   ```

3. Repository Clonen (per Anaconda Prompt)
   ```
   git clone https://github.com/MareenDev/AdRob.git
   ```
4. Repository in Editor öffnen
   ```
   cd adrob
   code
   ```

5. Bibliotheken aus requirements.txt in Conda-Umgebung laden 
   ```
   pip install -r requirements.txt
   ```
## Aufbau und Verwendung

Die Implementierung umfasst Skripte mit Ablauflogik und Skripte zur Bereitstellung von Objekten und Funktionen.

### Skripte mit Ablauflogik 
Die folgenden Skripte sind für die Verwendung der Implementierung per Konsolenaufruf auszuführen.
In einem ersten Schritt sind Ausgangsdaten(z.B. gelabelte Testdaten) für den Evasion-Angriff zusammenzutragen und aufzubereiten. Anschließend kann die Durchführung von Evasion-Angriffen auf Basis dieser Daten vorgenommen werden.
Nach Abschluss der Angriffsausführung können die im Angriff gesammelten Daten (berechnete Eingabedaten, Ausgabedaten, Aufrufanzahl) analyiert/interpretiert werden.

* **getData.py** - Dient der Sammlung von Ausgangsdaten für die Verwendung im Evasion-Angriff. Die Logik verwendet insb. Objekte des Skriptes **classes/attack.py**. Daten können bspw. per Googlebildersuche, oder durch auslesen von Torchvision-Dataset erfolgen.
  Gesammelte Daten werden in Ordner **data/input** abgespeichert.
* **prepareData.py** - Bereitet die Daten aus einem Ordner (i.A. data/input/...) für die Weiterverarbeitung durch das Framework auf und speichert diese als Numpy-Arrays ab.
* **runattack.py** - Durchführung von Evasion-Angriffen. Mit einem Skriptlauf können verschiedene Angriffskonfigurationen durchgeführt werden. Angriffsdaten werden unter **data/interpretation** abgespeichert
* **analyseModel.py** - Ermittelung von Maßen zu einer Angriffsausführung -
* **compareModels.py** - Vergleich mehrerer Maße zu verschiedenen Angriffsausführungen


### Skripte zur Objektdefinition & Funktionen 
Ordner **classes** kapselt die Logik für Objektdefinitionen und sonstige Funktionen. Eine Erweiterung aller enthaltener Skripte, mit Ausnahme des Skriptes classes/framework.py, ist vorgesehen.

* **classes/attack.py** definiert verfügbare Angriffe - Kann erweitert werden 
* **classes/data.py** führt Domänenspezifische Logik - Kann erweitert werden
* **classes/framework.py** umfasst Logik des Rahmenwerkes - Nicht zu erweitern
* **classes/metric.py** umfasst Metriken - Ausbaustufe!
* **classes/models.py** umfasst die API/Webservice-spezifische Logik für den Requestaufbau zu einem Angriff. - Bei Anwendung auf neuen Webservice sind hier entsprechende Objekte vom Typ RestApiCall zu erzeugen.
* **classes/plots.py** - Ausbaustufe
* **classes/visualize.py** - Ausbaustufe



