# Air Triage App — Hack Yeah 2026

![Logo AirTriage](assets/airtriage-logo.svg)

**AirTriage to oprogramowanie dla ratowników medycznych, które wykorzystuje drony do automatycznego triażu poszkodowanych. System wspiera szybką ocenę ich stanu i ustalanie priorytetów pomocy jeszcze przed dotarciem zespołów ratunkowych — bo w sytuacjach kryzysowych liczy się każda sekunda.**

---

## Konfiguracja

Wymagany Python 3.13 (patrz `.python-version`) i [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Uruchomienie

Otwórz `1_detection_and_movement.ipynb` w katalogu głównym repozytorium i uruchom komórki.
Notebook wczytuje filmy z `data/`, a wyniki zapisuje do `out/`.

Ten sam kod można wywołać bezpośrednio:

```python
from nbutils.tracker import track_speed

track_speed("data/1.mp4", out_path="out/speed_0.mp4", threshold=4.0)
```

## Modele

Wagi detektora leżą w `models/` i **nie są w repozytorium** (`yolov8m.pt` to ok. 50 MB).
Ultralytics pobiera je automatycznie przy pierwszym uruchomieniu, więc po `uv sync` nie trzeba
niczego pobierać ręcznie. Aby użyć innego modelu:

```python
track_speed(..., model_name="models/yolov8n.pt")
```

## Dane

W repozytorium leżą trzy małe filmy demonstracyjne (`data/1.mp4`, `data/drone.mp4`).
Natomiast `data/input/` **nie jest w repozytorium** — `WIN_20261004_07_22_34_Pro.mp4` waży ok. 66 MB,
więc trzeba go skopiować lokalnie przed uruchomieniem `nbutils/2_heart_rate_estimation.ipynb`.
Wyniki tego notatnika (`data/output/`) są generowane i również nie są w repozytorium.

Notatnik wyliczający tętno potrzebuje dodatkowej zależności `vitallens` (ciężka — pobiera
`onnxruntime` i `vitallens-core`):

```bash
uv sync --extra hr
```

## Konfiguracja trackera

Parametry trackingu BoT-SORT znajdują się w `tracker.yaml`.

## Struktura repozytorium

| Ścieżka | Zawartość | W git |
| --- | --- | --- |
| `data/*.mp4` | filmy demonstracyjne | tak |
| `nbutils/` | detekcja, tracking i przetwarzanie wideo | tak |
| `tracker.yaml` | konfiguracja trackera BoT-SORT | tak |
| `models/` | wagi YOLO | nie |
| `data/input/` | filmy wejściowe notatnika HR | nie |
| `data/output/` | filmy z tętnem (wynik notatnika HR) | nie |
| `out/` | filmy z wynikami | nie |
| `runs/` | wyjście Ultralytics | nie |

`nbutils` jest importowane z katalogu głównego repozytorium, a nie instalowane jako pakiet
(`pyproject.toml` ustawia `tool.uv.package = false`), więc uruchamiaj notatniki z korzenia.

`.gitignore` pilnuje też `*.pt`, `*.raw.mp4`, `*.avi`, podglądów `*_preview.webm`,
`.venv/` i cache Pythona. Wygenerowane artefakty można bezpiecznie usuwać — zawsze odtwarzane
przez ponowne uruchomienie. Wyjścia notatników nie są commitowane; przed commitem usuń je poleceniem:

```bash
uvx --from nbconvert jupyter-nbconvert --clear-output --inplace \
    1_detection_and_movement.ipynb nbutils/2_heart_rate_estimation.ipynb
```

