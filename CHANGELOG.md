# LabelForge Changelog

All notable changes to LabelForge are documented here.
Versions refer to the internal build number tracked by the development team.

---

## v36 — 2026-09-04

### QC Video
- **Cleaner keypoint dots** — replaced filled-circle + outline-circle (which looked like double rings) with a dark backing circle drawn first and the coloured dot on top; gives a crisp coloured dot with a natural dark border on both the full frame and the zoom panel

### Installer
- **GPU install order fixed** — CUDA-enabled PyTorch is now installed *after* Facemap, not before; `pip install facemap` pulls in CPU-only torch as a dependency and would silently overwrite a pre-installed CUDA build — installing last guarantees the CUDA wheel wins

### Docs
- **CHANGELOG pre-versioning history** — added detailed development history from 13.08.–26.08.2026 (early prototype, Create Base Model, Refine Existing Model)

---

## v35 — 2026-09-04

### QC Video
- **Zoom panel no longer stretches** — the cropped region is now letterboxed (aspect ratio preserved, black padding) instead of being scaled to fill the full panel height
- **Larger keypoint dots** — dot radius now scales with frame resolution (~0.6 % of the shorter frame dimension) instead of a fixed 3-pixel radius; all dots get a thin black outline for visibility on bright backgrounds
- **Keypoints drawn on the zoom panel** — predictions are now re-projected and drawn on the zoom panel too, not only on the full frame
- **Colors match the Label Maker** — the QC video now reads per-keypoint hex colors from the export manifest (`facemap_export.json`) and passes them through the training bundle into `training_manifest.json`; the zoom panel header shows `ZOOM • <focus label>` with a dark background strip

### Export
- `facemap_export.json` now includes a `keypoint_colors` field (name → hex) so downstream tools and QC scripts can use the same colors as the label editor

### Bundle
- `training_manifest.json` now includes `keypoint_colors` when a `facemap_export.json` is found next to the selected `labels.csv`

---

## v34 — 2026-09-04

### SSH / Remote
- **Remote Workstation no longer requires TOTP** — the JUSUF-specific MAC algorithm (`hmac-sha2-256-etm@openssh.com`) and `keyboard-interactive` authentication are now only applied when connecting to a `juelich.de` host; other SSH targets (e.g. internal CMP workstations) connect without MFA
- Added `_needs_totp()` helper in `workflow.py`; all four SSH operations (`Transfer`, `Start`, `Check`, `Fetch`) only block for a code when TOTP is actually required

### UI
- **Tab underline clipping fixed** — the orange active-tab indicator on "Training Workspace" was clipped by the TopBar container; fixed by reducing `padding-bottom` by 3 px on the `:checked` state to compensate for the border width

### Installer
- `pip install "numpy<2"` is now run automatically after Facemap installs, overwriting any numpy 2.x that torch/torchvision pulled in as a dependency (Facemap 1.0.8 is incompatible with numpy ≥ 2.0)

---

## v33 — 2026-09-03

### Local Training
- **Live epoch output** — `conda run` is now called with `--no-capture-output` and `python -u` so training progress (Epoch N: loss …) streams to the log in real time instead of appearing only after the run finishes
- **`__file__` NameError fixed** — `bundle.py` now injects `__file__` into the `exec()` namespace for both `training_entry.py` and `facemap_qc.py`; scripts that use `Path(__file__).resolve().parent` no longer crash

### Installer
- **GPU auto-detection** — the Install button now calls `shutil.which("nvidia-smi")`; if an NVIDIA GPU is found it installs CUDA-enabled PyTorch (`torch`, `torchvision` from `https://download.pytorch.org/whl/cu124`) before Facemap; the confirmation dialog tells the user whether GPU acceleration will be used

### SSH / MFA
- **TOTP popup** — added `_ask_totp()` in `workflow.py`; if the TOTP field is empty when Transfer / Start / Check / Fetch is pressed, a `QInputDialog` popup asks for the code instead of silently failing or requiring the user to scroll up
- `QInputDialog` and `shutil` imports added to `workflow.py`

### Build
- **LabelForgeAskpass.exe permanently bundled** — added to `LabelForge.spec` binaries so it is always included in PyInstaller output and never missing after a fresh build

---

## v32 — 2026-09-02

### HPC / JUSUF
- **libGL.so.1 error fixed** — `facemap_training_adapter.py` and `facemap_qc.py` now inject `unittest.mock.MagicMock` stubs for all Qt/PyQt5/pyqtgraph modules before importing Facemap, preventing the `ImportError: libGL.so.1` crash on headless compute nodes
- **Facemap model cache** — documented that `model_loader.get_model_params_path()` must be run once on the login node (which has internet) to populate `~/.facemap/models/`; compute nodes read from the shared filesystem

### SSH
- **SCP `-m` flag removed** — Windows OpenSSH `scp` does not support `-m`; replaced with `-o MACs=hmac-sha2-256-etm@openssh.com`

---

## v31 — 2026-09-01

### HPC / JUSUF
- Installed missing Python packages into the JUSUF Facemap venv: `h5py`, `matplotlib`, `natsort`, `numba`, `scikit-learn`, `tqdm`, `cellpose`
- Pinned numpy and torch back to system versions after `cellpose` installed incompatible numpy 2.x and torch 2.13.0
- Training confirmed running on JUSUF via `sbatch` (compute node, not login node)

### Policy
- All heavy computation (training, QC) moved to Slurm compute nodes via `sbatch slurm_job.sh`; running workloads on the JUSUF login node is a policy violation

---

## Training Workspace (pre-versioning, ab ~27.08.2026)

*Zeitraum vor v31, kein separates Versionstag.*

- Initial Facemap training bundle implementation (`facemap_training_adapter.py`, `facemap_qc.py`, `bundle.py`)
- JUSUF SSH workflow: preflight, transfer, start, check status, fetch results
- LabelForge Askpass helper for MFA-gated SSH connections
- QC clip pre-cutting during bundle creation to reduce transfer size
- Static zoom anchor computed from median keypoint position
- Job status timer and `sacct` fallback for completed/failed JUSUF jobs

---

## Refine Existing Model — 26.08.2026

Git-Commit: `ec286e9`

- Auswahl eines vorhandenen Facemap-`.pt` oder DLC-Projektordners als Parent-Modell
- Parent-Model-Metadaten werden gespeichert; Keypoint-Schema bei Refinement gesperrt
- Automatische nächste Versionsnummer aus dem Parent abgeleitet
- Externe Facemap-Modelle und DLC-Projekte importierbar
- Keypoints aus Facemap-CSV oder DLC-`bodyparts` ausgelesen
- Gruppen, Farben und Shortcuts für externe Modelle konfigurierbar
- `labelforge_model.json` als persistentes Modell-Metadaten-Format
- Header-Freischaltung erst nach erfolgreicher Parent-Auswahl
- Dark-Theme-Importdialog mit lesbarer Schrift
- QtCore-/ICU-PyInstaller-Problem behoben

---

## Vor der Versionierung — 13.08. bis 26.08.2026

*Früher Prototyp, keine Build-Nummern. Belege: `Backup\app V1.py`, `Backup\app(3).py`, `Backup\main_window V1.py`, `Backup\main_window V2.py`.*

### App-Grundgerüst
- Erste PySide6-App mit MainWindow
- Startseite mit „Label Workspace" und „Training Workspace" als Einstieg
- Dunkles Design mit Orange als Akzentfarbe, Corporate-Design-Elemente
- LabelForge-Icon und Extended Logo

### Create Base Model
- Projektinformationen, Keypoint-Gruppen und Farbpaletten
- Frame-Auswahl und Video-Vorschau
- Frame-Extraktion → `extraction_manifest.csv`
- Labeling-Oberfläche: visible / not_visible / unset
- Frame Mode und Keypoint Mode
- Zoom, Pan, Helligkeit, Kontrast und Gamma
- Review-Seite mit gruppierten Issues
- Facemap-Export und DeepLabCut-Projektexport
- DLC-Smoke-Test und Facemap-Strukturtest
- PyInstaller-Build mit MKL/OpenMP
