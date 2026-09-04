# LabelForge

Desktop application for creating, training, refining and quality-controlling FaceMap pose estimation models.

**Status: Playground / experimental — not yet released.**

---

## What this does

LabelForge provides an end-to-end workflow for FaceMap model development:

1. **Label Workspace** — Select frames from videos, define keypoint groups with colors, annotate body parts, review and export a training dataset
2. **Training Workspace** — Package a training run (model name, data, parameters), run it locally or submit it to JUSUF HPC via Slurm, monitor progress, fetch results and generate a QC video

---

## How to run (development)

```
conda activate fm_front        # or your local PySide6 environment
cd Playground/MickDaubenfeld/LabelForge
python app.py
```

---

## How to build the EXE

```
cd Playground/MickDaubenfeld/LabelForge

# Build the MFA askpass helper first (only needed once or after changes to ssh_askpass.py)
pyinstaller --noconfirm LabelForgeAskpass.spec

# Build the main app
pyinstaller --noconfirm LabelForge.spec

# Output: dist/LabelForge/LabelForge.exe
#         dist/LabelForge/LabelForgeAskpass.exe  (copied automatically via spec)
```

---

## File structure

```
LabelForge/
├── app.py                          ← entry point
├── LabelForge.spec                 ← PyInstaller config for the main app
├── LabelForgeAskpass.spec          ← PyInstaller config for the SSH MFA helper
├── ssh_askpass.py                  ← SSH askpass helper source
├── CHANGELOG.md                    ← full version history
│
├── labelforge/
│   ├── assets/                     ← icons and images
│   ├── model_metadata.py
│   └── ui/
│       ├── main_window.py          ← app shell and tab navigation
│       ├── common/                 ← shared widgets (image viewer, dialogs)
│       ├── create_base/            ← Label Workspace (keypoints, frames, annotation, export)
│       ├── refine_model/           ← Refine / Specialize workflow
│       └── training_workspace/     ← Training Workspace
│           ├── workflow.py         ← main UI and SSH orchestration
│           ├── bundle.py           ← training bundle builder
│           ├── remote.py           ← SSH/SCP command builders
│           ├── facemap_training_adapter.py  ← headless Facemap trainer (runs on HPC)
│           ├── facemap_qc.py       ← QC video generator (runs on HPC or locally)
│           └── naming.py           ← model name helpers
│
├── LF_checkFacemapExport_MD.py     ← dev: inspect a Facemap export folder
├── LF_checkDlcProject_MD.py        ← dev: inspect a DeepLabCut project folder
└── LF_devLabeling_MD.py            ← dev: standalone labeling prototype
```

> `app.py`, the spec files and the `labelforge/` package keep their original names —
> renaming them would break PyInstaller and Python imports.
> Dev/helper scripts follow the `LF_functionName_MD.py` convention (LF = LabelForge).

---

## Version history

See [CHANGELOG.md](CHANGELOG.md) for the full change log.

| Build | Date | Highlights |
|---|---|---|
| v35 | 2026-09-04 | QC letterbox zoom, label-maker colors, bigger dots, keypoints on zoom panel |
| v34 | 2026-09-04 | Remote Workstation SSH (no TOTP for non-JUSUF), tab underline fix, `numpy<2` in installer |
| v33 | 2026-09-03 | Live epoch output, `__file__` fix, GPU auto-detect in installer, TOTP popup |
| v32 | 2026-09-02 | Headless HPC fix (MagicMock), Facemap model cache, SCP MAC fix |
| v31 | 2026-09-01 | JUSUF venv packages, numpy/torch pin, training on compute node via sbatch |

Large models, videos, labels and generated builds are git-ignored.
Finished model packages are stored on the lab network; see
[`../../../docs/MODEL_ORGANIZATION.md`](../../../docs/MODEL_ORGANIZATION.md).
