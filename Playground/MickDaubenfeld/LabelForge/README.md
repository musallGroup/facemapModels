# LabelForge

LabelForge is a desktop workflow for creating Facemap base-model datasets and
refining existing Facemap or DeepLabCut models.

## Current capabilities

- select representative frames from videos;
- define, group, color, and label keypoints;
- review annotations before export;
- export training datasets and project metadata;
- import external Facemap models and recover their keypoint schema;
- use finished models as versioned parent models for refinement.

## Run from source

1. Create or activate a Python environment.
2. Install the runtime dependencies, including `PySide6`.
3. From this directory, run `python app.py`.

Large models, videos, labels, and generated builds are intentionally excluded
from Git. Finished model packages are stored on the lab network; see
[`../docs/MODEL_ORGANIZATION.md`](../docs/MODEL_ORGANIZATION.md).
