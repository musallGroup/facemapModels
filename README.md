# FaceMap Models

Documents the model development workflow for FaceMap-based pose estimation in head-fixed behavioral recordings.

The repository provides tools for creating, training, refining and maintaining FaceMap models. It is intended to standardize the complete model development process, from dataset creation to quality control, while keeping the workflow modular and reproducible.

Although the current focus is FaceMap, the workflow is designed such that future support for additional pose estimation frameworks (e.g. DeepLabCut) can be integrated with minimal changes.

---

## Repository structure

```
facemapModels/

│── README.md

│── LabelForge/

├── Playground/
│   └── MickDaubenfeld/
│       ├── FrontView/
│       └── SideView/
└── docs/
```

### README.md

Project documentation.

The separation between repository code, network datasets, finished model
packages, and archives is documented in
[`docs/MODEL_ORGANIZATION.md`](docs/MODEL_ORGANIZATION.md).

### LabelForge/

Desktop application for creating base-model datasets and refining existing
FaceMap or DeepLabCut models. It includes frame selection, keypoint grouping,
annotation, review, export, and external-model metadata workflows.

### Playground/

Development area for experimental scripts.

Scripts inside the playground should follow the naming convention

```
fM_functionName_MD.py
```

where

- **fM** = FacemapModel
- **functionName** = descriptive function name
- **MD** = developer initials

Experimental scripts remain inside the playground until they become stable and reusable.

---

## Development workflow

The intended model development workflow consists of the following stages:

```
Video Selection

↓

Frame Extraction

↓

Frame Annotation

↓

Training Dataset

↓

Model Training

↓

Prediction

↓

Quality Control

↓

Satisfied?

├── No  → Refine model

└── Yes → Finished project model
```

The workflow is intentionally iterative. Models are refined over multiple training rounds until satisfactory tracking performance is achieved.

---

## Pipeline stages

The long-term workflow is intended to provide tools covering every stage of model development.

### 1. Frame extraction

Training datasets can be generated using different extraction strategies depending on the application.

Current development focuses on supporting

- random frame extraction across multiple videos
- targeted frame extraction based on behavioral events
- manual frame selection using an interactive video browser

The goal is to simplify the creation of representative and diverse training datasets.

---

### 2. Annotation

Extracted frames are manually annotated to generate training datasets for pose estimation models.

LabelForge provides an integrated annotation and review workflow while remaining compatible with existing labeling pipelines.

---

### 3. Model training

Annotated datasets can be exported into the required format for FaceMap model training.

The repository focuses on preparing datasets while remaining independent of the underlying deep-learning framework.

---

### 4. Prediction

Trained models are applied to complete recordings in order to generate keypoint predictions.

Prediction results can subsequently be used for quality assessment or additional refinement rounds.

---

### 5. Quality control

Every training iteration should be evaluated visually.

The intended workflow generates prediction videos displaying

- predicted keypoints
- prediction confidence
- tracking quality

allowing rapid inspection before additional refinement.

---

### 6. Model refinement

Model development is performed iteratively.

Additional training data can be collected from difficult recordings, labeled and used to refine an existing project model.

Multiple refinement rounds are expected during development.

---

## Base model philosophy

Base models represent generic starting points for a particular recording configuration (camera angle, imaging setup, etc.).

Base models are intentionally kept stable.

Rather than continuously improving a base model using project-specific data, each project initializes its own model from the corresponding base model and performs independent refinement.

This approach

- preserves reproducible baseline models
- prevents project-specific bias from propagating into future projects
- allows multiple specialized models to coexist while sharing the same initialization

---

## Ongoing development

LabelForge is the central application for this workflow. Ongoing development focuses on moving proven Playground scripts into stable application features.

The application workflow covers or is being extended to cover

- dataset management
- frame extraction
- annotation support
- model training
- prediction
- quality control
- model versioning
- base model management
- project-specific model refinement

while remaining compatible with multiple pose estimation frameworks.

One future research direction is the extension of FaceMap with explicit keypoint visibility prediction in addition to keypoint localization.

Instead of relying solely on prediction likelihoods, future models may distinguish between

- visible keypoints
- intentionally occluded keypoints
- unlabeled keypoints

through an additional visibility prediction branch.

---

## Current status

This repository is under active development. LabelForge contains the stable interactive workflow.

Experimental processing and analysis scripts remain in the Playground until they are reusable enough to promote.
