# Model organization

The Git repository contains source code and documentation. Trained models,
labels, videos, extracted frames, predictions, and QC outputs remain on the
lab network.

## LabelForge-ready basemodels

Finished models intended as LabelForge parent models are exposed through these
network entry points:

- `Z:\Team\Mick\FM_Front_View\00_LabelForge_Basemodels`
- `Z:\Team\Mick\FM_Side_View\00_LabelForge_Basemodels`

Each model package contains:

- one promoted `.pt` model;
- `labelforge_model.json` with version, groups, colors, and locked output order;
- `labels.csv` and `keypoints.txt` as readable schema fallbacks;
- a package README with links to training and QC sources;
- `training_model_info.json` when source training metadata exists.

The currently promoted packages are:

| Acquisition | Purpose | Package |
|---|---|---|
| Front view | General face and tongue | `FrontView_FaceTongue_General_v1` |
| Side view, general | Face / pose, Mick cohort | `SideView_Face_Mick_v1` |
| Side view, general | Face / pose, Dennis cohort | `SideView_Face_Dennis_v1` |
| Side view, general | Eye / pupil | `SideView_Pupil_General_v1` |
| Side view, 2P | Face / pose | `SideView_Face_2P_v1` |
| Side view, 2P | Eye / pupil | `SideView_Pupil_2P_v1` |

Public model names use <View>_<Target>_<Dataset>_vN. Historical training filenames remain unchanged in their source folders so old scripts and training records stay reproducible.

## Data lifecycle

- `Current_Model` is the working source location for the promoted model.
- `00_LabelForge_Basemodels` is the stable and clearly visible LabelForge entry point.
- `Training_Data` and `Refinement_Training` retain labels needed for reproducibility.
- `Model_QC` contains evidence used to promote a model.
- `Cohort` and `Cohort_Summary` contain derived project results.
- `Archive` must have an inventory and a reason for retention.

Do not keep several undocumented versions in `Current_Model`. Older deliberate
releases belong in `Model_Releases`; failed, superseded, or reproducible
intermediates may be removed after verification.

## Archive deletion checklist

Before deleting an old dataset, verify that:

1. no active script or metadata refers to it;
2. it contains no promoted model used for analysis;
3. unique labels are either intentionally discarded or preserved in a compact archive;
4. published or ongoing analyses do not require it;
5. the exact deletion target and recovered space are recorded.
