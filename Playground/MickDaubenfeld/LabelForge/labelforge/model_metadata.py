from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re
from datetime import datetime, timezone

class ModelMetadataError(ValueError): pass

@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    model_family: str
    version: int
    backend: str
    keypoint_schema: dict
    source_path: Path
    @property
    def suggested_next_version(self): return f"{self.model_family}_v{self.version + 1}"
    def to_keypoint_draft(self): return {"locked": True, "groups": self.keypoint_schema["groups"]}

def split_model_version(name):
    match = re.fullmatch(r"(.+?)_v(\d+)", name.strip(), re.I)
    if not match: raise ModelMetadataError("Model name must end with a version such as Pupil_Base_v2.")
    return match.group(1).strip(" _"), int(match.group(2))

def load_parent_metadata(selection):
    selected = Path(selection)
    if not selected.exists(): raise ModelMetadataError("The selected parent model does not exist.")
    folder = selected if selected.is_dir() else selected.parent
    sidecar = next((folder/n for n in ("labelforge_model.json","model_metadata.json") if (folder/n).is_file()), None)
    if sidecar is None and selected.is_file() and selected.with_suffix(".json").is_file(): sidecar=selected.with_suffix(".json")
    if sidecar is None: raise ModelMetadataError("No LabelForge metadata sidecar was found. Keypoints and version cannot be guessed safely.")
    try: data=json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc: raise ModelMetadataError(f"Could not read metadata: {exc}") from exc
    name=str(data.get("model_name","")).strip()
    family,version=split_model_version(name)
    if str(data.get("model_family",family)) != family or int(data.get("version",version)) != version: raise ModelMetadataError("Model name, family and version do not match.")
    backend=str(data.get("backend","")).lower()
    if backend not in {"facemap","deeplabcut"}: raise ModelMetadataError("Backend must be facemap or deeplabcut.")
    if selected.is_file() and (selected.suffix.lower()!=".pt" or backend!="facemap"): raise ModelMetadataError("Select a Facemap .pt model with Facemap metadata.")
    if selected.is_dir() and (not (selected/"config.yaml").is_file() or backend!="deeplabcut"): raise ModelMetadataError("Select a DLC project with config.yaml and DLC metadata.")
    groups=data.get("keypoint_schema",{}).get("groups")
    if not isinstance(groups,list) or not groups: raise ModelMetadataError("No keypoint schema was found.")
    seen=set()
    for group in groups:
        if not group.get("keypoints"): raise ModelMetadataError("Every keypoint group must contain keypoints.")
        group.setdefault("name","Keypoints"); group.setdefault("palette","Orange")
        for keypoint in group["keypoints"]:
            kp_name=str(keypoint.get("name","")).strip()
            if not kp_name or kp_name in seen: raise ModelMetadataError("Keypoint names must be present and unique.")
            seen.add(kp_name); keypoint.setdefault("shortcut",""); keypoint.setdefault("color","#D18B47")
    return ModelMetadata(name,family,version,backend,{"locked":True,"groups":groups},sidecar)

def read_dlc_bodyparts(config_path):
    lines=Path(config_path).read_text(encoding="utf-8-sig").splitlines()
    bodyparts=[]; in_bodyparts=False; base_indent=0
    for raw in lines:
        clean=raw.split("#",1)[0].rstrip()
        if not clean.strip(): continue
        indent=len(clean)-len(clean.lstrip())
        if clean.lstrip().startswith("bodyparts:"):
            in_bodyparts=True; base_indent=indent
            inline=clean.split(":",1)[1].strip()
            if inline.startswith("[") and inline.endswith("]"):
                return [item.strip().strip("'\"") for item in inline[1:-1].split(",") if item.strip()]
            continue
        if in_bodyparts:
            stripped=clean.strip()
            if stripped.startswith("-"):
                name=stripped[1:].strip().strip("'\"")
                if name: bodyparts.append(name)
                continue
            if indent<=base_indent: break
    if not bodyparts: raise ModelMetadataError("No bodyparts list was found in config.yaml.")
    return bodyparts

def _keypoint_names_from_groups(groups):
    names=[]
    for group in groups or []:
        for keypoint in group.get("keypoints",[]):
            name=str(keypoint.get("name","")).strip()
            if name and name not in names: names.append(name)
    return names

def read_facemap_labels_csv(csv_path):
    import csv
    try:
        with Path(csv_path).open(newline="",encoding="utf-8-sig") as handle:
            fields=next(csv.reader(handle))
    except (OSError,StopIteration) as exc:
        raise ModelMetadataError(f"Could not read labels.csv: {exc}") from exc
    names=[]
    for field in fields:
        name=str(field).strip()
        if name.endswith("_x") and name[:-2] not in names: names.append(name[:-2])
    if not names: raise ModelMetadataError("No <keypoint>_x columns were found in labels.csv.")
    return names

def read_facemap_refined_bodyparts(data_path):
    try:
        import numpy as np
    except ImportError as exc:
        raise ModelMetadataError("NumPy is required to read Facemap refined-data files.") from exc
    try:
        loaded=np.load(Path(data_path),allow_pickle=True)
        data=loaded.item() if getattr(loaded,"shape",None)==() else loaded
    except (OSError,ValueError,EOFError) as exc:
        raise ModelMetadataError(f"Could not read Facemap refined data: {exc}") from exc
    if not isinstance(data,dict): raise ModelMetadataError("Facemap refined data does not contain a metadata dictionary.")
    raw=data.get("bodyparts")
    if raw is None: raise ModelMetadataError("Facemap refined data has no bodyparts field.")
    names=[str(name).strip() for name in raw if str(name).strip()]
    if not names or len(names)!=len(set(names)):
        raise ModelMetadataError("Facemap bodyparts must be present and unique.")
    return names

def discover_facemap_keypoints(model_path):
    """Return (keypoint names, source path) for a Facemap model without loading the .pt."""
    model=Path(model_path)
    if not model.is_file() or model.suffix.lower()!=".pt":
        raise ModelMetadataError("Select a Facemap .pt model.")

    for name in ("labelforge_model.json","model_metadata.json",f"{model.stem}.json"):
        sidecar=model.parent/name
        if not sidecar.is_file(): continue
        try:
            payload=json.loads(sidecar.read_text(encoding="utf-8"))
            names=_keypoint_names_from_groups(payload.get("keypoint_schema",{}).get("groups"))
        except (OSError,json.JSONDecodeError,AttributeError) as exc:
            raise ModelMetadataError(f"Could not read model metadata: {exc}") from exc
        if names: return names,sidecar

    labels=model.parent/"labels.csv"
    if labels.is_file(): return read_facemap_labels_csv(labels),labels

    refined_name=f"{model.stem}_Facemap_refined_data.npy"
    cohort_root=model.parent.parent
    candidates=(
        model.parent/refined_name,
        model.parent/"Facemap_Finetuning"/refined_name,
        cohort_root/"Facemap_Finetuning"/refined_name,
        cohort_root/"Refinement_Training"/"Facemap_Finetuning"/refined_name,
    )
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file(): return read_facemap_refined_bodyparts(candidate),candidate
    raise ModelMetadataError(
        "No keypoint metadata was found. Place labelforge_model.json, labels.csv, "
        f"or {refined_name} beside the model or in Refinement_Training/Facemap_Finetuning."
    )

def write_external_metadata(selection, model_family, version, groups, keypoint_source=None):
    selected=Path(selection)
    family=model_family.strip().replace(" ","_")
    if not family: raise ModelMetadataError("Enter a model family.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+",family):
        raise ModelMetadataError("Model family may only contain letters, numbers, underscores and hyphens.")
    if int(version)<1: raise ModelMetadataError("Version must be at least 1.")
    backend="facemap" if selected.is_file() else "deeplabcut"
    seen=set()
    for group in groups:
        for keypoint in group.get("keypoints",[]):
            name=keypoint.get("name","").strip()
            if not name or name in seen: raise ModelMetadataError("Keypoint names must be present and unique.")
            seen.add(name)
    if not seen: raise ModelMetadataError("Add at least one keypoint.")
    destination=(selected.parent if selected.is_file() else selected)/"labelforge_model.json"
    payload={
        "schema_version":1,
        "model_name":f"{family}_v{int(version)}",
        "model_family":family,
        "version":int(version),
        "backend":backend,
        "model_path":str(selected),
        "parent_model":None,
        "keypoint_schema":{"locked":True,"groups":groups},
        "training_dataset":None,
        "keypoint_source":str(keypoint_source) if keypoint_source else None,
        "training_settings":{},
        "created_at":datetime.now(timezone.utc).isoformat(),
        "imported_external_model":True,
    }
    try: destination.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    except OSError as exc: raise ModelMetadataError(f"Could not save metadata beside the model: {exc}") from exc
    return destination
