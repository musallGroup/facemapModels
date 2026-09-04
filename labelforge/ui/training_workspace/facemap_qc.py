"""Create a short full-frame plus focused Facemap QC preview."""
import json,sys
from pathlib import Path
from unittest.mock import MagicMock
# Headless HPC: mock Qt/GUI modules — not needed for inference on compute nodes.
for _m in [
    "PyQt5","PyQt5.QtGui","PyQt5.QtWidgets","PyQt5.QtCore","PyQt5.QtOpenGL",
    "qtpy","qtpy.QtGui","qtpy.QtWidgets","qtpy.QtCore",
    "pyqtgraph",
    "facemap.gui","facemap.gui.help_windows","facemap.gui.gui",
    "facemap.gui.io","facemap.gui.utils",
]:
    if _m not in sys.modules: sys.modules[_m]=MagicMock()
import cv2,numpy as np,torch
from facemap.pose.pose import Pose
R=Path(__file__).resolve().parent
C=json.loads((R/"training_manifest.json").read_text())

def rp(runtime,source):
    p=Path(C.get(runtime) or C.get(source,""));return p if p.is_absolute() else R/p

def normalize(value,n,k):
    raw=value[0] if isinstance(value,tuple) else value
    if isinstance(raw,torch.Tensor):raw=raw.detach().cpu().numpy()
    a=np.asarray(raw)
    if a.shape==(k,n,3):a=np.transpose(a,(1,0,2))
    if a.shape!=(n,k,3):raise RuntimeError(f"Unexpected Facemap prediction shape {a.shape}; expected {(n,k,3)}")
    return a

def hex_to_bgr(h):
    """Convert a CSS hex color (#RRGGBB) to an OpenCV BGR tuple."""
    h=h.lstrip("#")
    if len(h)==3:h=h[0]*2+h[1]*2+h[2]*2
    r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return (b,g,r)

# Default palette (BGR) used when the manifest carries no per-keypoint colors.
_DEFAULT_COLORS=[(225,153,66),(120,187,72),(54,137,237),(234,122,159),(101,101,245),(172,178,56)]

def build_color_map(labels):
    """Return a list of BGR colors aligned to `labels`, using manifest colors when available."""
    hex_map=C.get("keypoint_colors",{})
    result=[]
    for i,name in enumerate(labels):
        if name in hex_map:
            result.append(hex_to_bgr(hex_map[name]))
        else:
            result.append(_DEFAULT_COLORS[i%len(_DEFAULT_COLORS)])
    return result

def letterbox_zoom(crop, panel_w, panel_h):
    """Resize crop to fit panel_w × panel_h keeping aspect ratio; pad with black."""
    ch,cw=crop.shape[:2]
    scale=min(panel_w/cw, panel_h/ch)
    nw,nh=int(cw*scale),int(ch*scale)
    scaled=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_CUBIC)
    canvas=np.zeros((panel_h,panel_w,3),dtype=np.uint8)
    x0=(panel_w-nw)//2; y0=(panel_h-nh)//2
    canvas[y0:y0+nh,x0:x0+nw]=scaled
    return canvas,x0,y0,scale

def main():
    if not C.get("qc_enabled",True):return
    results=R/"results";qc=results/"qc";qc.mkdir(parents=True,exist_ok=True)
    model=results/(C["model_name"]+".pt");info=results/(C["model_name"]+"_model_info.json")
    if not model.is_file() or not info.is_file():raise RuntimeError("QC needs the trained model and model-info JSON")
    labels=json.loads(info.read_text()).get("labels",[])
    if not labels:raise RuntimeError("QC cannot find the trained keypoint names")
    colors=build_color_map(labels)
    # Prefer the pre-cut QC clip (new bundles) — it is already trimmed to the
    # right segment so we use all its frames.  Fall back to the full video for
    # older bundles that do not carry a clip.
    clip_key=C.get("runtime_qc_clip","")
    if clip_key:
        clip_path=Path(clip_key) if Path(clip_key).is_absolute() else R/clip_key
        video=clip_path;precut=clip_path.is_file()
    else:
        video=rp("runtime_qc_video","qc_video") if C.get("qc_video") or C.get("runtime_qc_video") else rp("runtime_initialization_video","initialization_video");precut=False
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError(f"QC video cannot be opened: {video}")
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 30)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if precut:
        count=total;start=0
    else:
        count=min(total,max(1,int(float(C.get("qc_duration_seconds",60))*fps)));start=max(0,(total-count)//2)
    indices=np.arange(start,start+count,dtype=np.int64)
    pose=Pose(filenames=[[str(video)]],bbox=[],bbox_set=False,resize=False,add_padding=False,gui=None,GUIobject=None,net=None,model_name=str(model))
    pose.bodyparts=list(labels);pose.batch_size=8;pose.pose_prediction_setup()
    pred=normalize(pose.predict_landmarks(video_id=0,frame_ind=indices),len(indices),len(labels))
    try:
        import h5py
        with h5py.File(qc/"predictions.h5","w") as f:
            f.attrs["video_path"]=str(video);f.attrs["model_path"]=str(model)
            f.create_dataset("frame_index",data=indices,compression="gzip")
            g=f.create_group("Facemap")
            for i,name in enumerate(labels):
                q=g.create_group(name)
                q.create_dataset("x",data=pred[:,i,0],compression="gzip")
                q.create_dataset("y",data=pred[:,i,1],compression="gzip")
                q.create_dataset("likelihood",data=pred[:,i,2],compression="gzip")
    except ImportError:
        np.savez_compressed(qc/"predictions.npz",frame_index=indices,predictions=pred,labels=np.asarray(labels))
    focus=str(C.get("qc_focus_label","")).strip()
    if focus not in labels:
        focus=next((x for word in ("pupil","eye","tongue","nose","mouth") for x in labels if word in x.lower()),labels[0])
    fi=labels.index(focus)
    context=float(C.get("qc_zoom_context",1.0))
    # Zoom crop: sized as a fraction of the frame so it scales with resolution.
    cw=max(80,int(min(w,h)*0.35*context))
    ch=max(60,int(min(w,h)*0.35*context))
    threshold=0.2
    # Compute a static zoom anchor from high-confidence predictions of the focus keypoint.
    valid=pred[:,fi,2]>=threshold
    if valid.any():
        anchor_x=float(np.median(pred[valid,fi,0]));anchor_y=float(np.median(pred[valid,fi,1]))
    else:
        anchor_x=w/2;anchor_y=h/2
    x1z=max(0,min(w-cw,int(anchor_x-cw/2)));y1z=max(0,min(h-ch,int(anchor_y-ch/2)))
    # Dot radius scales with frame size so it's visible on all resolutions.
    dot_r=max(4,int(min(w,h)*0.006))
    panel_w=max(360,min(640,w))
    writer=cv2.VideoWriter(str(qc/"preview.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w+panel_w,h))
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    for j in range(len(indices)):
        ok,frame=cap.read()
        if not ok:break
        row=pred[j]
        # Draw keypoints on full frame (dark backing circle first, colour on top)
        for i,(x,y,lik) in enumerate(row):
            if np.isfinite(x) and np.isfinite(y) and lik>=threshold:
                cv2.circle(frame,(int(x),int(y)),dot_r+2,(0,0,0),-1,cv2.LINE_AA)
                cv2.circle(frame,(int(x),int(y)),dot_r,colors[i],-1,cv2.LINE_AA)
        # Build zoom panel (letterboxed — no stretching)
        crop=frame[y1z:y1z+ch,x1z:x1z+cw]
        zoom,zx0,zy0,zscale=letterbox_zoom(crop,panel_w,h)
        # Draw keypoints on the zoom panel at adjusted coordinates
        for i,(x,y,lik) in enumerate(row):
            if np.isfinite(x) and np.isfinite(y) and lik>=threshold:
                lx=x-x1z; ly=y-y1z
                if 0<=lx<cw and 0<=ly<ch:
                    px=int(zx0+lx*zscale); py=int(zy0+ly*zscale)
                    zoom_r=max(5,int(dot_r*zscale*0.8))
                    cv2.circle(zoom,(px,py),zoom_r+2,(0,0,0),-1,cv2.LINE_AA)
                    cv2.circle(zoom,(px,py),zoom_r,colors[i],-1,cv2.LINE_AA)
        # Label header on zoom panel
        cv2.rectangle(zoom,(0,0),(panel_w,42),(30,30,30),-1)
        cv2.putText(zoom,f"ZOOM  •  {focus}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(220,220,220),2,cv2.LINE_AA)
        writer.write(np.hstack([frame,zoom]))
    cap.release();writer.release()
    summary={name:{"mean_likelihood":float(np.nanmean(pred[:,i,2])),"min_likelihood":float(np.nanmin(pred[:,i,2]))} for i,name in enumerate(labels)}
    (qc/"metadata.json").write_text(json.dumps({"video":str(video),"model":str(model),"focus_label":focus,"duration_seconds":len(indices)/fps,"zoom_context":context,"likelihood":summary},indent=2))
    print(f"QC complete: {qc/'preview.mp4'}")
main()
