"""Create a short full-frame plus focused Facemap QC preview."""
import json
from pathlib import Path
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
def main():
 if not C.get("qc_enabled",True):return
 results=R/"results";qc=results/"qc";qc.mkdir(parents=True,exist_ok=True)
 model=results/(C["model_name"]+".pt");info=results/(C["model_name"]+"_model_info.json")
 if not model.is_file() or not info.is_file():raise RuntimeError("QC needs the trained model and model-info JSON")
 labels=json.loads(info.read_text()).get("labels",[])
 if not labels:raise RuntimeError("QC cannot find the trained keypoint names")
 video=rp("runtime_qc_video","qc_video") if C.get("qc_video") or C.get("runtime_qc_video") else rp("runtime_initialization_video","initialization_video")
 cap=cv2.VideoCapture(str(video))
 if not cap.isOpened():raise RuntimeError(f"QC video cannot be opened: {video}")
 fps=float(cap.get(cv2.CAP_PROP_FPS) or 30);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
 count=min(total,max(1,int(float(C.get("qc_duration_seconds",60))*fps)));start=max(0,(total-count)//2);indices=np.arange(start,start+count,dtype=np.int64)
 pose=Pose(filenames=[[str(video)]],bbox=[],bbox_set=False,resize=False,add_padding=False,gui=None,GUIobject=None,net=None,model_name=str(model));pose.bodyparts=list(labels);pose.batch_size=8;pose.pose_prediction_setup()
 pred=normalize(pose.predict_landmarks(video_id=0,frame_ind=indices),len(indices),len(labels))
 try:
  import h5py
  with h5py.File(qc/"predictions.h5","w") as f:
   f.attrs["video_path"]=str(video);f.attrs["model_path"]=str(model);f.create_dataset("frame_index",data=indices,compression="gzip");g=f.create_group("Facemap")
   for i,name in enumerate(labels):
    q=g.create_group(name);q.create_dataset("x",data=pred[:,i,0],compression="gzip");q.create_dataset("y",data=pred[:,i,1],compression="gzip");q.create_dataset("likelihood",data=pred[:,i,2],compression="gzip")
 except ImportError:
  np.savez_compressed(qc/"predictions.npz",frame_index=indices,predictions=pred,labels=np.asarray(labels))
 focus=str(C.get("qc_focus_label","")).strip()
 if focus not in labels:
  focus=next((x for word in ("pupil","eye","tongue","nose","mouth") for x in labels if word in x.lower()),labels[0])
 fi=labels.index(focus);context=float(C.get("qc_zoom_context",1.0));cw=max(80,int(240*context));ch=max(60,int(180*context));threshold=0.2
 colors=[(66,153,225),(72,187,120),(237,137,54),(159,122,234),(245,101,101),(56,178,172)]
 cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start);panel_w=max(360,min(640,w));writer=cv2.VideoWriter(str(qc/"preview.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w+panel_w,h))
 last=np.array([w/2,h/2],float)
 for j in range(len(indices)):
  ok,frame=cap.read()
  if not ok:break
  row=pred[j]
  x,y,lik=row[fi]
  if np.isfinite(x) and np.isfinite(y) and lik>=threshold:last=.8*last+.2*np.array([x,y])
  for i,(x,y,lik) in enumerate(row):
   if np.isfinite(x) and np.isfinite(y) and lik>=threshold:cv2.circle(frame,(int(x),int(y)),3,colors[i%len(colors)],-1,cv2.LINE_AA)
  x1=max(0,min(w-cw,int(last[0]-cw/2)));y1=max(0,min(h-ch,int(last[1]-ch/2)));crop=frame[y1:y1+ch,x1:x1+cw]
  zoom=cv2.resize(crop,(panel_w,h),interpolation=cv2.INTER_CUBIC);cv2.putText(zoom,f"Focus: {focus}",(18,32),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA);writer.write(np.hstack([frame,zoom]))
 cap.release();writer.release()
 summary={name:{"mean_likelihood":float(np.nanmean(pred[:,i,2])),"min_likelihood":float(np.nanmin(pred[:,i,2]))} for i,name in enumerate(labels)}
 (qc/"metadata.json").write_text(json.dumps({"video":str(video),"model":str(model),"focus_label":focus,"duration_seconds":len(indices)/fps,"zoom_context":context,"likelihood":summary},indent=2))
 print(f"QC complete: {qc/'preview.mp4'}")
main()
