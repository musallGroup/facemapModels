"""Headless Facemap 1.0.8 adapter for LabelForge."""
import csv,hashlib,json,math,random
from pathlib import Path
import cv2,numpy as np,torch,torch.nn as nn
from facemap.pose.pose import Pose
import facemap.pose.pose as pm
from facemap.pose import datasets,model_training
pm.datasets=datasets;pm.model_training=model_training
R=Path(__file__).resolve().parent
C=globals().get("TRAINING_MANIFEST") or json.loads((R/"training_manifest.json").read_text())
def p(a,b):
 q=Path(C.get(a) or C.get(b,""));return q if q.is_absolute() else R/q
def sha(q):
 h=hashlib.sha256()
 with q.open("rb") as f:
  for x in iter(lambda:f.read(1048576),b""):h.update(x)
 return h.hexdigest()
def data(frames,csvpath):
 with csvpath.open("r",newline="",encoding="utf-8-sig") as f:
  r=csv.DictReader(f);fields=r.fieldnames or [];have=set(fields);labels=[x[:-2] for x in fields if x.endswith("_x") and x[:-2]+"_y" in have]
  if not labels:raise RuntimeError("labels.csv needs matching <label>_x and <label>_y columns")
  ims=[];points=[];used=[];skipped=[]
  for line,row in enumerate(r,2):
   if str(row.get("skipped","0")).strip()=="1":skipped.append([line,"skipped"]);continue
   fp=None
   for col in ("image","frame","filename","png_path"):
    v=str(row.get(col,"")).strip()
    if v:
     local=frames/Path(v).name;original=Path(v)
     if local.is_file():fp=local;break
     if original.is_file():fp=original;break
   im=cv2.imread(str(fp),0) if fp else None
   if im is None:skipped.append([line,"image unavailable"]);continue
   xy=np.full((len(labels),2),np.nan,np.float32)
   for i,label in enumerate(labels):
    xt=str(row.get(label+"_x","")).strip();yt=str(row.get(label+"_y","")).strip()
    if not xt or not yt:continue
    state=str(row.get(label+"_state","")).strip().lower();flag=str(row.get(label+"_visible","")).strip().lower();explicit=label+"_state" in row or label+"_visible" in row
    if explicit and state!="visible" and flag not in ("1","true","yes"):continue
    try:x,y=float(xt),float(yt)
    except ValueError:continue
    if math.isfinite(x) and math.isfinite(y):xy[i]=(x,y)
   if not np.isfinite(xy).all(1).any():skipped.append([line,"no keypoints"]);continue
   ims.append(im);points.append(xy);used.append(fp.name)
 if not ims:raise RuntimeError("No usable labelled frames found")
 if any(x.shape!=ims[0].shape for x in ims):raise RuntimeError("All training frames must have the same size")
 return np.stack(ims),np.asarray(points,np.float32),labels,used,skipped
def heads(net,n):
 for name in ("conv0","conv1","conv2"):
  old=getattr(net.Conv2_1x1,name);new=nn.Conv2d(old.in_channels,n,old.kernel_size,old.stride,old.padding,old.dilation,old.groups,old.bias is not None,old.padding_mode);nn.init.kaiming_normal_(new.weight,mode="fan_out",nonlinearity="relu")
  if new.bias is not None:nn.init.zeros_(new.bias)
  setattr(net.Conv2_1x1,name,new)
def state(x):
 if "state_dict" in x:return x["state_dict"]
 if "model_state_dict" in x:return x["model_state_dict"]
 return x
def main():
 seed=int(C.get("random_seed",42));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 frames=p("runtime_training_data","training_data");labels_csv=p("runtime_labels_or_config","labels_or_config");video=p("runtime_initialization_video","initialization_video");pv=C.get("runtime_parent_model") or C.get("parent_model","");parent=p("runtime_parent_model","parent_model") if pv else None
 images,keypoints,labels,used,skipped=data(frames,labels_csv);h,w=images.shape[1:3];bbox=[[0,h,0,w]]
 print(f"LabelForge Facemap: {len(used)} frames, {len(labels)} keypoints")
 pose=Pose(filenames=[[str(video)]],bbox=bbox,bbox_set=True,resize=True,add_padding=True,model_name=None);pose.load_model();heads(pose.net,len(labels));parent_hash=None
 if parent:parent_hash=sha(parent);pose.net.load_state_dict(state(torch.load(str(parent),map_location=pose.device)),strict=True)
 pose.net=pose.net.to(pose.device);pose.train(image_data=images,keypoints_data=keypoints,num_epochs=int(C.get("epochs",100)),batch_size=int(C.get("batch_size",1)),learning_rate=float(C.get("learning_rate",5e-5)),weight_decay=0.0,bbox=bbox)
 if parent and sha(parent)!=parent_hash:raise RuntimeError("Parent model changed during training")
 outdir=R/"results";outdir.mkdir(exist_ok=True);out=outdir/(C["model_name"]+".pt")
 if out.exists():raise FileExistsError(f"Result exists and is not overwritten: {out}")
 torch.save(pose.net.state_dict(),str(out));info={"model_name":C["model_name"],"facemap_version":"1.0.8","labels":labels,"frames_used":used,"frames_skipped":skipped,"parent_model":str(parent) if parent else None,"parent_sha256":parent_hash,"result_model":str(out.relative_to(R)),"device":str(pose.device),"bbox":bbox};(outdir/(C["model_name"]+"_model_info.json")).write_text(json.dumps(info,indent=2))
 print(f"Training complete: {out}")
main()
