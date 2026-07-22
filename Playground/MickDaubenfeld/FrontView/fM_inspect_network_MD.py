from pathlib import Path

from facemap.pose.pose import Pose

PROJECT_ROOT = Path(r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon")

TRAINING_VIDEO = PROJECT_ROOT / "Training_Data" / "Training_Videos" / "cam0_frontview_training_mix.avi"

pose = Pose(
    filenames=[[str(TRAINING_VIDEO)]],
    model_name=None,
)

pose.load_model()

print("\n==============================")
print("NETWORK")
print("==============================\n")

print(pose.net)

print("\n==============================")
print("NAMED MODULES")
print("==============================\n")

for name, module in pose.net.named_modules():
    print(name, "-->", module.__class__.__name__)

print("\n==============================")
print("PARAMETER SHAPES")
print("==============================\n")

for name, param in pose.net.named_parameters():
    print(name, tuple(param.shape))