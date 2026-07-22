import inspect
import facemap
from facemap.pose.pose import Pose

print("facemap module:", facemap)
print("Pose class:", Pose)

print("\nPose methods:")
for name in dir(Pose):
    if not name.startswith("_"):
        obj = getattr(Pose, name)
        if callable(obj):
            print(" -", name)

print("\nPose.__init__ signature:")
print(inspect.signature(Pose.__init__))

print("\nPose source file:")
print(inspect.getsourcefile(Pose))