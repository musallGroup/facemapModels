import inspect
from facemap.pose.pose import Pose

print(inspect.signature(Pose.train))
print("\n--- SOURCE ---\n")
print(inspect.getsource(Pose.train))