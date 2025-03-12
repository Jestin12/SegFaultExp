from DHTable import DHTable
from ArmKinematics import ArmKinematics
from ArmVisualiser import ArmVisualiser
import math

joint_angles = [0, 0, 0, 0, 0, 0]

table1 = DHTable(joint_angles)

kinematics = ArmKinematics(table1)

mat, transforms, x,y,z = kinematics.endEffectorPose()

print(x,y,z)

visualiser = ArmVisualiser()
visualiser.PlotUR5e(transforms)