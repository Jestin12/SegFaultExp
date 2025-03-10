from DHTable import DHTable
from ArmKinematics import ArmKinematics
from ArmVisualiser import ArmVisualiser
from claude import Arrow3D
import math
from claude import visualize_transforms
import matplotlib.pyplot as plt

joint_angles = [0,0,0,0,0,0]

table1 = DHTable(joint_angles)

kinematics = ArmKinematics(table1)

transforms =  kinematics.getAllJointPose()

print("End Effector Position: \n",kinematics.endeffectorPosition().round(2), "\n")

kinematics.checkCorrectness()

fig, ax = visualize_transforms(transforms, scale=1.0)
plt.show()