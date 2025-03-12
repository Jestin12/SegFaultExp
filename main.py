from DHTable import DHTable
from ArmKinematics import ArmKinematics
from ArmVisualiser import ArmVisualiser
import math

joint_angles = [0,0,0,0,0,0]

table1 = DHTable(joint_angles)

kinematics = ArmKinematics(table1)

transforms =  kinematics.getAllJointGlobPose()

print("Joint Positions: \n", transforms.round(2), "\n")

print("End Effector Position: \n",kinematics.endeffectorPosition().round(2), "\n")

kinematics.checkCorrectness()

visualiser = ArmVisualiser()
visualiser.PlotUR5e(transforms)


