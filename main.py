from DHTable import DHTable
from ArmKinematics import ArmKinematics
import math

joint_angles = [0,0,0,0,0,0]

table1 = DHTable(joint_angles)

kinematics = ArmKinematics(table1)

print("Transform Matrices: \n", kinematics.getAllJointPose().round(2), "\n")

print("End Effector Position: \n",kinematics.endeffectorPosition().round(2), "\n")

kinematics.checkCorrectness()
