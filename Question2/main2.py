from DHTable2 import DHTable
from ArmKinematics2 import ArmKinematics
from ArmVisualiser2 import ArmVisualiser
import math

THETA_1 = math.radians(30)
THETA_2 = math.radians(60) - math.radians(90)

joint_angles = [0,THETA_1,THETA_2,0]

table1 = DHTable(joint_angles, 100, 100)


print(table1.constructHT(0))
print(table1.constructHT(1))
print(table1.constructHT(2))
print(table1.constructHT(3))


kinematics = ArmKinematics(table1)

transforms =  kinematics.getAllJointGlobPose()

#print("Joint Positions: \n", transforms.round(2), "\n")

print("End Effector Position: \n",kinematics.endeffectorPosition().round(2), "\n")

kinematics.checkCorrectness()

visualiser = ArmVisualiser()
visualiser.PlotUR5e(transforms)


