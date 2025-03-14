from DHTable2 import DHTable
from ArmKinematics2 import ArmKinematics
from ArmVisualiser2 import ArmVisualiser
import math
from workspacePlotter import WorkspacePlotter
import numpy as np

THETA_1 = math.radians(89.18)
THETA_2 = math.radians(-94.53) - math.radians(90)

joint_angles = [0,-0.74,1.65,0]

table1 = DHTable(joint_angles, 500, 50)


print(table1.constructHT(0))
print(table1.constructHT(1))
print(table1.constructHT(2))
print(table1.constructHT(3))


kinematics = ArmKinematics(table1)

transforms =  kinematics.getAllJointGlobPose()

#print("Joint Positions: \n", transforms.round(2), "\n")

# print("End Effector Position: \n",kinematics.endeffectorPosition().round(2), "\n")

kinematics.checkCorrectness()

# plotter = WorkspacePlotter(50, 100, 10, 30, -4*np.pi/5, np.pi/2, -4*np.pi/3, 4*np.pi/3)
plotter = WorkspacePlotter(50, 100, 10, 30, 0, 2*np.pi, 0, 2*np.pi)
plotter.plotWorkspace()

# visualiser = ArmVisualiser()
# visualiser.PlotUR5e(transforms)

# visualiser.Show()


