from DHTable2 import DHTable
from ArmKinematics2 import ArmKinematics
from ArmVisualiser2 import ArmVisualiser
import math
from workspacePlotter import WorkspacePlotter
import numpy as np

THETA_1 = math.radians(89.18)
THETA_2 = math.radians(-94.53) - math.radians(90)

joint_angles = [0,-1.137,1.227 - np.pi/2,0]

table1 = DHTable(joint_angles, 500, 50)

kinematics = ArmKinematics(table1)

transforms =  kinematics.getAllJointGlobPose()

#print("Joint Positions: \n", transforms.round(2), "\n")

print("End Effector Position: \n",kinematics.endeffectorPosition().round(2), "\n")

kinematics.checkCorrectness()

# plotter = WorkspacePlotter(50, 100, 10, 30, -4*np.pi/5, np.pi/2, -4*np.pi/3, 4*np.pi/3)
# plotter = WorkspacePlotter(50, 100, 10, 30, 0, 2*np.pi, 0, 2*np.pi)
# plotter.plotWorkspace()

visualiser = ArmVisualiser()
visualiser.PlotUR5e(transforms)

visualiser.Show()


