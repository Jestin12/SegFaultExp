from DHTable2 import DHTable
from ArmKinematics2 import ArmKinematics
from ArmVisualiser2 import ArmVisualiser
import math
from workspacePlotter import WorkspacePlotter
import numpy as np

THETA_2 = math.radians(30)
THETA_3 = math.radians(60) - math.radians(90)

S1 = 500
S4 = 50

joint_angles = [0,THETA_2,THETA_3,0]

table1 = DHTable(joint_angles, S1, S4)

kinematics = ArmKinematics(table1)

transforms =  kinematics.getAllJointGlobPose()

#print("Joint Positions: \n", transforms.round(2), "\n")

np.set_printoptions(suppress=True)

print("End Effector Position: \n",kinematics.endeffectorPosition().round(2), "\n")

kinematics.checkCorrectness()

# plotter = WorkspacePlotter(50, 100, 10, 30, -4*np.pi/5, np.pi/2, -4*np.pi/3, 4*np.pi/3)
# # plotter = WorkspacePlotter(50, 100, 10, 30, 0, 2*np.pi, 0, 2*np.pi)
# plotter.plotWorkspace()

# visualiser = ArmVisualiser()
# visualiser.PlotUR5e(transforms)
# visualiser.PlotObstacle([400,0], 300)

# visualiser.Show()


