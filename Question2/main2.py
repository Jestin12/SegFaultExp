from cDHTable2 import cDHTable
from cArmKinematics2 import cArmKinematics
from cArmVisualiser2 import cArmVisualiser
import math
from cWorkspacePlotter import cWorkspacePlotter
import numpy as np

# *************************** cWorkspacePlotter.py ***************************************

# Filename:         main2.py
# Author:           Alfie

# Description:      The file is the main code of the Question2 robot arm simulator,
#                   it plots the robot's frames in a 3D plot using the cArmVisualiser
#                   class instantiation. cArmVisualiser receives the transformation matrices
#                   it requires from the cDHTable class instantiation, which receives the 
#                   required joint angles and extrution lengths.
#
#                   cWorkspacePlotter plots the possible positions of the end effector of
#                   the robot arm by obeying the joint limits provided in its constructor
#
#                   ** The robot arm is plotted in a 3D plot but only exists in the 2D plane
#                   X-Z.
#
# Dependencies:     cWorkspacePlotter.py   numpy   ArmKinematics2.py  DHTable2.py
#                   ArmVisualiser2.py       math

# ************************************************************************************


THETA_1 = math.radians(30)
THETA_2 = math.radians(60) - math.radians(90)

JointAngles = [0,THETA_1,THETA_2,0]

Table1 = cDHTable(JointAngles, 500, 50)


print(Table1.mConstructHT(0))
print(Table1.mConstructHT(1))
print(Table1.mConstructHT(2))
print(Table1.mConstructHT(3))


Kinematics = cArmKinematics(Table1)

Transforms =  Kinematics.mGetAllJointGlobPose()

print("Joint Positions: \n", Transforms.round(2), "\n")

print("End Effector Position: \n",Kinematics.mEndeffectorPosition().round(2), "\n")

Kinematics.mCheckCorrectness()

print("Takes about 20 seconds to render the workspace plot")
print("Close the workspace plot to view the frames plot")


# Plotter = cWorkspacePlotter(50, 100, 10, 30, -4*np.pi/5, np.pi/2, -4*np.pi/3, 4*np.pi/3)
# Plotter.mPlotWorkspace()

# visualiser = cArmVisualiser()
# visualiser.mPlotUR5e(Transforms)

# visualiser.Show()


