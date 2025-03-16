from cDHTable import cDHTable
from cArmKinematics import cArmKinematics
from cArmVisualiser import cArmVisualiser
from cWorkspacePlotter import cWorkspacePlotter
from cInverseKinematics import cInverseKinematics

import math
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

# Suppress scientific notation
np.set_printoptions(suppress=True)

def PerformCalcs(JointAngles, S1, S4):
    DHTable = cDHTable(JointAngles, S1, S4)
    Kinematics = cArmKinematics(DHTable, len(JointAngles))
    Transforms =  Kinematics.mGetAllJointGlobPose()
    print("Joint Positions: \n", Transforms.round(2), "\n")
    print("End Effector Position: \n",Kinematics.mEndeffectorPosition().round(2), "\n")
    Kinematics.mCheckCorrectness()
    visualiser = cArmVisualiser()
    visualiser.mPlotUR5e(Transforms)
    if len(JointAngles) == 4:
        visualiser.PlotObstacle([400,0], 300)
    visualiser.Show()

while(1):
    print("Would you like to simulate a 4 DOF or 6 DOF manipualtor? (enter 4 or 6)\n")
    ManType = int(input("DOF: "))

    if ManType == 4:
        print("Would you like to perform forward kinematics or inverse kinematics? (enter F or I)")
        KinType = input("Kinematic Type: ")
        if KinType == "F":
            print("Enter your manipulator parameters: \n")
            S1 = float(input("S1: "))
            S4 = float(input("S4: "))
            THETA_2  = float(input("THETA_2: "))
            THETA_3  = float(input("THETA_3: "))
            JointAngles = [0,np.radians(THETA_2),np.radians(THETA_3) - math.radians(90),0]
            PerformCalcs(JointAngles, S1, S4)
        elif KinType == "I":
            xe = int(input("Enter x coordinate for end effector (float/int):"))
            ye = int(input("Enter y coordinate for end effector (float/int):"))
            S1 = int(input("Enter s1 length (float/int):"))
            S4 = int(input("Enter s4 length (float/int):"))
            IK = cInverseKinematics(xe, ye, S1, S4)
            solutions = IK.ComputeIK()
            for i in range(len(solutions)):
                print("Solution " + str(i + 1) + ":" + "[" + str(solutions[i][0].evalf(2)) + "," + str(solutions[i][1].evalf(2)) + "]" + "\n")

    elif ManType == 6:
        JointAngles = [0,0,0,0,0,0]
        S1 = 0
        S4 = 0
        print("Enter the angles for the robot arm in degrees (default = 0 degrees for all).\n")
        JointAngles[0] = np.radians(float(input("Base angle: ")))
        JointAngles[1] = np.radians(float(input("Shoulder angle: ")))
        JointAngles[2] = np.radians(float(input("Elbow angle: ")))
        JointAngles[3] = np.radians(float(input("Wrist1 angle: ")))
        JointAngles[4] = np.radians(float(input("Wrist2 angle: ")))
        JointAngles[5] = np.radians(float(input("Wrist3 angle: ")))

        PerformCalcs(JointAngles, S1, S4)

    else:
        print("Incompatible DOF.")

