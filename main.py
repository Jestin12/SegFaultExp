from cDHTable import cDHTable
from cArmKinematics import cArmKinematics
from cArmVisualiser import cArmVisualiser
from cWorkspacePlotter import cWorkspacePlotter
from cInverseKinematics import cInverseKinematics

import math
import numpy as np


# *************************** cWorkspacePlotter.py ***************************************

# Filename:         main.py
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

<<<<<<< Updated upstream
# Perform main calculations and operations
=======
#   Performs forward kinematics calculations for the given joint parameters and plots
#   the resultant frames on a 3D figure
>>>>>>> Stashed changes
def PerformCalcs(JointAngles, S1, S4):
    DHTable = cDHTable(JointAngles, S1, S4)
    Kinematics = cArmKinematics(DHTable, len(JointAngles))
    Transforms =  Kinematics.mGetAllJointGlobPose()

    print("Joint Positions: \n", Transforms.round(2), "\n")
    print("End Effector Position: \n",Kinematics.mEndeffectorPosition().round(2), "\n")

<<<<<<< Updated upstream
#Prompt user for input
=======
    Kinematics.mCheckCorrectness()
    Visualiser = cArmVisualiser()
    Visualiser.mPlotUR5e(Transforms)

    if len(JointAngles) == 4:
        Visualiser.PlotObstacle([400,0], 300)

    Visualiser.Show()

#   Infinite loop used to interact with the user to select the desired calculation
#   for the desired manipulator system
>>>>>>> Stashed changes
while(1):
    print("Would you like to simulate a 4 DOF or 6 DOF manipualtor? (enter 4 or 6)\n")

    ManType = int(input("DOF: "))

    #   For Question 2 manipulator
    if ManType == 4:
        print("Would you like to perform forward kinematics or inverse kinematics? (enter F or I)")
        KinType = input("Kinematic Type: ")

        #   Performs forward kinematics and plots given joint parameters
        if KinType == "F":

            print("Should I plot the workspace or the frame transformations")
            PlotSpace = input("W for workspace / T for frames: ")

            if PlotSpace == "T":
                print("Enter your manipulator parameters: \n")

                S1 = float(input("S1: "))
                S4 = float(input("S4: "))
                THETA_2  = float(input("THETA_2: "))
                THETA_3  = float(input("THETA_3: "))
                JointAngles = [0,np.radians(THETA_2),np.radians(THETA_3) - math.radians(90),0]
                PerformCalcs(JointAngles, S1, S4)

            elif PlotSpace == "W":

                Restrictions = input("Should I include joint restrictions (Y for yes / N for no): ")

                if Restrictions == "Y":
                    S1Lowlim = float(input("S1 lower limit: "))
                    S1Uplim = float(input("S1 upper limit: "))

                    S4LowLim = float(input("S4 lower limit: "))
                    S4UpLim = float(input("S4 upper limit: "))

                    Theta2LowLim = np.radians(float(input("Theta2 lower Limit: ")))
                    Theta2UpLim = np.radians(float(input("Theta2 upper Limit: ")))
                    
                    Theta3LowLim = np.radians(float(input("Theta3 lower Limit: ")))
                    Theta3UpLim = np.radians(float(input("Theta3 upper Limit: ")))

                    Man4Workspace = cWorkspacePlotter(S1Lowlim, S1Uplim, S4LowLim, S4UpLim, Theta2LowLim, Theta2UpLim, Theta3LowLim, Theta3UpLim)
                    Man4Workspace.mPlotWorkspace()
                
                elif Restrictions == "N":
                    Man4Workspace = cWorkspacePlotter()
                    Man4Workspace.mPlotWorkspace()




        elif KinType == "I":
            Xe = int(input("Enter x coordinate for end effector (float/int):"))
            Ye = int(input("Enter y coordinate for end effector (float/int):"))
            S1 = int(input("Enter S1 length (float/int):"))
            S4 = int(input("Enter S4 length (float/int):"))
            IK = cInverseKinematics(Xe, Ye, S1, S4)
            Solutions = IK.mComputeIK()

            for i in range(len(Solutions)):
                print("Solution " + str(i + 1) + ":" + "[" + str(Solutions[i][0].evalf(2)) + "," + str(Solutions[i][1].evalf(2)) + "]" + "\n")

    #   For Question 1 manipulator
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

