from cDHTable import cDHTable
from cArmKinematics import cArmKinematics
from cArmVisualiser import cArmVisualiser
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np


# *************************** main.py ***************************************

# Filename:       main.py
# Author:         Alfie

# Description:  This file contains the main code of the robotic arm simulator,
#               it produces a DH Table to calculate the frame transformation matrices
#               by instantiating the cDHTable class. The code checks for singularities
#               and returns an end effector position by instantiating the cArmKinematics 
#               class.
#               The code plots the frames of the robot limb on a 3D plot through the 
#               cArmVisualiser class.
#               The code uses Standard input to recursively calculate transformation matrices
#               and plot the frames of the robot limb.
#
# Dependencies: numpy   cDHTable.py     cArmKinematics.py     cArmVisualiser.py

# ************************************************************************************



#   Triggering Standard Input for joint angles

#   Visualises a plot described by the Standard input, then 
#   generates a new plot when the old plot's window is closed
#   using new Standard input values

while(1):
    print("Enter the angles for the robot arm in degrees\n")
    Base = float(input("Base angle: "))
    Shoulder = float(input("Shoulder angle: "))
    Elbow = float(input("Elbow angle: "))
    Wrist1 = float(input("Wrist1 angle: "))
    Wrist2 = float(input("Wrist2 angle: "))
    Wrist3 = float(input("Wrist3 angle: "))

    JointAngles = [np.radians(Base), 
                   np.radians(Shoulder), 
                   np.radians(Elbow), 
                   np.radians(Wrist1), 
                   np.radians(Wrist2), 
                   np.radians(Wrist3)]

    DHTable = cDHTable(JointAngles)
    Kinematics = cArmKinematics(DHTable)
    Transforms = Kinematics.mGetAllJointGlobPose()
    Kinematics.mCheckCorrectness()

    # Print("Joint Positions: \n", Transforms.round(2), "\n")
    print("End Effector Position: \n",Kinematics.mEndEffectorPosition().round(2), "\n")

    #   Extracts the rotation matrix of the last transform matrix in Transforms, returning
    #   the transform matrix from frame 0 to the end effector frame
    RotationMatrix = Transforms[-1][:3, :3]


    #   Calculating the roll, pitch and yaw of the end effector relative to the global frame
    # Yaw (Rotation about Z-axis)
    Rz = np.arctan2(RotationMatrix[1, 0], RotationMatrix[0, 0])  

    # Pitch (Rotation about Y-axis)
    Ry = np.arctan2(-RotationMatrix[2, 0], np.sqrt(RotationMatrix[2, 1]**2 + RotationMatrix[2, 2]**2))  

    # Roll (Rotation about X-axis)
    Rx = np.arctan2(RotationMatrix[2, 1], RotationMatrix[2, 2])

    #Convert euler angles to rotation vector
    r = R.from_euler('ZYX', (Rz, Ry, Rx))
    RotVec = r.as_rotvec()

    #Print the Euler angles
    print("Roll (Rx): ", round(Rx, 3))
    print("Pitch (Ry): ", round(Ry, 3))
    print("Yaw (Rz): ", round(Rz, 3))

    #Print Rotation Vector
    print("Rx: ", round(RotVec[0], 3))
    print("Ry: ", round(RotVec[1], 3))
    print("Rz: ", round(RotVec[2], 3))

    #Instantiate the visualiser and plot the frames
    Visualiser = cArmVisualiser()
    Visualiser.mPlotUR5e(Transforms)
