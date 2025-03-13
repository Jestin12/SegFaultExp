from DHTable import DHTable
from ArmKinematics import ArmKinematics
from ArmVisualiser import ArmVisualiser
import math
import numpy as np

# joint_angles = [0,-(math.pi/2),0,-(math.pi/2),0,0]

# table1 = DHTable(joint_angles)


# print(table1.constructHT(0))
# print(table1.constructHT(1))
# print(table1.constructHT(2))
# print(table1.constructHT(3))
# print(table1.constructHT(4))
# print(table1.constructHT(5))


# Kinematics = ArmKinematics(table1)

# Transforms =  Kinematics.getAllJointGlobPose()

# #print("Joint Positions: \n", transforms.round(2), "\n")

# print("End Effector Position: \n",Kinematics.endeffectorPosition().round(2), "\n")

# Kinematics.checkCorrectness()

# Visualiser = ArmVisualiser()
# Visualiser.PlotUR5e(Transforms)


while(1):
    print("Enter the angles for the robot arm in degrees\n")
    base = int(input("base angle: "))
    shoulder = int(input("Shoulder angle: "))
    elbow = int(input("elbow angle: "))
    wrist1 = int(input("wrist1 angle: "))
    wrist2 = int(input("wrist2 angle: "))
    wrist3 = int(input("wrist3 angle: "))

    JointAngles = [base * (math.pi / 180), 
                   shoulder * (math.pi / 180), 
                   elbow * (math.pi / 180), 
                   wrist1 * (math.pi / 180), 
                   wrist2 * (math.pi / 180), 
                   wrist3 * (math.pi / 180)]

    DhTable = DHTable(JointAngles)
    Kinematics = ArmKinematics(DhTable)
    Transforms = Kinematics.getAllJointGlobPose()

    # print("Joint Positions: \n", Transforms.round(2), "\n")
    print("End Effector Position: \n",Kinematics.endeffectorPosition().round(2), "\n")

    RotationMatrix = Transforms[-1][:3, :3]

    # Yaw (Rotation about Z-axis)
    Rz = np.arctan2(RotationMatrix[1, 0], RotationMatrix[0, 0])  

    # Pitch (Rotation about Y-axis)
    Ry = np.arctan2(-RotationMatrix[2, 0], np.sqrt(RotationMatrix[2, 1]**2 + RotationMatrix[2, 2]**2))  

    # Roll (Rotation about X-axis)
    Rx = np.arctan2(RotationMatrix[2, 1], RotationMatrix[2, 2])  


    print("Roll: ", Rx)
    print("Pitch: ", Ry)
    print("Yaw: ", Rz)


    Visualiser = ArmVisualiser()
    Visualiser.PlotUR5e(Transforms)
