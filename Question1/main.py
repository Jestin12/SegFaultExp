from DHTable import DHTable
from ArmKinematics import ArmKinematics
from ArmVisualiser import ArmVisualiser
import numpy as np

# Triggering Standard Input for joint angles 
while(1):
    print("Enter the angles for the robot arm in degrees\n")
    base = int(input("base angle: "))
    shoulder = int(input("Shoulder angle: "))
    elbow = int(input("elbow angle: "))
    wrist1 = int(input("wrist1 angle: "))
    wrist2 = int(input("wrist2 angle: "))
    wrist3 = int(input("wrist3 angle: "))

    JointAngles = [np.radians(base), 
                   np.radians(shoulder), 
                   np.radians(elbow), 
                   np.radians(wrist1), 
                   np.radians(wrist2), 
                   np.radians(wrist3)]

    DhTable = DHTable(JointAngles)
    Kinematics = ArmKinematics(DhTable)
    Transforms = Kinematics.getAllJointGlobPose()
    Kinematics.checkCorrectness()

    # Print("Joint Positions: \n", Transforms.round(2), "\n")
    print("End Effector Position: \n",Kinematics.endeffectorPosition().round(2), "\n")

    RotationMatrix = Transforms[-1][:3, :3]

    # Yaw (Rotation about Z-axis)
    Rz = np.arctan2(RotationMatrix[1, 0], RotationMatrix[0, 0])  

    # Pitch (Rotation about Y-axis)
    Ry = np.arctan2(-RotationMatrix[2, 0], np.sqrt(RotationMatrix[2, 1]**2 + RotationMatrix[2, 2]**2))  

    # Roll (Rotation about X-axis)
    Rx = np.arctan2(RotationMatrix[2, 1], RotationMatrix[2, 2])  


    #Print the Euler angles
    print("Roll: ", round(Rx, 3))
    print("Pitch: ", round(Ry, 3))
    print("Yaw: ", round(Rz, 3))

    #Instantiate the visualiser and plot the frames
    Visualiser = ArmVisualiser()
    Visualiser.PlotUR5e(Transforms)
