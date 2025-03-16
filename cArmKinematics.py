import numpy as np

# *************************** cArmKinematics.py ***************************************

# Filename:       cArmVisualiser.py
# Author:         Alfie

# Description:  This file defines the cArmKinematics class which finds the 
#               end effector position of the robot arm as well as the homogeneous 
#               transforms from the origin to each joint. It also checks for singularities
#               using the Jacobian matrix.    

# Dependencies: numpy

# ************************************************************************************


class cArmKinematics:

    TRANSFORM_DIM = 4
    WORKSPACE_DIM = 3

    # Class constructor 
    def __init__(self, DHTable, NumJoints):
        self.DHTable = DHTable
        self.NumJoints = NumJoints


    #Check for singularites using Jacobian matrices
    def mCheckCorrectness(self):
        #Initialise the position and rotation vectors
        PositionVector = np.array([np.zeros(self.WORKSPACE_DIM) for _ in range(self.NumJoints)])
        RotationVector = np.array([np.eye(self.WORKSPACE_DIM) for _ in range(self.NumJoints)])
        LinearVelocity = np.array([np.zeros(self.WORKSPACE_DIM) for _ in range(self.NumJoints)])
        AngularVelocity = np.array([np.zeros(self.WORKSPACE_DIM) for _ in range(self.NumJoints)])

        #Obtain the position, rotation, linear velocity and angular velocity vectors from the joint pose
        for FrameNum in range(self.NumJoints):
            PositionVector[FrameNum] = self.JointPoseGlob[FrameNum][:self.WORKSPACE_DIM, -1]
            RotationVector[FrameNum] = self.JointPoseGlob[FrameNum][:self.WORKSPACE_DIM, :self.WORKSPACE_DIM]
            AngularVelocity[FrameNum] = RotationVector[FrameNum][:, -1]
            LinearVelocity[FrameNum] = np.cross(AngularVelocity[FrameNum], PositionVector[-1] - PositionVector[FrameNum])

        J = np.zeros((self.WORKSPACE_DIM + self.WORKSPACE_DIM, self.NumJoints))

        for FrameNum in range(self.NumJoints):
            J[:self.WORKSPACE_DIM, FrameNum] = LinearVelocity[FrameNum]
            J[self.WORKSPACE_DIM:, FrameNum] = AngularVelocity[FrameNum]


        #Check for singularites
        Singular = np.linalg.matrix_rank(J) < min(J.shape)

        if Singular:
            print("Singularities detected")

        else:
            print("No singularities detected")


    def mEndeffectorPosition(self):
        return self.JointPoseGlob[-1][:self.WORKSPACE_DIM, -1]


    # Determine the final end effector position 
    def mGetAllJointGlobPose(self):
        self.JointPoseGlob = np.array([np.eye(self.TRANSFORM_DIM) for _ in range(self.NumJoints)])
     
        for FrameNum in range(self.NumJoints):
            self.JointPoseGlob[FrameNum] = self.JointPoseGlob[FrameNum-1]@self.DHTable.mConstructHT(FrameNum, Standard=True)

        return self.JointPoseGlob


    
    