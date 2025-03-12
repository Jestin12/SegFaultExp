import numpy as np

class ArmKinematics:

    JOINTS = 6
    TRANSFORM_DIM = 4
    WORKSPACE_DIM = 3

    # Class constructor 
    def __init__(self, DHTable):
        self.DHTable = DHTable


    #Check for singularites using Jacobian matrices
    def checkCorrectness(self):
        #Initialise the position and rotation vectors
        positionVector = np.array([np.zeros(self.WORKSPACE_DIM) for _ in range(self.JOINTS)])
        rotationVector = np.array([np.eye(self.WORKSPACE_DIM) for _ in range(self.JOINTS)])
        linearVelocity = np.array([np.zeros(self.WORKSPACE_DIM) for _ in range(self.JOINTS)])
        angularVelocity = np.array([np.zeros(self.WORKSPACE_DIM) for _ in range(self.JOINTS)])

        #Obtain the position, rotation, linear velocity and angular velocity vectors from the joint pose
        for frame_num in range(self.JOINTS):
            positionVector[frame_num] = self.jointPoseGlob[frame_num][:self.WORKSPACE_DIM, -1]
            rotationVector[frame_num] = self.jointPoseGlob[frame_num][:self.WORKSPACE_DIM, :self.WORKSPACE_DIM]
            angularVelocity[frame_num] = rotationVector[frame_num][:, -1]
            linearVelocity[frame_num] = np.cross(angularVelocity[frame_num], positionVector[-1] - positionVector[frame_num])

        #Compute the Jacobian matrix
        J = np.zeros((6,6))
        J = np.vstack((np.transpose(linearVelocity), np.transpose(angularVelocity)))

        #Check for singularites
        if np.linalg.det(J) == 0:
            print("Singularities detected")
        else:
            print("No singularities detected")

    def endeffectorPosition(self):
        return self.jointPoseGlob[-1][:self.WORKSPACE_DIM, -1]


    # Determine the final end effector position 
    def getAllJointGlobPose(self):
        self.jointPoseGlob = np.array([np.eye(self.TRANSFORM_DIM) for _ in range(self.JOINTS)])

        for frame_num in range(self.JOINTS):
            self.jointPoseGlob[frame_num] = self.jointPoseGlob[frame_num-1]@self.DHTable.constructHT(frame_num)

        return self.jointPoseGlob


    
    