import numpy as np
from sympy import Matrix

class ArmKinematics:

    # Class constructor 
    def __init__(self, DHTable):
        self.DHTable = DHTable


    # def checkCorrectness(self, ):


    # Determine the final end effector position 
    def endEffectorPose(self):
        finalTransform = np.eye(4)
        transforms = []

        for frame_num in range(6):

            print(finalTransform)
            finalTransform = finalTransform@self.DHTable.constructHT(self.DHTable.DH_Table, frame_num)

            # I put this after because i assume we don't want the identity matrix as on of the 
            # transform matrices
            transforms.append(finalTransform)

        return finalTransform, transforms, finalTransform[0,3], finalTransform[1,3], finalTransform[2,3]
    