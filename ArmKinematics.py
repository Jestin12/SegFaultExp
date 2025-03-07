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

        for frame_num in range(6):
            finalTransform = finalTransform@self.DHTable.constructHT(self.DHTable.DH_Table, frame_num)

        return finalTransform, finalTransform[0,3], finalTransform[1,3], finalTransform[2,3]
    