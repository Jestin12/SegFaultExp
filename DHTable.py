import numpy as np
import math

class DHTable:

    # Constant definitions
    TABLE_ROWS = 6
    TABLE_COLUMNS = 4
    D = 0
    THETA = 1
    A = 2
    ALPHA = 3

    # Initialise DH table 
    def __init__(self, joint_angles):
        self.joint_angles = joint_angles
        self.DH_Table = np.zeros((6, 4))

        d_i = [163, 138, 131, 127, 100, 100]
        a_i = [0, 425, 392, 0, 0, 0]
        alpha_i = [(math.pi)/2, -(math.pi), math.pi, -(math.pi), math.pi, 0]

        for row in range(self.TABLE_ROWS):
            self.DH_Table[row][self.D] = d_i[row]
            self.DH_Table[row][self.THETA] = self.joint_angles[row]
            self.DH_Table[row][self.A] = a_i[row]
            self.DH_Table[row][self.ALPHA] = alpha_i[row]

    # Construct homogeneous transform matrix
    def constructHT(self, DH_Table, frame_num):
        HT = np.matrix([[math.cos(DH_Table[frame_num][self.THETA]), -(math.sin(DH_Table[frame_num][self.THETA])*math.cos(DH_Table[frame_num][self.ALPHA])), (math.sin(DH_Table[frame_num][self.THETA])*math.sin(DH_Table[frame_num][self.ALPHA])), DH_Table[frame_num][self.A]*math.cos(DH_Table[frame_num][self.THETA])], 
                        [math.sin(DH_Table[frame_num][self.THETA]), (math.cos(DH_Table[frame_num][self.THETA])*math.cos(DH_Table[frame_num][self.ALPHA])), -(math.cos(DH_Table[frame_num][self.THETA])*math.sin(DH_Table[frame_num][self.ALPHA])), DH_Table[frame_num][self.A]*math.sin(DH_Table[frame_num][self.THETA])],
                        [0, math.sin(DH_Table[frame_num][self.ALPHA]), math.cos(DH_Table[frame_num][self.ALPHA]), DH_Table[frame_num][self.D]],
                        [0,0,0,1]])

        return HT
