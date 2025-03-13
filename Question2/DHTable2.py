import numpy as np
import math

class DHTable:

    # Constant definitions
    TABLE_ROWS = 4
    TABLE_COLUMNS = 4
    D = 0
    THETA = 1
    A = 2
    ALPHA = 3

    L2 = 600
    L3 = 450

    # Initialise DH table 
    def __init__(self, joint_angles, s1, s4):
        self.joint_angles = joint_angles
        self.DH_Table = np.zeros((6, 4))

        d_i = [100, 0, 0, self.L3+s4]
        a_i = [0, self.L2, 0, 0]
        alpha_i = [math.pi/2,0,-(math.pi/2),0]

        for row in range(self.TABLE_ROWS):
            self.DH_Table[row][self.D] = d_i[row]
            self.DH_Table[row][self.THETA] = self.joint_angles[row]
            self.DH_Table[row][self.A] = a_i[row]
            self.DH_Table[row][self.ALPHA] = alpha_i[row]

    #Helper function to get DH parameters
    def _get_DH_parameters(self, frame_num):
        # Extract parameters
        theta = self.DH_Table[frame_num][self.THETA]
        d = self.DH_Table[frame_num][self.D]
        a = self.DH_Table[frame_num][self.A]
        alpha = self.DH_Table[frame_num][self.ALPHA]

        # Compute trigonometric values
        c_t = math.cos(theta)
        s_t = math.sin(theta)
        c_a = math.cos(alpha)
        s_a = math.sin(alpha)

        return c_t, s_t, c_a, s_a, d, a

    # Construct homogeneous transform matrix (either standard or modified)
    def constructHT(self, frame_num, standard=True):
        c_t, s_t, c_a, s_a, d_i, a_i = self._get_DH_parameters(frame_num)

        if standard:
            SHT = np.array([
                [c_t, -s_t * c_a, s_t * s_a, a_i * c_t],
                [s_t, c_t * c_a, -c_t * s_a, a_i * s_t],
                [0, s_a, c_a, d_i],
                [0, 0, 0, 1]
            ])
            return SHT
        else:
            MHT = np.array([
                [c_t, -s_t, 0, a_i],
                [s_t * c_a, c_t * c_a, -s_a, -s_a * d_i],
                [s_t * s_a, c_t * s_a, c_a, c_a * d_i],
                [0, 0, 0, 1]
            ])
            return MHT

