from sympy import symbols, Eq, solve, sqrt, atan2, acos, cos, sin, pprint
from cArmVisualiser2 import cArmVisualiser
from cArmKinematics2 import cArmKinematics
import numpy as np

class cInverseKinematics:

    # Define symbolic variables
    theta_1, theta_2 = symbols('theta_1 theta_2')
    L2, L3 = 600, 450   # Link lengths

    def __init__(self, xe, ye, s1, s4):
        self.xe = xe
        self.ye = ye
        self.s1 = s1
        self.s4 = s4

    # Given values
    # x, y = 750, 0  # Target position of end effector
    # s1, s4 = 500, 50  # Offset values
    # L2, L3 = 600, 450 + s4  # Link lengths

    

    def ComputeIK(self):

        # Define inverse kinematics equations
        eq1 = Eq(self.xe, self.L2*cos(self.theta_1) + (self.L3 + self.s4)*cos(self.theta_1 + self.theta_2))
        eq2 = Eq(self.ye, self.s1 + self.L2*sin(self.theta_1) + (self.L3 + self.s4)*sin(self.theta_1 + self.theta_2))

        # Solve the system
        solution = solve((eq1, eq2), (self.theta_1, self.theta_2))
        # pprint(solution)

        return solution

        # for i in len(solution):
        #     theta_2_sol_1 = solution[0][0].evalf()
        #     theta_3_sol_1 = solution[0][1].evalf()

        #     theta_2_sol_2 = solution[1][0].evalf()
        #     theta_3_sol_2 = solution[1][1].evalf()

        # return theta_2_sol_1, theta_3_sol_1, theta_2_sol_2, theta_3_sol_2

    # print("Theta 1: ", theta_value_1)
    # print("Theta 2: ", theta_value_2)
    # print("Theta 3: ", theta_value_3)
    # print("Theta 4: ", theta_value_4)

