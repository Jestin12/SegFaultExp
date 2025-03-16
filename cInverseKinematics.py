from sympy import symbols, Eq, solve, sqrt, atan2, acos, cos, sin, pprint
from cArmVisualiser import cArmVisualiser
from cArmKinematics import cArmKinematics
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

    def ComputeIK(self):

        # Define inverse kinematics equations
        eq1 = Eq(self.xe, self.L2*cos(self.theta_1) + (self.L3 + self.s4)*cos(self.theta_1 + self.theta_2))
        eq2 = Eq(self.ye, self.s1 + self.L2*sin(self.theta_1) + (self.L3 + self.s4)*sin(self.theta_1 + self.theta_2))

        # Solve the system
        solution = solve((eq1, eq2), (self.theta_1, self.theta_2))

        return solution

