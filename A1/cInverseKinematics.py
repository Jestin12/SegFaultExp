from sympy import symbols, Eq, solve, sqrt, atan2, acos, cos, sin, pprint
from cArmVisualiser import cArmVisualiser
from cArmKinematics import cArmKinematics
import numpy as np

# *************************** cInverseKinematics.py ***************************************

# Filename:         cInverseKinematics.py
# Author:           Alfie

# Description:      The file defines the cInverseKinematics class which solves for the 
#                   unknown joint angles Theta1 and Theta2 of the 4 degrees of freedom
#                   robotic arm, given the required end effector position defined by:
#                   (Xe, Ye) and the extent of extrusion of the prismatic joints defined
#                   by S1 and S4.
#
#                   The method mComputeIK attempts to solve the inverse kinematic equations
#                   and returns a list of possible Theta1 and Theta2 angles to satisfy the
#                   end effector position (Xe, Ye) given S1 and S4.

# Dependencies:     sympy   cArmVisualiser  cArmKinematics  numpy

# ************************************************************************************

class cInverseKinematics:

    # Define symbolic variables
    Theta1, Theta2 = symbols('Theta1 Theta2')
    L2, L3 = 600, 450   # Link lengths

    def __init__(self, Xe, Ye, S1, S4):
        self.Xe = Xe
        self.Ye = Ye
        self.S1 = S1
        self.S4 = S4 

    def mComputeIK(self):

        # Define inverse kinematics equations
        Eq1 = Eq(self.Xe, self.L2*cos(self.Theta1) + (self.L3 + self.S4)*cos(self.Theta1 + self.Theta2))
        Eq2 = Eq(self.Ye, self.S1 + self.L2*sin(self.Theta1) + (self.L3 + self.S4)*sin(self.Theta1 + self.Theta2))

        # Solve the system
        Solution = solve((Eq1, Eq1), (self.Theta1, self.Theta2))
        
        #   Returns a list of sets [{Theta1, Theta2}, {Theta1, Theta2}, ...]
        #   Returns an empty list [] if no solutions are found
        return Solution

