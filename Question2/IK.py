from sympy import symbols, Eq, solve, sqrt, atan2, acos, cos, sin, pprint
from ArmVisualiser2 import ArmVisualiser
from ArmKinematics2 import ArmKinematics
import numpy as np

# Define symbolic variables
theta_1, theta_2 = symbols('theta_1 theta_2')

# Given values
x, y = 750, 0  # Target position of end effector
s1, s4 = 500, 50  # Offset values
L2, L3 = 600, 450 + s4  # Link lengths

# Define inverse kinematics equations
eq1 = Eq(x, L2*cos(theta_1) + L3*cos(theta_1 + theta_2))
eq2 = Eq(y, s1 + L2*sin(theta_1) + L3*sin(theta_1 + theta_2))

# Solve the system
solution = solve((eq1, eq2), (theta_1, theta_2))
pprint(solution)

theta_value_1 = solution[0][0].evalf()
theta_value_2 = solution[0][1].evalf()

theta_value_3 = solution[1][0].evalf()
theta_value_4 = solution[1][1].evalf()

print("Theta 1: ", theta_value_1)
print("Theta 2: ", theta_value_2)
print("Theta 3: ", theta_value_3)
print("Theta 4: ", theta_value_4)

