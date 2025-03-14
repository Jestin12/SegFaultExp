from sympy import symbols, Eq, solve, sqrt, atan2, acos
from ArmVisualiser2 import ArmVisualiser
from ArmKinematics2 import ArmKinematics

# Define symbolic variables
theta_1, theta_2 = symbols('theta_1 theta_2')

# Given values
x, y = 750, -50  # Target position of end effector
s1, s4 = 500, 50  # Additional link parameters
L1, L2 = 600, 450 + s4  # Link lengths

# Compute r using SymPy
r = sqrt(x**2 + y**2)

# Define inverse kinematics equations
eq1 = Eq(theta_2, acos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)))
eq2 = Eq(theta_1, atan2(y, x) - acos((r**2 + L1**2 - L2**2) / (2 * L1 * r)))

# Solve the system
solution = solve((eq1, eq2), (theta_1, theta_2))

kinematics = ArmKinematics(table1)

transforms =  kinematics.getAllJointGlobPose()

kinematics.checkCorrectness()

visualiser = ArmVisualiser()
visualiser.PlotUR5e(transforms)

visualiser.Show()
