import matplotlib.pyplot as plt
import numpy as np  
from ArmKinematics2 import ArmKinematics
from DHTable2 import DHTable

class WorkspacePlotter():

    NUM_POINTS = 20

    def __init__(self, s1_llim, s1_ulim, s4_llim, s4_ulim, theta_2_llim, theta_2_ulim, theta_3_llim, theta_3_ulim):
        self.s1_llim = s1_llim
        self.s1_ulim = s1_ulim

        self.s4_llim = s4_llim
        self.s4_ulim = s4_ulim

        self.theta_2_llim = theta_2_llim
        self.theta_2_ulim = theta_2_ulim
        
        self.theta_3_llim = theta_3_llim
        self.theta_3_ulim = theta_3_ulim

    def plotWorkspace(self):
        end_effector_positions = []

        # Loop over all the parameters to compute end effector positions
        for s1 in np.linspace(self.s1_llim, self.s1_ulim, self.NUM_POINTS):
            for s2 in np.linspace(self.s4_llim, self.s4_ulim, self.NUM_POINTS):
                for theta_2 in np.linspace(self.theta_2_llim, self.theta_2_ulim, self.NUM_POINTS):
                    for theta_3 in np.linspace(self.theta_3_llim, self.theta_3_ulim, self.NUM_POINTS):
                        joint_angles = [0, theta_2, theta_3 - np.pi/2, 0]
                        table = DHTable(joint_angles, s1, s2)

                        kinematics = ArmKinematics(table)  
                        kinematics.getAllJointGlobPose()
                        end_effector_positions.append(kinematics.endeffectorPosition())

        end_effector_positions = np.array(end_effector_positions)

        x_vals = end_effector_positions[:, 0]  # x-position
        z_vals = end_effector_positions[:, 2]  # z-position

        plt.scatter(x_vals, z_vals, color='blue', marker='o', label="End Effector Position")

        # Add labels and grid
        plt.xlabel("X position (mm)")
        plt.ylabel("Z position (mm)")
        plt.title("Workspace Plot in x-z Plane")
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.show()