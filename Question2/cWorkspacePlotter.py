import matplotlib.pyplot as plt
import numpy as np  
from Question2.cArmKinematics2 import cArmKinematics
from Question2.cDHTable2 import cDHTable

# *************************** cWorkspacePlotter.py ***************************************

# Filename:         cWorkspacePlotter.py
# Author:           Alfie

# Description:      The file defines the cWorkspacePlotter which visualises on a 2D plot
#                   what positions the end effector of the robot limb can achieve given
#                   a set of joint limits.
#                   This is done by recursively producing transformation matrices of the
#                   robot arm using the DHTable2 class and extracting the end effector 
#                   position using the ArmKinematics2 class, for every possible combination
#                   of joint angles and joint extrusions.
#
#
# Dependencies:     matplotlib.pyplot   numpy   ArmKinematics2  DHTable2

# ************************************************************************************


class cWorkspacePlotter():

    NUM_POINTS = 20

    #   cWorkspacePlotter constructor, receives joint limits
    def __init__(self, S1Lowlim, S1Uplim, S4LowLim, S4UpLim, Theta2LowLim, Theta2UpLim, Theta3LowLim, Theta3UpLim):
        self.S1Lowlim = S1Lowlim
        self.S1Uplim = S1Uplim

        self.S4LowLim = S4LowLim
        self.S4UpLim = S4UpLim

        self.Theta2LowLim = Theta2LowLim
        self.Theta2UpLim = Theta2UpLim
        
        self.Theta3LowLim = Theta3LowLim
        self.Theta3UpLim = Theta3UpLim

    def mPlotWorkspace(self):
        EndEffectorPosition = []

        # Loop over all the parameters to compute end effector positions
        for S1 in np.linspace(self.S1Lowlim, self.S1Uplim, self.NUM_POINTS):

            for S2 in np.linspace(self.S4LowLim, self.S4UpLim, self.NUM_POINTS):

                for Theta2 in np.linspace(self.Theta2LowLim, self.Theta2UpLim, self.NUM_POINTS):

                    for Theta3 in np.linspace(self.Theta3UpLim, self.Theta3LowLim, self.NUM_POINTS):
                        
                        JointAngles = [0, Theta2, Theta3 - np.pi/2, 0]
                        Table = cDHTable(JointAngles, S1, S2)

                        Kinematics = cArmKinematics(Table)  
                        Kinematics.mGetAllJointGlobPose()
                        EndEffectorPosition.append(Kinematics.mEndeffectorPosition())

        EndEffectorPosition = np.array(EndEffectorPosition)

        Xvals = EndEffectorPosition[:, 0]  # x-position
        Zvals = EndEffectorPosition[:, 2]  # z-position

        plt.scatter(Xvals, Zvals, color='blue', marker='o', label="End Effector Position")

        # Add labels and grid
        plt.xlabel("X position (mm)")
        plt.ylabel("Z position (mm)")
        plt.title("Workspace Plot in x-z Plane")
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.show()