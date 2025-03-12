import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# *************************** ArmVisualiser.py ***************************************

# Filename:       ArmVisualiser.py
# Author:         Jestin

# Description:  This file defines the ArmVisualiser class which plots the reference
#               frames of the robot arm on a matplotlib figure.
#               Using the PlotUR5e method it receives a list of 4x4 transform matrices
#               and extracts the rotation matrix and global position vectors to visulise the
#               frames on a 3D graph

# Dependencies: numpy   matplotlib.pyplot   mpl_toolkits.mplot3d 

# ************************************************************************************

class ArmVisualiser:

    #   Instantiates the ArmVisualiser class and instantiates a matplotlib 3D figure
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')


    #   Plots the frames
    #   transformations -> list of 4x4 transformation matrices
    def PlotUR5e(self, transformations):

        #   Origin holds the position vector of the frame with respect of to the 
        #   global frame
        Origin = np.array([[0],[0],[0]])

        #   FrameArrow I, J and K represent the elementary vectors that are used to 
        #   visualise the axis x, y and z of the frames 
        FrameArrowI = np.array([[100], [0], [0]])
        FrameArrowJ = np.array([[0], [100], [0]])
        FrameArrowK = np.array([[0], [0], [100]])

        #   Plots the vectors FrameArrow I, J and K as coloured arrows on the 3D graph
        self.ax.quiver(*Origin, *FrameArrowI, color='g', label="X-axis") # x-axis is greeen
        self.ax.quiver(*Origin, *FrameArrowJ, color='r', label="Y-axis") # y-axis is red
        self.ax.quiver(*Origin, *FrameArrowK, color='b', label="Z-axis") # z-axis is blue

        #   Annotates the frames with their frame number and global coordinate
        self.ax.text(Origin[0,0], Origin[1,0], Origin[2,0], f"Frame{0}", color='black', fontsize=10, ha='center')
        # self.ax.text(Origin[0,0], Origin[1,0], Origin[2,0], f"({Origin[0,0]}, {Origin[1,0]}, {Origin[2,0]})", color='black', fontsize=10, ha='center')    


        for i, T in enumerate(transformations):

            #   Assigns Origin to be the coordinate (position vector) of the frame
            Origin = T[:3, 3]
    
            #   Rotates the elementary vectors using the rotation matrix of the 
            #   transformation matrix to align with the new frame's axis
            FrameArrowI = T[:3, :3]@FrameArrowI
            FrameArrowJ = T[:3, :3]@FrameArrowJ
            FrameArrowK = T[:3, :3]@FrameArrowK

        #   Plots the vectors FrameArrow I, J and K as coloured arrows on the 3D graph
            self.ax.quiver(*Origin, *FrameArrowI, color='g', label="X-axis") # x-axis is greeen
            self.ax.quiver(*Origin, *FrameArrowJ, color='r', label="Y-axis") # y-axis is red
            self.ax.quiver(*Origin, *FrameArrowK, color='b', label="Z-axis") # z-axis is blue

            # self.ax.text(*Origin, f"Frame{i+1} ", color='black', fontsize=10, ha='center')
            # self.ax.text(Origin[0,0], Origin[1,0], Origin[2,0], f"({Origin[0,0]}, {Origin[1,0]}, {Origin[2,0]})", color='black', fontsize=10, ha='center')

        #   Sets the limits of the 3D graph and axis labels
        self.ax.set_xlim([-1500, 1500])
        self.ax.set_ylim([-1500, 1500])
        self.ax.set_zlim([-1500, 1500])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # self.ax.legend()

        #   Generates the 3D graph window
        plt.show()

