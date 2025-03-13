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

    # Instantiates the ArmVisualiser class and initializes a matplotlib 3D figure
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    # Plots the frames
    # transformations -> list of 4x4 transformation matrices
    def PlotUR5e(self, transformations):

        #   Origin holds the position vector (1D array) of the frame with respect of to the 
        #   global frame
        Origin = np.array([0, 0, 0])

        #   OriginPrev holds the position vector (1D array) of the previous frame so the link
        #   between the to joints can be plotted
        OriginPrev = np.array([0, 0, 0])

        #   FrameArrow I, J and K represent the elementary vectors that are used to 
        #   visualise the axis x, y and z of the frames 
        FrameArrowI = np.array([50, 0, 0])  # X-axis (red)
        FrameArrowJ = np.array([0, 50, 0])  # Y-axis (green)
        FrameArrowK = np.array([0, 0, 50])  # Z-axis (blue)

        #   Plots the vectors FrameArrow I, J and K as coloured arrows on the 3D graph        
        self.ax.quiver(*Origin, *FrameArrowI, color='r', label="X-axis")  # x-axis in red
        self.ax.quiver(*Origin, *FrameArrowJ, color='g', label="Y-axis")  # y-axis in green
        self.ax.quiver(*Origin, *FrameArrowK, color='b', label="Z-axis")  # z-axis in blue

        #   Annotates the frames with their frame number and global coordinate
        self.ax.text(Origin[0], Origin[1], Origin[2], f"Frame 0", color='black', fontsize=10, ha='center')

        # Loop through transformations to plot subsequent frames
        for i, T in enumerate(transformations):

            #   Assigns Origin to be the coordinate (position vector) of the frame
            #   found in the transformation matrix (0-i)
            Origin = T[:3, 3]

            #   Rotates the elementary vectors using the rotation matrix of the 
            #   transformation matrix to align with the new frame's axis
            FrameArrowI = T[:3, :3] @ np.array([50, 0, 0])  # X-axis vector in this frame
            FrameArrowJ = T[:3, :3] @ np.array([0, 50, 0])  # Y-axis vector in this frame
            FrameArrowK = T[:3, :3] @ np.array([0, 0, 50])  # Z-axis vector in this frame

            #   Plots the vectors FrameArrow I, J and K as coloured arrows on the 3D graph
            if i == 0:
                self.ax.quiver(*Origin, *FrameArrowI, color='r', label="X-axis")
                self.ax.quiver(*Origin, *FrameArrowJ, color='g', label="Y-axis")
                self.ax.quiver(*Origin, *FrameArrowK, color='b', label="Z-axis")
            else:
                self.ax.quiver(*Origin, *FrameArrowI, color='r')
                self.ax.quiver(*Origin, *FrameArrowJ, color='g')
                self.ax.quiver(*Origin, *FrameArrowK, color='b')

            #   Plot the line connecting the previous frame to the current one
            self.ax.plot([OriginPrev[0], Origin[0]], [OriginPrev[1], Origin[1]], [OriginPrev[2], Origin[2]], marker='o', linestyle='-', color='purple', linewidth=2)

            #   Update OriginPrev to the current Origin for the next iteration
            OriginPrev = np.copy(Origin)


        # Set limits and labels for the 3D graph
        self.ax.set_xlim([-1000, 1000])
        self.ax.set_ylim([-1000, 1000])
        self.ax.set_zlim([0, 1000])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        print("Close the plot window to continue onto the next plot")

        #   Generates the 3D graph window
        plt.show()



