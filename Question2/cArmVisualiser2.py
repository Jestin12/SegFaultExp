import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# *************************** cArmVisualiser.py ***************************************

# Filename:       cArmVisualiser.py
# Author:         Jestin

# Description:  This file defines the cArmVisualiser class which plots the reference
#               frames of the robot arm on a matplotlib figure.
#               Using the mPlotUR5e method it receives a list of 4x4 transform matrices
#               and extracts the rotation matrix and global position vectors to visulise the
#               frames on a 3D graph

# Dependencies: numpy   matplotlib.pyplot   mpl_toolkits.mplot3d 

# ************************************************************************************


class cArmVisualiser:

    # Instantiates the cArmVisualiser class and initializes a matplotlib 3D figure
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    # Plots the frames
    # Transformations -> list of 4x4 transformation matrices
    def mPlotUR5e(self, Transformations):

        # Initialize origin and frame vectors (as 1D arrays)
        Origin = np.array([0, 0, 0])
        OriginPrev = np.array([0, 0, 0])

        # Frame vectors representing X, Y, Z axes
        FrameArrowI = np.array([50, 0, 0])  # X-axis (red)
        FrameArrowJ = np.array([0, 50, 0])  # Y-axis (green)
        FrameArrowK = np.array([0, 0, 50])  # Z-axis (blue)

        # Plot the axes for the global frame (origin)
        self.ax.quiver(*Origin, *FrameArrowI, color='r', label="X-axis")  # x-axis in red
        self.ax.quiver(*Origin, *FrameArrowJ, color='g', label="Y-axis")  # y-axis in green
        self.ax.quiver(*Origin, *FrameArrowK, color='b', label="Z-axis")  # z-axis in blue

        # Annotate the initial frame (frame 0) at origin
        self.ax.text(Origin[0], Origin[1], Origin[2], f"Frame 0", color='black', fontsize=10, ha='center')

        # Loop through Transformations to plot subsequent frames
        for i, T in enumerate(Transformations):
            # Extract position (Origin) from transformation matrix
            Origin = T[:3, 3]

            # Apply rotation matrix to frame vectors
            FrameArrowI = T[:3, :3] @ np.array([50, 0, 0])  # X-axis vector in this frame
            FrameArrowJ = T[:3, :3] @ np.array([0, 50, 0])  # Y-axis vector in this frame
            FrameArrowK = T[:3, :3] @ np.array([0, 0, 50])  # Z-axis vector in this frame

            # Plot the axes for the current frame
            if i == 0:
                self.ax.quiver(*Origin, *FrameArrowI, color='r', label="X-axis")
                self.ax.quiver(*Origin, *FrameArrowJ, color='g', label="Y-axis")
                self.ax.quiver(*Origin, *FrameArrowK, color='b', label="Z-axis")
            else:
                self.ax.quiver(*Origin, *FrameArrowI, color='r')
                self.ax.quiver(*Origin, *FrameArrowJ, color='g')
                self.ax.quiver(*Origin, *FrameArrowK, color='b')

            # Plot the line connecting the previous frame to the current one
            self.ax.plot([OriginPrev[0], Origin[0]], [OriginPrev[1], Origin[1]], [OriginPrev[2], Origin[2]],
                         marker='o', linestyle='-', color='purple', linewidth=2)

            # Update previous origin for the next iteration
            OriginPrev = np.copy(Origin)

        # Set limits and labels for the 3D graph
        self.ax.set_xlim([0, 1000])
        self.ax.set_ylim([0, 1000])
        self.ax.set_zlim([0, 1000])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')


    #   Plots the circular obstacle on the X-Z plane
    def PlotObstacle(self, Position, Radius):
        
        Theta = np.linspace(0, 2*np.pi, 100)    #   Angle values

        #   Creating circle coordinates in the XY plane
        x = Position[0] + Radius * np.cos(Theta)
        y = np.zeros_like(x)
        z = Position[1] + Radius * np.sin(Theta)

        #   Plots the circle
        self.ax.plot(x, y, z, linestyle='-', color='yellow', linewidth=2)

        print("circle should've plotted")


    #   Shows the plots on the figure
    def Show(self):
        plt.show()


