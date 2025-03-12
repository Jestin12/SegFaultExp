import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ArmVisualiser:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')


    def plot_UR5e(self, transformations, scale=0.05):

        points = np.array([[0,0,0]])

        Origin = np.array([[0],[0],[0]])

        FrameArrowI = np.array([[100], [0], [0]])
        FrameArrowJ = np.array([[0], [100], [0]])
        FrameArrowK = np.array([[0], [0], [100]])

        self.ax.quiver(Origin[0,0], Origin[1,0], Origin[2,0], *FrameArrowI, color='g', label="X-axis") # x-axis is greeen
        self.ax.quiver(Origin[0,0], Origin[1,0], Origin[2,0], *FrameArrowJ, color='r', label="Y-axis") # y-axis is red
        self.ax.quiver(Origin[0,0], Origin[1,0], Origin[2,0], *FrameArrowK, color='b', label="Z-axis") # z-axis is blue

        # self.ax.text(Origin[0,0], Origin[1,0], Origin[2,0], f"({Origin[0,0]}, {Origin[1,0]}, {Origin[2,0]})", color='black', fontsize=10, ha='center')
        self.ax.text(Origin[0,0], Origin[1,0], Origin[2,0], f"Frame{0}", color='black', fontsize=10, ha='center')


        for i, T in enumerate(transformations):

            # Origin = T@Origin
            Origin = T[:3, 3]
            print(Origin)

            FrameArrowI = T[:3, :3]@FrameArrowI
            FrameArrowJ = T[:3, :3]@FrameArrowJ
            FrameArrowK = T[:3, :3]@FrameArrowK

            self.ax.quiver(*Origin, *FrameArrowI, color='g', label="X-axis") # x-axis is greeen
            self.ax.quiver(Origin[0,0], Origin[1,0], Origin[2,0], *FrameArrowJ, color='r', label="Y-axis") # y-axis is red
            self.ax.quiver(Origin[0,0], Origin[1,0], Origin[2,0], *FrameArrowK, color='b', label="Z-axis") # z-axis is blue

            # self.ax.text(Origin[0,0], Origin[1,0], Origin[2,0], f"({Origin[0,0]}, {Origin[1,0]}, {Origin[2,0]})", color='black', fontsize=10, ha='center')
            self.ax.text(Origin[0,0], Origin[1,0], Origin[2,0], f"Frame{i+1} ", color='black', fontsize=10, ha='center')


        self.ax.set_xlim([-1000, 1000])
        self.ax.set_ylim([-1000, 1000])
        self.ax.set_zlim([-1000, 1000])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # self.ax.legend()
        plt.show()

