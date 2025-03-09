import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ArmVisualiser:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')


    def plot_UR5e(self, transformations):

        points = np.array([[0,0,0]])

        # Extract joint positions
        for i, T in enumerate(transformations):
            pos = np.array(T[:3, 3]).flatten()  # Ensure it's a 1D array
            print(f"Joint {i+1} Position: {pos}, Shape: {pos.shape}")  # Debugging
            points = np.vstack((points, pos.reshape(1, 3)))  # Stack as (1,3) row

        print("Final Points Array:\n", points)
        print("Shape of Final Points:", points.shape)  # Debugging

        # Ensure points is a 2D array with shape (n, 3)
        if points.shape[1] != 3:
            raise ValueError(f"Points array shape mismatch! Expected (n, 3), got {points.shape}")

        # Extract x, y, z as separate 1D arrays
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Ensure x, y, z are 1D arrays
        if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
            raise ValueError("x, y, or z coordinates are not 1D arrays!")

        # Plot joints and links
        self.ax.plot(x, y, z, '-o', label="UR5e Arm", markersize=6)
        plt.show()

        for i, T in enumerate(transformations):
            self.draw_coordinate_frame(T, label=f'J{i+1}')

        # Set axis limits and labels
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([0, 1.5])
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Y Axis")
        self.ax.set_zlabel("Z Axis")
        self.ax.legend()
        plt.show()

    def draw_coordinate_frame(self, T, scale=0.1, label=""):
        """Draw coordinate frames for each joint."""
        origin = T[:3, 3]
        x_axis = T[:3, 0] * scale
        y_axis = T[:3, 1] * scale
        z_axis = T[:3, 2] * scale

        self.ax.quiver(*origin, *x_axis, color='r', linewidth=2, label=f'{label} X')
        self.ax.quiver(*origin, *y_axis, color='g', linewidth=2, label=f'{label} Y')
        self.ax.quiver(*origin, *z_axis, color='b', linewidth=2, label=f'{label} Z')
