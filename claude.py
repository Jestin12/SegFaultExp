import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for visualization"""
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def plot_coordinate_frame(ax, transform_matrix, scale=1.0, label=None):
    """
    Plot a 3D coordinate frame based on a 4x4 homogeneous transformation matrix
    
    Parameters:
    -----------
    ax : Axes3D object
        Matplotlib 3D axis
    transform_matrix : ndarray (4x4)
        Homogeneous transformation matrix
    scale : float
        Scale factor for the coordinate frame arrows
    label : str
        Text label for the coordinate frame
    """
    # Extract the rotation matrix (R) and translation vector (t)
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3]
    
    # Define the origin of the coordinate frame
    origin = t
    
    # Create axis vectors for the coordinate frame
    x_axis = origin + scale * R[:, 0]
    y_axis = origin + scale * R[:, 1]
    z_axis = origin + scale * R[:, 2]
    
    # X-axis (Red)
    x_arrow = Arrow3D([origin[0], x_axis[0]],
                      [origin[1], x_axis[1]],
                      [origin[2], x_axis[2]],
                      mutation_scale=20, lw=2, arrowstyle='-|>', color='r')
    ax.add_artist(x_arrow)
    
    # Y-axis (Green)
    y_arrow = Arrow3D([origin[0], y_axis[0]],
                      [origin[1], y_axis[1]],
                      [origin[2], y_axis[2]],
                      mutation_scale=20, lw=2, arrowstyle='-|>', color='g')
    ax.add_artist(y_arrow)
    
    # Z-axis (Blue)
    z_arrow = Arrow3D([origin[0], z_axis[0]],
                      [origin[1], z_axis[1]],
                      [origin[2], z_axis[2]],
                      mutation_scale=20, lw=2, arrowstyle='-|>', color='b')
    ax.add_artist(z_arrow)
    
    # Add text label if provided
    if label:
        ax.text(origin[0], origin[1], origin[2], label, fontsize=12)

def visualize_transforms(transforms, labels=None, scale=1.0, view_angles=(30, 30), figsize=(10, 8)):
    """
    Visualize multiple coordinate frames based on homogeneous transformation matrices
    
    Parameters:
    -----------
    transforms : list of ndarray
        List of 4x4 homogeneous transformation matrices
    labels : list of str
        List of labels for each coordinate frame
    scale : float
        Scale factor for the coordinate frame arrows
    view_angles : tuple of (elevation, azimuth)
        View angles for the 3D plot
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get axis limits
    min_vals = np.zeros(3)
    max_vals = np.zeros(3)
    
    # Default labels if none provided
    if labels is None:
        labels = [f"Frame {i}" for i in range(len(transforms))]
    
    # Plot each coordinate frame
    for i, transform in enumerate(transforms):
        label = labels[i] if i < len(labels) else f"Frame {i}"
        plot_coordinate_frame(ax, transform, scale=scale, label=label)
        
        # Update axis limits
        t = transform[:3, 3]
        for j in range(3):
            min_vals[j] = min(min_vals[j], t[j] - scale)
            max_vals[j] = max(max_vals[j], t[j] + scale)
    
    # Set equal aspect ratio
    max_range = max(max_vals - min_vals)
    mid_x = (max_vals[0] + min_vals[0]) * 0.5
    mid_y = (max_vals[1] + min_vals[1]) * 0.5
    mid_z = (max_vals[2] + min_vals[2]) * 0.5
    
    ax.set_xlim(mid_x - max_range/1.5, mid_x + max_range/1.5)
    ax.set_ylim(mid_y - max_range/1.5, mid_y + max_range/1.5)
    ax.set_zlim(mid_z - max_range/1.5, mid_z + max_range/1.5)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title('Homogeneous Transformation Visualization')
    
    # Set view angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    plt.tight_layout()
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Create a world coordinate frame (identity transform)
    T_world = np.eye(4)
    
    # Create a transform with translation and rotation
    # This represents a coordinate frame rotated 45 degrees around z-axis
    # and translated to position [2, 3, 1]
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    T_frame1 = np.eye(4)
    T_frame1[:3, :3] = R_z
    T_frame1[:3, 3] = np.array([2, 3, 1])
    
    # Create another transform for demonstration
    # This one is rotated 30 degrees around x-axis
    # and then translated relative to the first frame
    phi = np.radians(30)
    c, s = np.cos(phi), np.sin(phi)
    R_x = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    
    T_frame2 = np.eye(4)
    T_frame2[:3, :3] = R_x
    T_frame2[:3, 3] = np.array([1, 1, 2])
    
    # Compose transformations
    T_frame2_world = T_frame1 @ T_frame2
    
    # Visualize all coordinate frames
    transforms = [T_world, T_frame1, T_frame2_world]
    labels = ["World", "Frame 1", "Frame 2"]
    
    fig, ax = visualize_transforms(transforms, labels, scale=1.0)
    plt.show()