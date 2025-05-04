

# In[13]:


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes

def plot_position_covariance(axes: Axes, position: np.ndarray, position_covariance: np.ndarray):
  from matplotlib import patches
  w, v = np.linalg.eigh(position_covariance)
  k = 0.5
  x, y = position

  angle = np.arctan2(v[1, 0], v[0, 0])
  e1 = patches.Ellipse((x, y),
                          np.sqrt(w[0]),
                          np.sqrt(w[1]),
                          angle=np.rad2deg(angle),
                          fill=False,
                          color='orange')
  axes.add_artist(e1)
  return e1



class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise, initial_state):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        self.B = np.array([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.Q = process_noise**2 * (np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]]))

        self.R = np.array([[measurement_noise**2, 0],
                           [0, measurement_noise**2]])

        self.P = np.eye(4) * 1.0
        self.x = initial_state

    def predict(self, u):
        
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x

    def update(self, z):
        
        S = self.H @ self.P @ self.H.T + self.R
        y = z - self.H @ self.x

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


        return self.x

# Simulation parameters
dt = 0.1
time_steps = 100
process_noise = 3.0
measurement_noise = 0.01

# Initial state [x, y, vx, vy]
true_state = np.array([[0], [0], [1], [1]])
kf = KalmanFilter(dt, process_noise, measurement_noise, initial_state=true_state.copy())

# Ground truth trajectory
true_trajectory = []
predicted_trajectory = []
measured_positions = []
estimated_position_covariance = []
noisy_trajectory1 = []
noisy_trajectory2 = []

def generate_u(process_noise: float):
    
    u = np.array([[2.0], 
                  [2.0]]) 

    # Addition of Gaussian noise with mean = 0 and standard deviation = process noise 
    u_noisy = u + np.random.normal(0, process_noise, size=(2,1))

    return u, u_noisy

def generate_y(x_true: np.ndarray, measurement_noise: float):
    
    # Extracting positions from state vector 
    y = x_true[:2]

    y_noisy = y + np.random.normal(0, measurement_noise)

    return y, y_noisy

time = np.arange(time_steps) * dt

np.random.seed(42)
for t in range(time_steps):
    # control input
    u, u_noisy = generate_u(process_noise)

    # simulation prediction
    true_state = kf.A @ true_state + kf.B @ u

    # measurement of position
    y, y_noisy = generate_y(true_state, measurement_noise)

    # Generating 3 sets of noisy data
    y1, y_noisy1 = generate_y(true_state, measurement_noise)
    y2, y_noisy2 = generate_y(true_state, measurement_noise)

    predicted_state = kf.predict(u_noisy)
    measured_state = kf.update(y_noisy)

    measured1 = kf.update(y_noisy1)
    measured2 = kf.update(y_noisy2)

    #TODO after prediction step

    #add measurements to the kalman filter at different rates and compare with the prediction only solution
    estimated_position_covariance.append(kf.P[:2, :2])
    true_trajectory.append(true_state[:2])
    predicted_trajectory.append(predicted_state[:2])
    measured_positions.append(measured_state[:2])

    noisy_trajectory1.append(measured1[:2])
    noisy_trajectory2.append(measured2[:2])

true_trajectory = np.array(true_trajectory)
predicted_trajectory = np.array(predicted_trajectory)
measured_positions = np.array(measured_positions)
estimated_position_covariance = np.array(estimated_position_covariance)

noisy_trajectory1 = np.array(noisy_trajectory1)
noisy_trajectory2 = np.array(noisy_trajectory2)


# plotting

fig = plt.figure(figsize=(10, 6))
ax = fig.gca()

# for cov, pos in zip(estimated_position_covariance, predicted_trajectory):
#   plot_position_covariance(ax, pos, cov)

# Plot results
ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'g-', label='Ground Truth')


ax.plot(measured_positions[:, 0], measured_positions[:, 1], 'rx', label='Measurements (Noisy 1)')

# Additional noisy measurements 
# ax.plot(noisy_trajectory1[:, 0], noisy_trajectory1[:, 1], 'bx', label='Measurements (Noisy 2)')
# ax.plot(noisy_trajectory2[:, 0], noisy_trajectory2[:, 1], 'mx', label='Measurements (Noisy 3)')

ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'b-', label='Kalman Filter Estimate')

ax.legend()
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Kalman Filter Tracking a Mobile Robot with Control Commands')
# ax.set_title('Three sets of Noisy Measurements')


plt.show()

