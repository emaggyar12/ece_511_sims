import control as ct
import numpy as np
import matplotlib.pyplot as plt
import math

# constants
Ts = 0.1
c = 0.2
v_eq = 10
u_eq = c * (v_eq ** 2)

# Linearized System Response

# initial conditions
A_matrix = np.array([[1, Ts], [0, 1 - 2 * c * Ts * v_eq]])
B_matrix = np.array([[0], [Ts]])
C_matrix = np.array([0, 1])
D_matrix = np.array([0])

states = ['position', 'velocity']
inputs = ['control_input']
outputs = ['velocity']

sys = ct.ss(A_matrix, B_matrix, C_matrix, D_matrix, dt=Ts, name='Vehicle Sim', inputs=inputs, states=states, outputs=outputs)

# Initial State
del_x0 = np.array([0, 1])

# time vector
t1 = np.arange(0, 2, Ts)
t2 = np.arange(2, 6.0, Ts)
t3 = np.arange(6.0, 20, Ts)
t = np.hstack((t1, t2, t3))

# input perturbation
u1 = np.zeros_like(t1)
u2 = 2 * np.ones_like(t2)
u3 = np.zeros_like(t3)
u = np.hstack((u1, u2, u3))

response = ct.forced_response(sysdata=sys, timepts=t, inputs=u, initial_state=del_x0)

# Plot all states
plt.figure()

for i, state in enumerate(states):
    plt.plot(response.time, response.states[i], label=state)

plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('Linearized State Trajectories (Perturbation)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Nonlinear system response

# defining initial conditions and input vector
u_nonlinear = u_eq + u

x0_star = np.array([0, v_eq])
x0 = x0_star + del_x0

def position_update(t, x, u, params):
    position = x[0]
    velocity = x[1]

    input = u[0]

    # Compute next position
    p_next = position + Ts * velocity

    # Compute next velocity
    v_next = velocity + Ts * (input - c * velocity * abs(velocity))

    return np.array([p_next, v_next])

def output_update(t, x, u, params):
    velocity = x[1]

    output = velocity

    return np.array([output])

# Nonlinear dynamics:

vehicle_sim = ct.nlsys(
    position_update, output_update, name='vehicle_sim',
    states=states, outputs=outputs, inputs=inputs, dt=Ts
)

# simulate system
nonlinear_resp = ct.input_output_response(vehicle_sim, timepts=t, inputs=u_nonlinear, initial_state=x0)

# Plot nonlinear states
for i, state in enumerate(states):
    plt.figure()

    plt.plot(nonlinear_resp.time, nonlinear_resp.states[i], label=state)

    plt.xlabel('Time (s)')
    plt.ylabel(state)
    plt.title(f'Nonlinear {state.capitalize()} Trajectory')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# Plotting nonlinear and linear response on the same plot

linear_velocity = v_eq + response.outputs

plt.figure()

plt.plot(response.time, linear_velocity, label='Linearized')
plt.plot(
    nonlinear_resp.time,
    nonlinear_resp.outputs,
    linestyle='--',
    label='Nonlinear'
)

plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.title('Linearized vs Nonlinear Velocity')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plotting error

# Absolute difference between linearized and nonlinear outputs
output_difference = np.abs(linear_velocity - nonlinear_resp.outputs)

plt.figure()

plt.plot(response.time, output_difference)

plt.xlabel('Time (s)')
plt.ylabel('Absolute Difference')
plt.title('Absolute Difference Between Linearized and Nonlinear Velocity')
plt.grid(True)
plt.tight_layout()

plt.show()

plt.show()
