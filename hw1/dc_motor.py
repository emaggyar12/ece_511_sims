import control as ct
import numpy as np
import matplotlib.pyplot as plt

# constants
Km = .05
R = 1.2
L = .05
Jm = 8e-4
J = .02
N = 12
Je = J + (N^2)*Jm

A_matrix = np.array([[0, 1, 0], [0, 0, N*Km / Je], [0, -N*Km/L, -R/L]])
B_matrix = np.array([[0, 0], [0, -1/Je], [1/L, 0]])
C_matrix = np.array([[1, 0, 0], [0, 1, 0]])
D_matrix = np.array([[0, 0], [0, 0]])

# initial states
x0 = np.array([0, 0, 0])

# input and time vectors
t1 = np.arange(0, 2.01, .01)
t2 = np.arange(2.01, 4.01, .01)
t = np.hstack((t1, t2))
print(t.shape)

v1 = 3*np.ones(t1.shape)
v2 = -3*np.ones(t2.shape)
v = np.hstack((v1, v2))

TL = np.zeros(t.shape)

input_vector = np.vstack((v, TL))
print(input_vector.shape)

states = ['theta', 'omega', 'current']
inputs = ['voltage', 'load_torque']
outputs = ['theta', 'omega']


# define system
sys = ct.ss(A_matrix, B_matrix, C_matrix, D_matrix, name='DC Motor', inputs=inputs, states=states, outputs=outputs)

# simulate response
response = ct.forced_response(sysdata=sys, timepts=t, inputs=input_vector, initial_state=x0)

# Plot all outputs
plt.figure()

for i, output in enumerate(outputs):
    plt.plot(response.time, response.outputs[i], label=output)

plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Outputs')
plt.title('Output Trajectories')
plt.legend()
plt.tight_layout()


# Plot all states
plt.figure()

for i, state in enumerate(states):
    plt.plot(response.time, response.states[i], label=state)

plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('States')
plt.title('State Trajectories')
plt.legend()
plt.tight_layout()

plt.show()