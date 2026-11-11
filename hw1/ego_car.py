import control as ct
import numpy as np
import matplotlib.pyplot as plt
import math

# constants
beta = math.exp(-0.1/.5)

A_matrix = np.array([[1, -0.1, 0], [0, 1, -0.1], [0, 0, beta]])
B_matrix = np.array([[0, 0], [0, .1], [1 - beta, 0]])
C_matrix = np.array([[1, 0, 0], [0, 1, 0]])
D_matrix = np.array([[0, 0], [0, 0]])

states = ['dist_error', 'vel_error', 'ego_acc']
inputs = ['ego_contr_acc', 'lead_acc']
outputs = ['dist_error', 'vel_error']

sys = ct.ss(A_matrix, B_matrix, C_matrix, D_matrix, dt=0.1, name='Ego Vehicle', inputs=inputs, states=states, outputs=outputs)


class SimulationSettings:
    def __init__(self, x0=0, t=0, u=0):
        self.x0 = x0
        self.t = t
        self.u = u

# part 2 simulation
def part2_sims():
    # initial states
    x0 = np.array([10, 0, 0])

    # time vector
    t = np.arange(0, 20.1, .1)

    u = np.zeros(t.shape)
    a_L = np.zeros(t.shape)

    input_vector = np.vstack((u, a_L))

    response = ct.forced_response(sysdata=sys, timepts=t, inputs=input_vector, initial_state=x0)

    return response

def part3_sims():
    # initial states
    x0 = np.array([10, 0, 0])

    # time vectors
    t1 = np.arange(0, 5, 0.1)
    t2 = np.arange(5, 8, 0.1)
    t3 = np.arange(8, 12, 0.1)
    t4 = np.arange(12, 20.1, 0.1)
    t = np.hstack((t1, t2, t3, t4))

    # input vectors
    u = np.zeros(t.shape)

    a_L_1 = np.zeros(t1.shape)
    a_L_2 = -1.5*np.ones(t2.shape)
    a_L_3 = 0.5*np.ones(t3.shape)
    a_L_4 = np.zeros(t4.shape)
    a_L = np.hstack((a_L_1, a_L_2, a_L_3, a_L_4))

    input_vector = np.vstack((u, a_L))

    response = ct.forced_response(sysdata=sys, timepts=t, inputs=input_vector, initial_state=x0)

    return response

# Return specific simulation
# response = part2_sims()
response = part3_sims()

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