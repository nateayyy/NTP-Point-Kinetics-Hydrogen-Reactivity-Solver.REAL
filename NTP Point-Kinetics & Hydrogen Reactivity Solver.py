import numpy as np
import matplotlib.pyplot as plt

# beta_i: Delayed neutron fractions for each group
beta_i = np.array([0.00021, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027])
# lambda_i: Decay constants for each group (s^-1)
lambda_i = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])

beta_total = np.sum(beta_i)
Lambda = 1e-4  # Prompt neutron generation time (seconds)

dt = 0.001     # Time step of 1 millisecond
t_max = 10.0   # Simulate 10 seconds total
time = np.arange(0, t_max, dt)
n_steps = len(time)

Power = np.zeros(n_steps)
C = np.zeros((6, n_steps))  # 6 rows for precursor groups, columns for time steps

# Initial Steady-State Conditions
Power[0] = 100.0  # Starting at 100 Megawatts

# Calculate equilibrium precursor concentrations at 100 MW
for j in range(6):
    C[j, 0] = (beta_i[j] / lambda_i[j]) * (Power[0] / Lambda)

# Simulating a sudden surge of cold liquid hydrogen at t = 1.0 second
# which acts as a dense moderator and injects positive reactivity.
rho = np.zeros(n_steps)
rho_step = 0.003  # Positive step insertion (approx 0.46 $)

for i, t in enumerate(time):
    if t >= 1.0:
        rho[i] = rho_step
    else:
        rho[i] = 0.0

# Euler
for i in range(n_steps - 1):
    
    # 1. Calculate the rate of change of Power (dP/dt)
    # Equation: dP/dt = [ (rho - beta) / Lambda ] * P + sum(lambda_i * C_i)
    dP_dt = ((rho[i] - beta_total) / Lambda) * Power[i]
    for j in range(6):
        dP_dt += lambda_i[j] * C[j, i]
        
    # Update Power for the next time step
    Power[i+1] = Power[i] + dP_dt * dt
    
    # 2. Calculate the rate of change for each Precursor Group (dC_i/dt)
    # Equation: dC_i/dt = (beta_i / Lambda) * P - lambda_i * C_i
    for j in range(6):
        dC_dt = (beta_i[j] / Lambda) * Power[i] - lambda_i[j] * C[j, i]
        C[j, i+1] = C[j, i] + dC_dt * dt

# Visualized Data
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time, Power, color='tab:red', linewidth=2.5, label='Reactor Power (MW)')

# Formatting the graph
ax.set_title('NTP Reactor Transient: Power Spike from Cold Hydrogen Surge', fontweight='bold', fontsize=14)
ax.set_xlabel('Time (Seconds)', fontweight='bold')
ax.set_ylabel('Reactor Power (Megawatts)', fontweight='bold')
ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Hydrogen Surge (Reactivity Insertion)')

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left', fontsize=11)
plt.tight_layout()

plt.show()

idx_spike = np.argmax(Power)
print(f"--- Transient Analysis ---")
print(f"Baseline Power: 100 MW")
print(f"Reactivity Injected: +{rho_step} \Delta k/k at t=1.0s")
print(f"Peak Power Reached: {Power[idx_spike]:.2f} MW")