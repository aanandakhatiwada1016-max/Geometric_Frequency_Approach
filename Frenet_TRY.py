import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
os.chdir(os.path.dirname(__file__))

# ==========================================
# 1. GLOBAL PARAMETERS & TIME VECTOR
# ==========================================
f_o = 50.0                   # Nominal Frequency [Hz]
omega_o = 2 * np.pi * f_o    # Nominal Angular Frequency [rad/s]
V_base = 15.0                # Base voltage [kV]
alpha = 2 * np.pi / 3        # 120 deg shift 
fs = 10000                   # Sampling Frequency [Hz]
t_end = 0.1                  # 100 ms simulation
t = np.linspace(0, t_end, int(fs * t_end))

# ==========================================
# 2. INPUT SELECTION 
# ==========================================

# --- Case E1: Balanced, Stationary 
#V, omega_a = 15.0, omega_o
#v_a = V * np.sin(omega_a * t + np.pi/6)
#v_b = V * np.sin(omega_a * t + np.pi/6 - alpha)
#v_c = V * np.sin(omega_a * t + np.pi/6 + alpha)

# --- Case E2: Frequency Mismatch 
#V, omega_a = 15.0, 1.2 * omega_o
#v_a = V * np.sin(omega_a * t + np.pi/6)
#v_b = V * np.sin(omega_a * t + np.pi/6 - alpha)
#v_c = V * np.sin(omega_a * t + np.pi/6 + alpha)

# --- Case E3: Magnitude change frequency remains same 
#V, omega_a = 15+3*np.sin(0.2*omega_o*t), omega_o
#v_a = V * np.sin(omega_a * t + np.pi/6)
#v_b = V * np.sin(omega_a * t + np.pi/6 - alpha)
#v_c = V * np.sin(omega_a * t + np.pi/6 + alpha)

# --- Case E4: Magnitude remains same Frequency change 
#V, omega_a = 15, omega_o+2*np.pi*np.sin(20*np.pi*t)
#v_a = V * np.sin(omega_a * t + np.pi/6)
#v_b = V * np.sin(omega_a * t + np.pi/6 - alpha)
#v_c = V * np.sin(omega_a * t + np.pi/6 + alpha)

# --- Case E5: Unbalanced --- 
#V, omega_a = 15.0, omega_o
#v_a = 1.0 * V * np.sin(omega_a * t + np.pi/6)
#v_b = 1.2 * V * np.sin(omega_a * t + np.pi/6 - alpha)
#v_c = 0.8 * V * np.sin(omega_a * t + np.pi/6 + alpha)

#--- Case E6: Harmonic Content
V, omega_a = 15.0, omega_o
v_a = V * ((np.sin(omega_a * t + np.pi/6)) + (0.1 * np.sin(5 * (omega_a * t + np.pi/6))))
v_b = V * ((np.sin(omega_a * t +np.pi/6- alpha)) + (0.2 * np.sin(5 * (omega_a * t +np.pi/6 - alpha))))
v_c = V * ((np.sin(omega_a * t +np.pi/6+ alpha)) + (0.1 * np.sin(5 * (omega_a * t +np.pi/6+ alpha))))

v_abc = np.array([v_a, v_b, v_c])

# ==========================================
# 3. UNIFIED PROCESSING CODE
# ==========================================

# A. Park Transform (v_dqo) 
theta_p = omega_o * t  # Fixed at nominal grid frequency

v_d = np.sqrt(2/3) * (v_abc[0]*np.sin(theta_p) + v_abc[1]*np.sin(theta_p-alpha) + v_abc[2]*np.sin(theta_p+alpha))
v_q = np.sqrt(2/3) * (v_abc[0]*np.cos(theta_p) + v_abc[1]*np.cos(theta_p-alpha) + v_abc[2]*np.cos(theta_p+alpha))
v_0 = np.sqrt(1/3) * (v_abc[0] + v_abc[1] + v_abc[2])

# Combine into a single array for plotting
v_dqo = np.vstack([v_d, v_q, v_0])

# B. Frenet Frame Invariants 
v_prime = np.gradient(v_abc, axis=1)*fs 
v_double_prime = np.gradient(v_prime, axis=1)*fs 

#Normalization
v_mag = np.linalg.norm(v_abc, axis=0) # s' = |v|


#Azimuthal Frequency (Omega_Kappa)
cross_v_vp = np.cross(v_abc.T, v_prime.T)

cross_mag = np.linalg.norm(cross_v_vp, axis=1)


omega_kappa = cross_mag/v_mag**2


#frenet_freq = (omega_kappa - omega_o) / omega_o

# Torsional Frequency (omega_tau) 
# omega_tau = (v . v' x v'') / |v x v'|^2 *|v|^3
cross_vp_vpp = np.cross(v_prime.T, v_double_prime.T)
dot_triple = np.einsum("ij,ij->i", v_abc.T, cross_vp_vpp)


denom_tau = (cross_mag**2) * (v_mag**3)

omega_tau = dot_triple/denom_tau[0]

#print("cross_mag range:", cross_mag.min(), cross_mag.max())
#print("v_mag range:", v_mag.min(), v_mag.max())
#print("dot_triple range:", dot_triple.min(), dot_triple.max())
#print("denom range:", (cross_mag**2 * v_mag**3).min(), (cross_mag**2 * v_mag**3).max())
#print("omega_tau range:", omega_tau.min(), omega_tau.max())
#print("omega_kappa range:", omega_kappa.min(), omega_kappa.max())

# ==========================================
# 6. METHOD 4: FRENET FRAME
# ==========================================
def method_frenet_frame(v_abc):
    # Design low-pass Butterworth filter
    cutoff = 2 * f_o  # 100 Hz cutoff for 50 Hz signal
    b, a = butter(4, cutoff / (0.5 * fs), btype='low')

    # Apply zero-phase filtering to each phase
    v_abc_filtered = np.zeros_like(v_abc)
    for i in range(3):
        v_abc_filtered[i, :] = filtfilt(b, a, v_abc[i, :])
    
    # Numerical derivatives
    v_prime = np.gradient(v_abc_filtered, axis=1) * fs
    #v_double_prime = np.gradient(v_prime, axis=1) * fs

    # Frenet Frame Invariants
    cross_v_vp = np.cross(v_abc_filtered, v_prime, axis=0)
    mag_v = np.linalg.norm(v_abc_filtered, axis=0)
    mag_cross = np.linalg.norm(cross_v_vp, axis=0)

    omega_kappa = mag_cross / (mag_v**2 + 1e-9)
    frenet_freq = (omega_kappa - omega_o) / omega_o

    return frenet_freq

frenet_freq = method_frenet_frame(v_abc)

# ==========================================
# 4. PLOTTING (Individual Figures)
# ==========================================
def format_plot(title, ylabel):
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

# (a) v_abc
plt.figure(figsize=(6, 4))
plt.plot(t, v_abc[0]/V_base, 'k-', label='$v_a$')
plt.plot(t, v_abc[1]/V_base, 'k--', label='$v_b$')
plt.plot(t, v_abc[2]/V_base, 'k:', label='$v_c$')
format_plot("v_abc [pu]", "v [pu(kV)]")

# (b) v_dqo
plt.figure(figsize=(6, 4))
plt.plot(t, v_dqo[0]/V_base, 'k-', label='$v_d$')
plt.plot(t, v_dqo[1]/V_base, 'k--', label='$v_q$')
plt.plot(t, v_dqo[2]/V_base, 'k:', label='$v_o$')
format_plot("v_dqo [pu]", "v [pu(kV)]")

# (c) v_TNB [cite: 381-385]
plt.figure(figsize=(6, 4))
plt.plot(t, (v_mag/V_base), 'k-', label='$v_T$')
plt.plot(t, np.zeros_like(t), 'k--', label='$v_N$')
plt.plot(t, np.zeros_like(t), 'k:', label='$v_B$')
format_plot("v_TNB [pu]", "v [pu(kV)]")

# (d) omega_kappa and omega_tau 
plt.figure(figsize=(6, 4))
plt.plot(t, ((omega_kappa - omega_o)/omega_o), 'k-', label='$\\omega_{\\kappa} - \\omega_o$')
plt.plot(t, (omega_tau/omega_o)*1e5, 'k:', label='$\\omega_{\tau}$')

format_plot("Frenet Frequencies", "$\\omega$ [pu(rad/s)]")

plt.show()


# ==========================================
fig, axs = plt.subplots(5, 1, figsize=(20, 10), sharex=True)
# Method 4: Frenet Frame
axs[4].plot(t, frenet_freq, label='Frenet Frame', color='purple')
axs[4].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axs[4].set_title('Method 4: Frenet Frame Invariants')
axs[4].set_xlabel('Time (s)')
axs[4].set_ylim(-2, 2)
axs[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()