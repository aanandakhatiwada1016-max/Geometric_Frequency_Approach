import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.signal import hilbert
import os
os.chdir(os.path.dirname(__file__))

# ==========================================
# 1. GLOBAL PARAMETERS
# ==========================================
f_o = 50.0                   # Nominal Frequency [Hz]
omega_o = 2 * np.pi * f_o    # Nominal Angular Frequency [rad/s]


# ==========================================
# 2. INPUT SELECTION 
# ==========================================
# 1. LOAD REAL DATA
filepath = 'test_10000.csv'
df = pd.read_csv(filepath)

# Step A: Create a clean time vector 
# Based on the data, we know it is exactly 10000 Hz.
fs=10000  # Sampling frequency in Hz
t = np.arange(len(df)) / fs  # Time vector in seconds

v_base = 19000 #convert 19kv to volt
v_peak = (v_base/np.sqrt(3))* np.sqrt(2)  # Peak voltage for normalization
theta = omega_o * t  # Nominal phase angle for reference
# Step B: Extract and Normalize (pu)
# Using the column names exactly as they appear in  CSV
va_raw = df['Macchina'].values
#Generate Phase B and Phase C from Phase A with 120-degree phase 
def make_three_phase_from_timewave(va_raw, fs, f_o):
    """
    Generate three-phase waveforms from Phase A using circular shift.
    
    Parameters:
    va_raw : array_like - Phase A waveform (N x 1), real
    fs : float - Sampling frequency [Hz]
    f_o : float - Fundamental frequency [Hz] (e.g., 50)
    
    Returns:
    v_abc : Three-phase waveforms shifted by ~120°
    """
    v_a = np.asarray(va_raw, dtype=float)
    
    
    # Samples per period for 120° = 1/3 of the period
    samples_per_period = fs / f_o  # Samples in one period
    shift_120 = int(round(samples_per_period / 3.0))
    
    # Circular shift (assume periodic signal)
    v_b = np.roll(v_a, -shift_120)  # in anticipo di 120° (leads by 120°)
    v_c = np.roll(v_a, +shift_120)  # in ritardo di 120° (lags by 120°)
    
    v_abc = np.stack((v_a, v_b, v_c), axis=0).T
    
    return v_abc

# Generate three-phase
vabc_raw = make_three_phase_from_timewave(va_raw, fs, f_o)

# Step E: Normalize to per-unit
v_abc = vabc_raw / v_peak

# Extract individual phases for clarity
v_a = v_abc[:, 0]
v_b = v_abc[:, 1]
v_c = v_abc[:, 2]


v_abc = np.array([v_a, v_b, v_c])



# 5. Calculate Sampling Info for the Algorithms
avg_ts = np.median(np.diff(t))
fs = 1.0 / avg_ts
Ts = avg_ts
N0 = int(fs/f_o)
print(f"Data Loaded Successfully. Detected Sampling Frequency: {fs:.2f} Hz")




# ==========================================
# 3. METHOD 1: SRF-PLL
# ==========================================
def method_srf_pll(v_abc):
    wc = 2 * np.pi * 5 #natural angular frequency
    zeta = 0.707        # Damping Ratio: determines the stability and overshoot of the PLL when settling
    Kp, Ki = 2 * zeta * wc, wc**2  # Proportinal Gain and Integral gain
    va0, vb0, vc0 = v_abc[:,0]

    v_alpha0 = np.sqrt(2/3)*(va0 - 0.5*vb0 - 0.5*vc0)

    v_beta0 = np.sqrt(2/3)*(np.sqrt(3)/2*(vb0-vc0))

    theta_hat = np.arctan2(v_beta0, v_alpha0)
    integral_error = 0
    
    srf_freq = []
    srf_pu = []

    for i in range(len(t)):
        va, vb, vc = v_abc[:, i]
        # Clarke & Park
        v_alpha = np.sqrt(2/3) * (va - 0.5*vb - 0.5*vc)
        v_beta = np.sqrt(2/3) * (np.sqrt(3)/2 * (vb - vc))
        vd = v_alpha * np.cos(theta_hat) + v_beta * np.sin(theta_hat)
        vq = -v_alpha * np.sin(theta_hat) + v_beta * np.cos(theta_hat)
        # PI Loop
        integral_error += vq * Ts
        omega_hat = omega_o + Kp * vq + Ki * integral_error
        theta_hat = np.mod(theta_hat + omega_hat * Ts, 2 * np.pi)
        omega_hat = np.clip(omega_hat,2*np.pi*45,2*np.pi*55)
        srf_freq.append(omega_hat / (2 * np.pi))
        #srf_freq.append ((omega_hat - omega_o) / omega_o)
        srf_pu.append (omega_hat / omega_o)

    return np.array(srf_freq,), np.array(srf_pu)

# ==========================================
# 4. METHOD 2: DSOGI-FLL
# ==========================================

def method_dsogi_fll(v_abc):
    # --- 2. CLARKE TRANSFORM
    v_alpha = (2/3)*(v_a - 0.5*v_b - 0.5*v_c)
    v_beta = (2/3)*((np.sqrt(3)/2)*v_b -(np.sqrt(3)/2)*v_c)

    # --- 3. DSOGI PARAMETERS
    k = np.sqrt(2)
    Gamma = 10
    w_est = 2*np.pi*50

    # --- 4. STATES
    v_alpha_est = v_alpha[0]  # Initial condition for SOGI states
    qv_alpha_est = v_beta[0]

    v_beta_est = v_beta[0]
    qv_beta_est = -v_alpha[0]


    freq_log = []
    freq_pu = []
    
    for n in range(len(t)):
        va = v_alpha[n]
        vb = v_beta[n]

        # Errors
        e_alpha = va - v_alpha_est
        e_beta  = vb - v_beta_est

        # SOGI Alpha
        dv_alpha = (k*w_est*e_alpha - w_est*qv_alpha_est)
        dqv_alpha = w_est*v_alpha_est
        v_alpha_est += Ts*dv_alpha
        qv_alpha_est += Ts*dqv_alpha

        # SOGI Beta
        dv_beta = (k*w_est*e_beta - w_est*qv_beta_est)
        dqv_beta = w_est*v_beta_est
        v_beta_est += Ts*dv_beta
        qv_beta_est += Ts*dqv_beta
        # PNSC
        Vpos_alpha = 0.5*(v_alpha_est - qv_beta_est)
        Vpos_beta = 0.5*(qv_alpha_est + v_beta_est)
        Vneg_alpha = 0.5*(v_alpha_est + qv_beta_est)
        Vneg_beta = 0.5*(-qv_alpha_est + v_beta_est)
        # FLL
        V2 = Vpos_alpha**2 + Vpos_beta**2 + 1e-12
        gamma= (k * w_est / V2) * Gamma
        eps_fll = (e_alpha*qv_alpha_est + e_beta*qv_beta_est)

        dw = -gamma * eps_fll



        w_est += Ts * dw


        w_pu = w_est  / omega_o
        freq_log.append(w_est / (2 * np.pi))
        freq_pu.append(w_pu)
        #freq_log2.append(w_est/(2*np.pi))

    return np.array(freq_log), np.array(freq_pu)

    

# ==========================================
# 5. METHOD 3: THREE-LEVEL DFT
# ==========================================
def dft_estimate_frequency(v_abc):
    v_alpha = (2/3) * (v_abc[0] - 0.5*v_abc[1] - 0.5*v_abc[2]) # Clarke α-axis
    # DFT Filter Coefficients 
    nn = np.arange(N0)
    h_c = (2/N0) * np.cos(2*np.pi*nn/N0 + np.pi/N0)
    h_s = -(2/N0) * np.sin(2*np.pi*nn/N0 + np.pi/N0)
    
    # Helper to apply FIR filter
    def apply_filter(s, h):
        return np.convolve(s, h, mode='valid')

    # Level 1 
    x_c = apply_filter(v_alpha, h_c)
    x_s = apply_filter(v_alpha, h_s)
    
    # Level 2 
    x_cc = apply_filter(x_c, h_c)
    x_ss = apply_filter(x_s, h_s)
    
    # Level 3 
    x_ccc = apply_filter(x_cc, h_c)
    x_ccs = apply_filter(x_cc, h_s)
    x_ssc = apply_filter(x_ss, h_c)
    x_sss = apply_filter(x_ss, h_s)
    
    # Frequency Calculation Formula 
    num = x_ccc**2 + x_ccs**2
    den = x_ssc**2 + x_sss**2
    ratio = (num / (den + 1e-12))**0.25
    
    term = np.tan(np.pi/N0) * ratio
    f_est = (f_o * N0 / np.pi) * np.arctan(term)
    
    # 2-cycle Moving Average for stability 
    window = int(2 * N0)
    f_est = np.convolve(f_est, np.ones(window)/window, mode='valid')
    w_pu = (2*np.pi*f_est)  / omega_o
    
    return f_est, w_pu
    


# ==========================================
# 6. METHOD 4: FRENET FRAME (CORRECTED)
# ==========================================
def method_frenet_frame(v_abc):

    # Numerical derivatives
    v_prime = np.gradient(v_abc,Ts, axis=1, edge_order=2)  # Use Ts for correct scaling 
    v_double_prime = np.gradient(v_prime,Ts, axis=1, edge_order=2)  # Use Ts for correct scaling

    # Frenet Frame Invariants
    cross_vp_vpp = np.cross(v_prime, v_double_prime, axis=0)
    cross_v_vp = np.cross(v_abc, v_prime, axis=0)
    mag_v = np.linalg.norm(v_abc, axis=0)
    mag_cross = np.linalg.norm(cross_v_vp, axis=0)

    # Frenet angular frequency
    omega_kappa = mag_cross / (mag_v**2 + 1e-9)
    
    # Affine differentiation angular frequency (CORRECTED)
    # Add safeguards: abs() to avoid negative values, epsilon to avoid division by zero
    cross_ratio = np.abs(cross_vp_vpp) / (np.abs(cross_v_vp) + 1e-9)
    omega_a = np.sqrt(cross_ratio)
    
    # Replace any NaN or Inf values with nominal frequency
    omega_a = np.nan_to_num(omega_a, nan=2*np.pi*50, posinf=2*np.pi*50, neginf=2*np.pi*50)

    return omega_kappa, omega_a


def filter_frequency(omega_kappa, omega_a, cutoff):
    
    #w_geom= omega_kappa/omega_o  #in pu
    #w_affine= omega_a/omega_o    #in pu
    # CORRECT: Normalize cutoff frequency
    # Wn = cutoff / (fs/2)  where fs/2 is Nyquist frequency
    Wn = cutoff / (fs / 2)
    
    # Design filter
    b, a = butter(N=2, Wn=Wn, btype='low') #N = 2 for a second-order filter
    
    # Filter both signals
    w_filtered = lfilter(b, a, omega_kappa)
    w_affine_filtered = lfilter(b, a, omega_a)
    
    return w_filtered, w_affine_filtered

# ==========================================
# 7. MAIN CONTROL CODE (RUN ALL METHODS)
# ==========================================
srf_freq, srf_pu = method_srf_pll(v_abc)
dsogi_freq, dsogi_pu = method_dsogi_fll(v_abc)
dft_freq, dft_pu = dft_estimate_frequency(v_abc)
omega_kappa, omega_a = method_frenet_frame(v_abc)
w_filtered, w_affine_filtered = filter_frequency(omega_kappa, omega_a, cutoff=10)  # Cutoff frequency in Hz
    


# ==========================================
# 8. FINAL VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 6))

#plt.plot(t, srf_pu-1, label='SRF-PLL')
#plt.plot(t, dsogi_pu-1, label='DSOGI-FLL')
# --- Symmetrical Time Vector Alignment ---
# Calculate total lost samples due to 'valid' modes
total_lost = len(v_abc[0]) - len(dft_pu)
# Distribute the lost samples equally at the start and end
start_idx = total_lost // 2
end_idx = start_idx + len(dft_pu)

t_dft = t[start_idx:end_idx]
#plt.plot(t_dft, dft_pu-1, label='3-Level DFT', linestyle='--')
#start = int(0.25*fs)

plt.plot(t, ((w_filtered/omega_o)-1), label='Frenet Frame', linestyle=':')
#plt.axhline(y=1, color='r', linestyle='-.', label='Target (pu)')
plt.title('Comparison of 4 Frequency Estimation Methods')
plt.xlabel('Time (s)')
plt.ylabel('Frequency Deviation (pu)')
plt.ylim(-2, 2)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 9. VISUALIZATION (Frequency  in PU)
# ==========================================

fig, axs = plt.subplots(3, 2, figsize=(20, 10), sharex=True)

# Top Plot: Three-Phase Voltages
axs[0,0].plot(t, v_a, 'r', label='Phase A')
axs[0,0].plot(t, v_b, 'g', label='Phase B')
axs[0,0].plot(t, v_c, 'b', label='Phase C')
axs[0,0].set_title('Normalized Input Voltages')
axs[0,0].set_ylabel('Amplitude (pu)')
axs[0,0].grid(True, alpha=0.3)
axs[0,0].legend(loc='upper right')

# Method 1: SRF-PLL
axs[1,0].plot(t, srf_pu, label='SRF-PLL')
axs[1,0].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axs[1,0].set_title('Method 1: SRF-PLL')
axs[1,0].set_ylabel('Frequency (pu)')
axs[1,0].set_ylim(0.5, 1.5)
axs[1,0].grid(True, alpha=0.3)

# Method 2: DSOGI-FLL
axs[2,0].plot(t, dsogi_pu, label='DSOGI-FLL', color='orange')
axs[2,0].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axs[2,0].set_title('Method 2: DSOGI-FLL')
axs[2,0].set_ylabel('Frequency (pu)')
axs[2,0].set_ylim(-10, 10)
axs[2,0].grid(True, alpha=0.3)

# Method 3: 3-Level DFT 
# --- Symmetrical Time Vector Alignment ---
# Calculate total lost samples due to 'valid' modes
total_lost = len(v_abc[0]) - len(dft_pu)
# Distribute the lost samples equally at the start and end
start_idx = total_lost // 2
end_idx = start_idx + len(dft_pu)

t_dft = t[start_idx:end_idx]
axs[0,1].plot(t_dft, dft_pu, label='3-Level DFT', color='green')
axs[0,1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axs[0,1].set_title('Method 3: 3-Level DFT')
axs[0,1].set_ylabel('Frequency (pu)')
axs[0,1].set_ylim(0.8, 1.2)
axs[0,1].grid(True, alpha=0.3)

# Method 4: Frenet Frame
#axs[4].plot(t, frenet_pu[0, :], label='Frenet Frame', color='purple')
axs[1,1].plot(t,w_filtered/omega_o, label='Frenet Frame', color='purple')
axs[1,1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axs[1,1].set_title('Method 4: Frenet Frame Method1')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_ylabel('Frequency (pu)')
axs[1,1].set_ylim(0, 3)
axs[1,1].grid(True, alpha=0.3)

# Method 5: Affine Differentiation (Frenet Frame)
axs[2,1].plot(t, w_affine_filtered[0, :len(t)]/omega_o, label='Affine Differentiation', color='red')
axs[2,1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axs[2,1].set_title('Method 5: Affine Differentiation')
axs[2,1].set_xlabel('Time (s)')
axs[2,1].set_ylabel('Frequency (pu)')
axs[2,1].set_ylim(-0.5, 8)
axs[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()