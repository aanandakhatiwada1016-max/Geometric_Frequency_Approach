import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.signal import medfilt
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
filepath = 'TR energization_5 KHz.csv'
df = pd.read_csv(filepath)

# Step A: Create a clean time vector (avoids the Timestamp parsing errors)
# Based on the data, we know it is exactly 5000 Hz.
t = np.arange(len(df)) / 5000

# Step B: Extract and Normalize (pu)
# Using the column names exactly as they appear in  CSV
va_raw = df['Report_1:V_R_5'].values
vb_raw = df['Report_1:V_S_6'].values
vc_raw = df['Report_1:V_T_7'].values

# Normalize by the peak of Phase A to get 1.0 pu
v_peak = np.max(np.abs(va_raw))
va, vb, vc = va_raw / v_peak, vb_raw / v_peak, vc_raw / v_peak
v_abc = np.array([va, vb, vc])

# 4. Extract and Normalize Voltages (Per-Unit)

v_raw_a = df['Report_1:V_R_5'].values
v_raw_b = df['Report_1:V_S_6'].values
v_raw_c = df['Report_1:V_T_7'].values


v_a = v_raw_a / v_peak
v_b = v_raw_b / v_peak
v_c = v_raw_c / v_peak

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
    wc = 2 * np.pi * 20 #natural angular frequency
    zeta = 0.707        # Damping Ratio: determines the stability and overshoot of the PLL when settling
    Kp, Ki = 2 * zeta * wc, wc**2  # Proportinal Gain and Integral gain
    theta_hat, integral_error = (np.pi/6 - np.pi/2), 0.0  #initialized
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
    Gamma = 50
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
    v_prime = np.gradient(v_abc, axis=1) * fs
    v_double_prime = np.gradient(v_prime, axis=1) * fs

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


def filter_frequency_moving_median(omega_kappa, omega_a, f=50.0, window_size=21):
    """
    Apply moving median filter to frequency data.
    
    Parameters
    ----------
    omega_kappa : array_like
        Frenet frame angular frequency [rad/s]
    omega_a : array_like
        Affine angular frequency [rad/s]
    f : float, optional
        Nominal frequency [Hz]. Default is 50 Hz.
    window_size : int, optional
        Median filter window size. Default is 11 (must be odd).
        Larger window = more smoothing.
        Recommended: 9, 11, 13, 15, 21

    Returns
    -------
    w_filter : ndarray
        Filtered Frenet frequency (per-unit)
    w_filter_affine : ndarray
        Filtered affine frequency (per-unit)
    """
    
    # Nominal angular frequency
    w0 = 2 * np.pi * f
    
    # Convert to 1D arrays
    w = np.asarray(omega_kappa, dtype=float).ravel()
    w_affine = np.asarray(omega_a, dtype=float).ravel()
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Check array lengths match
    if len(w) != len(w_affine):
        print(f"WARNING: Array length mismatch! omega_kappa: {len(w)}, omega_a: {len(w_affine)}")
        min_len = min(len(w), len(w_affine))
        w = w[:min_len]
        w_affine = w_affine[:min_len]
    
    # Convert frequency to per-unit
    w_geom = w / w0
    w_affine_geom = w_affine / w0
    
    # Apply moving median filter (simple one-liner!)
    w_filtered = medfilt(w_geom, kernel_size=window_size)
    w_affine_filtered = medfilt(w_affine_geom, kernel_size=window_size)
    
    return w_filtered, w_affine_filtered

# ==========================================
# 7. MAIN CONTROL CODE (RUN ALL METHODS)
# ==========================================
srf_freq, srf_pu = method_srf_pll(v_abc)
dsogi_freq, dsogi_pu = method_dsogi_fll(v_abc)
dft_freq, dft_pu = dft_estimate_frequency(v_abc)
omega_kappa, omega_a = method_frenet_frame(v_abc)
w_filtered, w_affine_filtered = filter_frequency_moving_median(omega_kappa, omega_a, f=f_o, window_size=21)
    


# ==========================================
# 8. FINAL VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(t, srf_pu, label='SRF-PLL')
plt.plot(t, dsogi_pu, label='DSOGI-FLL')
# --- Symmetrical Time Vector Alignment ---
# Calculate total lost samples due to 'valid' modes
total_lost = len(v_abc[0]) - len(dft_pu)
# Distribute the lost samples equally at the start and end
start_idx = total_lost // 2
end_idx = start_idx + len(dft_pu)

t_dft = t[start_idx:end_idx]
plt.plot(t_dft, dft_pu, label='3-Level DFT', linestyle='--')
plt.plot(t, w_filtered, label='Frenet Frame', linestyle=':')
plt.axhline(y=1, color='r', linestyle='-.', label='Target (pu)')
plt.title('Comparison of 4 Frequency Estimation Methods')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (pu)')
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
axs[1,0].set_ylim(-2, 2)
axs[1,0].grid(True, alpha=0.3)

# Method 2: DSOGI-FLL
axs[2,0].plot(t, dsogi_pu, label='DSOGI-FLL', color='orange')
axs[2,0].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axs[2,0].set_title('Method 2: DSOGI-FLL')
axs[2,0].set_ylabel('Frequency (pu)')
axs[2,0].set_ylim(-2, 2)
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
axs[0,1].set_ylim(-2, 2)
axs[0,1].grid(True, alpha=0.3)

# Method 4: Frenet Frame
#axs[4].plot(t, frenet_pu[0, :], label='Frenet Frame', color='purple')
axs[1,1].plot(t, w_filtered, label='Frenet Frame', color='purple')
axs[1,1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axs[1,1].set_title('Method 4: Frenet Frame Method1')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_ylabel('Frequency (pu)')
axs[1,1].set_ylim(-2, 2)
axs[1,1].grid(True, alpha=0.3)

# Method 5: Affine Differentiation (Frenet Frame)
axs[2,1].plot(t, w_affine_filtered, label='Affine Differentiation', color='red')
axs[2,1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axs[2,1].set_title('Method 5: Affine Differentiation')
axs[2,1].set_xlabel('Time (s)')
axs[2,1].set_ylabel('Frequency (pu)')
axs[2,1].set_ylim(-2, 2)
axs[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()