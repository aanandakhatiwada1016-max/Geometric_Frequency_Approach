import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==========================================
# 1. GLOBAL PARAMETERS
# ==========================================
f_o = 50.0  

omega_o = 2 * np.pi * f_o    # Nominal Angular Frequency [rad/s]
fs = 10000                   # Sampling Frequency [Hz]
Ts = 1/fs
t_end = 4
t = np.arange(0, t_end, 1/fs) # 500ms simulation

alpha = 2 * np.pi / 3
N0 = fs/f_o

V_base = 15.0


# ==========================================
# 2. INPUT SELECTION 
# ==========================================

# --- Case E1: Balanced, Stationary 
V, omega_a = 15.0, omega_o
v_a = V * np.sin(omega_a * t + np.pi/6)
v_b = V * np.sin(omega_a * t + np.pi/6 - alpha)
v_c = V * np.sin(omega_a * t + np.pi/6 + alpha)

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

# --- Case E4: Frequency Modulation (Continuous Sine Variation) ---
#fm = 10.0   # Modulation frequency = 10 Hz
#A = 1.0     # Frequency deviation amplitude = +/- 1 Hz
#V= 15
     # Voltage amplitude = 1 pu
# 1. Correct phase calculation
#theta = omega_o * t - (A / fm) * np.cos(2 * np.pi * fm * t) + (np.pi / 6)

# 2. Balanced 3-phase voltages
#v_a = V * np.sin(theta)
#v_b = V * np.sin(theta - 2*np.pi/3)
#v_c = V * np.sin(theta + 2*np.pi/3)
#v_abc = np.vstack([v_a, v_b, v_c])

# 3. True frequency signal for comparison
#f = 50.0 + A * np.sin(2 * np.pi * fm * t)

# --- Case E5 : Ramp Magnitude + Ramp Frequency

# Voltage: 10 -> 15 V in first 4 seconds
#V = np.where(t < 2,10 + (15-10)*t/2,15)

# Frequency: 49 -> 52 Hz in first 4 seconds,
# then back to 50 Hz
#f = np.where(t < 2,49 + (52-49)*t/2,52 - (2/2)*(t-2))

#theta = np.cumsum(2*np.pi*f/fs)

#v_a = V*np.sin(theta + np.pi/6)
#v_b = V*np.sin(theta + np.pi/6 - alpha)
#v_c = V*np.sin(theta + np.pi/6 + alpha)

# --- Case E6: Unbalanced --- 
#V, omega_a = 15.0, omega_o
#v_a = 1.0 * V * np.sin(omega_a * t + np.pi/6)
#v_b = 1.2 * V * np.sin(omega_a * t + np.pi/6 - alpha)
#v_c = 0.8 * V * np.sin(omega_a * t + np.pi/6 + alpha)

#--- Case E7: Harmonic Content
#V, omega_a = 15.0, omega_o

#v_a = V * ((np.sin(omega_a * t + np.pi/6)) + (0.1 * np.sin(5 * (omega_a * t + np.pi/6))))
#v_b = V * ((np.sin(omega_a * t +np.pi/6- alpha)) + (0.2 * np.sin(5 * (omega_a * t +np.pi/6 - alpha))))
#v_c = V * ((np.sin(omega_a * t +np.pi/6+ alpha)) + (0.1 * np.sin(5 * (omega_a * t +np.pi/6+ alpha))))


#... FREQUENCY IN ARRAY
if np.isscalar(omega_a):
   f = np.full_like(t, omega_a/(2*np.pi))
else:
     f = omega_a/(2*np.pi)


#VOLTAGE IN ARRAY
v_abc = np.array([v_a, v_b, v_c])

# ==========================================
# 3. METHOD 1: SRF-PLL
# ==========================================
def method_srf_pll(v_abc):
    wc = 2 * np.pi * 20 #natural angular frequency
    zeta = 0.707        # Damping Ratio: determines the stability and overshoot of the PLL when settling
    Kp, Ki = 2 * zeta * wc, wc**2  # Proportinal Gain and Integral gain
    
    theta_hat = 0.0
    integral_error = 0.0  #initialized
    srf_freq = []
    srf_pu = []

    for i in range(len(t)):
        va, vb, vc = v_abc[:, i]
        # Clarke & Park
        v_alpha = np.sqrt(2/3) * (va - 0.5*vb - 0.5*vc)
        v_beta = np.sqrt(2/3) * (np.sqrt(3)/2 * (vb - vc))
        vd = v_alpha * np.cos(theta_hat) + v_beta * np.sin(theta_hat)
        vq = -v_alpha * np.sin(theta_hat) + v_beta * np.cos(theta_hat)
        # --- FIX A: Voltage Normalization (p.u. conversion) ---
        v_mag = np.sqrt(v_alpha**2 + v_beta**2)
        vq_pu = vq / v_mag if v_mag > 1e-6 else 0.0
        # PI Loop
        integral_error += vq_pu * Ts
        omega_hat = omega_o + Kp * vq_pu + Ki * integral_error
        theta_hat = np.mod(theta_hat + omega_hat * Ts, 2 * np.pi)
        # Clamp estimated angular frequency to realistic grid limits (e.g., 40 Hz to 70 Hz)
        omega_min = 2 * np.pi * 40.0
        omega_max = 2 * np.pi * 70.0
        omega_hat = np.clip(omega_o + Kp * vq_pu + Ki * integral_error, omega_min, omega_max)

        srf_freq.append(omega_hat / (2 * np.pi))
        #srf_freq.append ((omega_hat - omega_o) / omega_o)
        srf_pu.append ((omega_hat - omega_o) / omega_o)

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


        w_pu = (w_est - omega_o) / omega_o
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
    w_pu = (2*np.pi*f_est - omega_o) / omega_o
    
    return f_est, w_pu
    


# ==========================================
# 6. METHOD 4: FRENET FRAME
# ==========================================

def method_frenet_frame(v_abc):
    v_prime = np.gradient(v_abc, axis=1) * fs
    v_double_prime = np.gradient(v_prime, axis=1) * fs

    cross_vp_vpp = np.cross(v_prime, v_double_prime,axis=0)
    cross_v_vp = np.cross(v_abc, v_prime, axis=0)
    mag_v = np.linalg.norm(v_abc, axis=0)
    mag_cross = np.linalg.norm(cross_v_vp, axis=0)

    omega_kappa = mag_cross / (mag_v**2 + 1e-9)
    frenet_freq = omega_kappa / (2 * np.pi)
    frenet_pu = (omega_kappa - omega_o) / omega_o

    #revised approach using omega_a from the Frenet frame method for unbalanced case
    #omega_aff = np.sqrt(cross_vp_vpp/cross_v_vp)
    #frenet_freq = omega_aff / (2 * np.pi)
    #frenet_pu = (omega_aff - omega_o) / omega_o

    return frenet_pu, frenet_freq



def evaluate_frequency_estimator(f_true, f_est, t, tolerance=0.02):

    error = np.abs(f_est - f_true)

    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(error)

    settling_time = np.nan

    hold = 100      # Require 100 consecutive samples

    for i in range(len(error)-hold):
        if np.all(error[i:i+hold] < tolerance):
            settling_time = t[i]
            break

    return rmse, max_error, settling_time
# ==========================================
# 7. MAIN CONTROL CODE (RUN ALL METHODS)
# ==========================================
srf_freq, srf_pu = method_srf_pll(v_abc)
dsogi_freq, dsogi_pu = method_dsogi_fll(v_abc)
dft_freq, dft_pu = dft_estimate_frequency(v_abc)
frenet_pu, frenet_freq = method_frenet_frame(v_abc)



PLL     = evaluate_frequency_estimator(f, srf_freq, t)
FLL     = evaluate_frequency_estimator(f, dsogi_freq, t)
FRENET  = evaluate_frequency_estimator(f, frenet_freq, t)

delay_samples = len(f) - len(dft_freq)
delay_time = delay_samples / fs

f_true_dft = f[delay_samples:]

dft_rmse = np.sqrt(np.mean((dft_freq - f_true_dft)**2))
dft_max = np.max(np.abs(dft_freq - f_true_dft))

DFT = (dft_rmse, dft_max, delay_time)

# ==========================================
# 8. FINAL EVALUATION AND COMPARISON
# ==========================================
print("-"*85)

print("{:<15} {:>10} {:>12} {:>18}".format(
    "Method", "RMSE", "MaxErr", "Settling Time (s)"
))

print("-"*85)

for name, result in zip(
    ["PLL", "FLL", "Frenet", "3-Level DFT"],
    [PLL, FLL, FRENET, DFT]
):

    print("{:<15} {:10.4f} {:12.4f} {:18.4f}".format(
        name,
        result[0],
        result[1],
        result[2]
    ))

print("-"*85)


#Visualization of frequency deviation for all methods
# Calculate total lost samples due to 'valid' modes
total_lost = len(v_abc[0]) - len(dft_pu)
# Distribute the lost samples equally at the start and end
start_idx = total_lost // 2
end_idx = start_idx + len(dft_pu)
t_dft = t[start_idx:end_idx]

# After running Case E1: (Change Here after each case changes)
srf_pu_E1, dsogi_pu_E1 = srf_pu, dsogi_pu
dft_pu_E1, frenet_pu_E1 = dft_pu, frenet_pu
t_dft_E1 = t_dft
f_true_E1 = f

# ══════════════════════════════════════════════════════
# GLOBAL SETTINGS — apply once at the top
# ══════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':      'Times New Roman',
    'font.size':        10,
    'axes.titlesize':   15,
    'axes.labelsize':   20,
    'legend.fontsize':  15,
    'xtick.labelsize':  15,
    'ytick.labelsize':  15,
    'lines.linewidth':  1.5,
    'grid.alpha':       0.3,
    'figure.dpi':       150,
})

styles = {
    'SRF-PLL':      {'color': '#1f77b4', 'linestyle': '-',  'linewidth': 1.8},
    'DSOGI-FLL':    {'color': '#ff7f0e', 'linestyle': '--', 'linewidth': 1.8},
    '3-Level DFT':  {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': 1.8},
    'Frenet Frame': {'color': "#EB1313", 'linestyle': ':',  'linewidth': 2.2},
}

def add_all_methods(ax, t, srf, dsogi, dft, frenet, t_dft):
    ax.plot(t,     srf,    label='SRF-PLL',      **styles['SRF-PLL'])
    ax.plot(t,     dsogi,  label='DSOGI-FLL',    **styles['DSOGI-FLL'])
    ax.plot(t_dft, dft,    label='3-Level DFT',  **styles['3-Level DFT'])
    ax.plot(t,     frenet, label='Frenet Frame', **styles['Frenet Frame'])
def save_fig(fig, name):
    fig.tight_layout(pad=1.5)
    fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
 
    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════
# CASE-PLOT - By changing After running each case 
# ══════════════════════════════════════════════════════

# E1a — Full simulation
fig, ax = plt.subplots(figsize=(10, 3.5))
add_all_methods(ax, t, srf_pu_E1, dsogi_pu_E1, dft_pu_E1, frenet_pu_E1, t_dft_E1)
ax.axhline(0, color='k', linestyle='-.', linewidth=1)
ax.set_xlim(0, 0.5);  ax.set_ylim(-0.05, 0.05)
#ax.set_title('Case E1: Balanced Nominal Operation — Full Simulation (0–4 s)', pad=8)
ax.set_ylabel('Freq. Deviation (pu)');  ax.set_xlabel('Time (s)')
ax.legend(loc='upper right', ncol=1, framealpha=0.9)
ax.grid(True);  ax.yaxis.set_tick_params(pad=4)
save_fig(fig, 'E1a_full')

# E1b — Transient + Steady state
fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(11, 4),
                                gridspec_kw={'wspace': 0.38})
add_all_methods(ax2, t, srf_pu_E1, dsogi_pu_E1, dft_pu_E1, frenet_pu_E1, t_dft_E1)
ax2.axhline(0, color='k', linestyle='-.', linewidth=1)
ax2.axvline(0.079, color='#1f77b4', linestyle=':', linewidth=1.3, label='PLL settled (0.079 s)')
ax2.axvline(0.024, color='#ff7f0e', linestyle=':', linewidth=1.3, label='FLL settled (0.024 s)')
ax2.set_xlim(0, 0.20);  ax2.set_ylim(-0.32, 0.18)
ax2.set_title('(a)  Transient Detail (0–0.20 s)', pad=8)
ax2.set_ylabel('Freq. Deviation (pu)');  ax2.set_xlabel('Time (s)')
ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax2.grid(True);  ax2.yaxis.set_tick_params(pad=4)

add_all_methods(ax3, t, srf_pu_E1, dsogi_pu_E1, dft_pu_E1, frenet_pu_E1, t_dft_E1)
ax3.axhline(0, color='k', linestyle='-.', linewidth=1)
ax3.set_xlim(3.5, 4.0);  ax3.set_ylim(-0.005, 0.005)
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax3.set_title('(b)  Steady-State Detail (3.5–4.0 s)', pad=8)
ax3.set_ylabel('Freq. Deviation (pu)');  ax3.set_xlabel('Time (s)')
ax3.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax3.grid(True);  ax3.yaxis.set_tick_params(pad=4)
save_fig(fig, 'E1b_detail')
