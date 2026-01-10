import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

# --- 1. Parametreler ---
fs, T_chirp, B, fc = 2e6, 0.25e-3, 150e6, 24e9
L, N = 64, int(fs * T_chirp)
slope, dt = B / T_chirp, L * T_chirp
snr_db = 12 
num_trials = 10 

target_defs = [
    {"r0": 25.0, "v": 4.5,  "color": 'blue',   "name": "H1"},
    {"r0": 50.0, "v": -2.0, "color": 'green',  "name": "H2"},
    {"r0": 75.0, "v": 8.5,  "color": 'red',    "name": "H3"}
]

# --- 2. Fonksiyonlar ve Kalman Sınıfı ---
def get_crlb_std(snr_db, B, fc, L, T_chirp):
    snr_lin = 10**(snr_db / 10)
    return c / (2 * np.pi * B * np.sqrt(2 * snr_lin))

class KalmanTracker:
    def __init__(self, r0, v0):
        self.x = np.array([[r0], [v0]])
        self.P = np.eye(2) * 5.0
        self.F = np.array([[1, dt], [0, 1]])
        self.Q = np.eye(2) * 0.01 
        self.R = np.eye(2) * 0.1   
        self.innov = np.zeros((2,1))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        self.innov = z.reshape(2,1) - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z.reshape(2,1) - self.x)
        self.P = (np.eye(2) - K) @ self.P

# --- 3. Veri Yapıları ---
steps = 60
std_r_limit = get_crlb_std(snr_db, B, fc, L, T_chirp)

mle_r_all = np.zeros((num_trials, len(target_defs), steps))
map_r_all = np.zeros((num_trials, len(target_defs), steps))
mle_v_all = np.zeros((num_trials, len(target_defs), steps))
map_v_all = np.zeros((num_trials, len(target_defs), steps))
mle_err_all = np.zeros((num_trials, len(target_defs), steps))
map_err_all = np.zeros((num_trials, len(target_defs), steps))
mle_P_all = np.zeros((num_trials, len(target_defs), steps))
map_P_all = np.zeros((num_trials, len(target_defs), steps))

# --- 4. Monte Carlo Simülasyonu ---
print(f"Simülasyon başladı: {num_trials} deneme yapılıyor...")
for tr in range(num_trials):
    trackers_mle = [KalmanTracker(t["r0"], t["v"]) for t in target_defs]
    trackers_map = [KalmanTracker(t["r0"], t["v"]) for t in target_defs]
    
    for s in range(steps):
        data = np.zeros((N, L), dtype=complex)
        for i, t in enumerate(target_defs):
            curr_r = t["r0"] + t["v"] * (s * dt)
            t_vec = np.linspace(0, T_chirp, N)
            for l in range(L):
                phi = 2 * np.pi * fc * (2 * (curr_r + t["v"] * l * T_chirp) / c)
                data[:, l] += np.exp(1j*(2*np.pi*(2*slope*curr_r/c)*t_vec + phi))
        
        data += (np.random.normal(0,1,(N,L)) + 1j*np.random.normal(0,1,(N,L))) / np.sqrt(2 * 10**(snr_db/10))
        rd_map = np.abs(np.fft.fftshift(np.fft.fft2(data), axes=1))

        for i in range(len(target_defs)):
            xm_p, _ = trackers_mle[i].predict()
            xa_p, Pa_p = trackers_map[i].predict()

            gt_r = target_defs[i]["r0"] + target_defs[i]["v"] * (s * dt)
            r_bin = int((gt_r*2*slope/c)*N/fs)
            v_bin = int((target_defs[i]["v"]*2*fc/c)*L*T_chirp + L/2)
            
            window = rd_map[max(0,r_bin-10):min(N,r_bin+10), v_bin-2:v_bin+2]
            r_axis = np.linspace((max(0,r_bin-10))/N*fs*c/(2*slope), (min(N,r_bin+10)-1)/N*fs*c/(2*slope), window.shape[0])

            mle_r = r_axis[np.argmax(window.max(axis=1))]
            prior_var = Pa_p[0,0] + 0.5 
            prior = np.exp(-((r_axis - xa_p[0,0])**2) / (2 * prior_var))
            map_r = r_axis[np.argmax(window.max(axis=1) * prior)]

            trackers_mle[i].update(np.array([mle_r, target_defs[i]["v"]]))
            trackers_map[i].update(np.array([map_r, target_defs[i]["v"]]))

            mle_r_all[tr, i, s] = trackers_mle[i].x[0,0]
            map_r_all[tr, i, s] = trackers_map[i].x[0,0]
            mle_v_all[tr, i, s] = trackers_mle[i].x[1,0]
            map_v_all[tr, i, s] = trackers_map[i].x[1,0]
            mle_err_all[tr, i, s] = abs(mle_r - gt_r)
            map_err_all[tr, i, s] = abs(map_r - gt_r)
            mle_P_all[tr, i, s] = trackers_mle[i].P[0,0]
            map_P_all[tr, i, s] = trackers_map[i].P[0,0]

# --- 5. Ortalamalar ---
mean_mle_r, mean_map_r = np.mean(mle_r_all, axis=0), np.mean(map_r_all, axis=0)
mean_mle_v, mean_map_v = np.mean(mle_v_all, axis=0), np.mean(map_v_all, axis=0)
mean_mle_err, mean_map_err = np.mean(mle_err_all, axis=0), np.mean(map_err_all, axis=0)
mean_mle_P, mean_map_P = np.mean(mle_P_all, axis=0), np.mean(map_P_all, axis=0)

# --- 6. GÖRSELLEŞTİRME ---
fig1, axs1 = plt.subplots(2, 2, figsize=(16, 11))
fig1.suptitle(f"FMCW Radar: MLE vs MAP Monte Carlo Analizi ({num_trials} Deneme Ortalaması)", fontsize=14)

for i, t in enumerate(target_defs):
    gt_r_path = [t["r0"] + t["v"] * (s * dt) for s in range(steps)]
    # Sol Üst: Mesafe
    axs1[0,0].plot(gt_r_path, 'k--', alpha=0.3, label="Gerçek" if i==0 else "")
    axs1[0,0].plot(mean_mle_r[i], color=t['color'], ls=':', label=f"{t['name']} MLE-KF")
    axs1[0,0].plot(mean_map_r[i], color=t['color'], ls='-', lw=2, label=f"{t['name']} MAP-KF")
    axs1[0,0].set_xlabel("Adım (Step)"); axs1[0,0].set_ylabel("Mesafe (m)")
    
    # Sağ Üst: Hız
    axs1[0,1].plot([t["v"]]*steps, 'k--', alpha=0.3, label="Gerçek" if i==0 else "")
    axs1[0,1].plot(mean_mle_v[i], color=t['color'], ls=':', alpha=0.5,  label=f"{t['name']} MLE-KF")
    axs1[0,1].plot(mean_map_v[i], color=t['color'], ls='-', alpha=0.8 , label=f"{t['name']} MAP-KF")
    axs1[0,1].set_xlabel("Adım (Step)"); axs1[0,1].set_ylabel("Hız (m/s)")

    # Sol Alt: Hata
    axs1[1,0].plot(mean_mle_err[i], color=t['color'], ls=':', alpha=0.3,label=f"{t['name']} MAP-KF")
    axs1[1,0].plot(mean_map_err[i], color=t['color'], ls='-', alpha=0.6,label=f"{t['name']} MLE-KF")
    axs1[1,0].set_xlabel("Adım (Step)"); axs1[1,0].set_ylabel("Ortalama Hata (m)")

axs1[1,0].axhline(y=std_r_limit, color='black', lw=2, label=f"CRLB ({std_r_limit:.3f}m)")

# Sağ Alt: RMSE
axs1[1,1].plot(np.mean(mean_mle_err, axis=0), 'r--', label="Avg MLE RMSE")
axs1[1,1].plot(np.mean(mean_map_err, axis=0), 'b-', label="Avg MAP RMSE")
axs1[1,1].set_xlabel("Adım (Step)"); axs1[1,1].set_ylabel("RMSE (m)")
axs1[1,1].set_title("Sistem Geneli Ortalama Hata")

for ax in axs1.flat: ax.grid(True); ax.legend(fontsize='x-small')

# FIG 2: Kovaryans
fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Filtre Sağlığı: Kovaryans Yakınsaması", fontsize=14)
for i, t in enumerate(target_defs):
    axs2[0].plot(mean_mle_P[i], color=t['color'], ls=':')
    axs2[1].plot(mean_map_P[i], color=t['color'], ls='-')

axs2[0].set_title("Kovaryans (MLE-KF)"); axs2[1].set_title("Kovaryans (MAP-KF)")
for ax in axs2: 
    ax.set_xlabel("Adım (Step)"); ax.set_ylabel("Varyans ($m^2$)")
    ax.grid(True)

plt.tight_layout()
plt.show()