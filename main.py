import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

# --- 1. Sistem ve Hedef Parametreleri ---
fs, T_chirp, B, fc = 2e6, 0.25e-3, 150e6, 24e9
L, N = 64, int(fs * T_chirp)
slope, dt = B / T_chirp, L * T_chirp
snr_db = 20 

target_defs = [
    {"r0": 25.0, "v": 4.5,  "color": 'blue',   "name": "Hedef 1"},
    {"r0": 50.0, "v": -2.0, "color": 'green',  "name": "Hedef 2"},
    {"r0": 75.0, "v": 8.5,  "color": 'red',    "name": "Hedef 3"}
]

# --- 2. Yardımcı Fonksiyonlar ve Kalman Sınıfı ---
def get_crlb_std(snr_db, B, fc, L, T_chirp):
    snr_lin = 10**(snr_db / 10)
    std_r = c / (2 * np.pi * B * np.sqrt(2 * snr_lin))
    std_v = c / (2 * np.pi * fc * (L * T_chirp) * np.sqrt(2 * snr_lin))
    return std_r, std_v

class KalmanTracker:
    def __init__(self, r0, v0):
        self.x = np.array([[r0], [v0]])
        self.P = np.eye(2) * 5.0
        self.F = np.array([[1, dt], [0, 1]])
        self.Q = np.eye(2) * 0.005 # Process Noise
        self.R = np.eye(2) * 0.1   # Measurement Noise
        self.innovation = np.zeros((2, 1))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Inovasyon hesaplama (z - Hx)
        self.innovation = z.reshape(2,1) - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ self.innovation
        self.P = (np.eye(2) - K) @ self.P

# --- 3. Simülasyon Döngüsü ---
steps = 60
trackers = [KalmanTracker(t["r0"], t["v"]) for t in target_defs]
std_r_limit, std_v_limit = get_crlb_std(snr_db, B, fc, L, T_chirp)

# Veri Kaydı
history = {i: {"gt_r":[], "gt_v":[], "kf_r":[], "kf_v":[], "mle_err_r":[], 
               "innov_r":[], "innov_v":[], "P_r":[], "P_v":[]} for i in range(3)}
rmse_mle, rmse_kf = [], []

for s in range(steps):
    data = np.zeros((N, L), dtype=complex)
    for i, t in enumerate(target_defs):
        curr_r = t["r0"] + t["v"] * (s * dt)
        history[i]["gt_r"].append(curr_r); history[i]["gt_v"].append(t["v"])
        # Sinyal Sentezi
        t_vec = np.linspace(0, T_chirp, N)
        for l in range(L):
            phi = 2 * np.pi * fc * (2 * (curr_r + t["v"] * l * T_chirp) / c)
            data[:, l] += np.exp(1j*(2*np.pi*(2*slope*curr_r/c)*t_vec + phi))
    
    # Gürültü Ekleme
    data += (np.random.normal(0,1,(N,L)) + 1j*np.random.normal(0,1,(N,L))) / np.sqrt(2 * 10**(snr_db/10))
    rd_map = np.abs(np.fft.fftshift(np.fft.fft2(data), axes=1))

    sq_err_mle, sq_err_kf = [], []

    for i in range(3):
        trackers[i].predict()
        # MLE Kestirimi (Local Search)
        r_bin = int((history[i]["gt_r"][-1]*2*slope/c)*N/fs)
        v_bin = int((history[i]["gt_v"][-1]*2*fc/c)*L*T_chirp + L/2)
        sub = rd_map[max(0,r_bin-4):min(N,r_bin+4), max(0,v_bin-4):min(L,v_bin+4)]
        loc = np.unravel_index(np.argmax(sub), sub.shape)
        
        m_r = ((r_bin-4+loc[0])/N)*fs*c/(2*slope)
        m_v = ((v_bin-4+loc[1]-L/2)/(L*T_chirp))*c/(2*fc)
        
        # Güncelleme ve Kayıt
        trackers[i].update(np.array([m_r, m_v]))
        history[i]["kf_r"].append(trackers[i].x[0,0]); history[i]["kf_v"].append(trackers[i].x[1,0])
        history[i]["mle_err_r"].append(abs(m_r - history[i]["gt_r"][-1]))
        history[i]["innov_r"].append(trackers[i].innovation[0,0])
        history[i]["innov_v"].append(trackers[i].innovation[1,0])
        history[i]["P_r"].append(trackers[i].P[0,0])
        history[i]["P_v"].append(trackers[i].P[1,1])
        
        sq_err_mle.append((m_r - history[i]["gt_r"][-1])**2)
        sq_err_kf.append((trackers[i].x[0,0] - history[i]["gt_r"][-1])**2)

    rmse_mle.append(np.sqrt(np.mean(sq_err_mle)))
    rmse_kf.append(np.sqrt(np.mean(sq_err_kf)))

# --- 4. Görselleştirme: FİGÜR 1 (Takip ve Performans) ---
fig1, axs1 = plt.subplots(2, 2, figsize=(16, 10))
fig1.suptitle("FMCW Radar: Çoklu Hedef Takibi ve Performans Analizi", fontsize=16)

# A. Mesafe Takibi
for i, t in enumerate(target_defs):
    axs1[0, 0].plot(history[i]["gt_r"], 'k--', alpha=0.5, label="Gerçek" if i==0 else "")
    axs1[0, 0].plot(history[i]["kf_r"], color=t["color"], lw=2, label=f"{t['name']} Takip")
axs1[0, 0].set_title("Mesafe: Gerçek vs Takip"); axs1[0, 0].legend(); axs1[0, 0].grid(True)

# B. Hız Takibi
for i, t in enumerate(target_defs):
    axs1[0, 1].plot(history[i]["gt_v"], 'k--', alpha=0.5)
    axs1[0, 1].plot(history[i]["kf_v"], color=t["color"], lw=2)
axs1[0, 1].set_title("Hız: Gerçek vs Takip"); axs1[0, 1].grid(True)

# C. Mesafe Hatası vs CRLB
for i, t in enumerate(target_defs):
    axs1[1, 0].plot(history[i]["mle_err_r"], 'o', color=t["color"], ms=4, alpha=0.5)
axs1[1, 0].axhline(y=std_r_limit, color='black', lw=2, label=f"CRLB Alt Sınırı ({std_r_limit:.4f}m)")
axs1[1, 0].set_title("Mesafe MLE Hatası ve CRLB"); axs1[1, 0].legend(); axs1[1, 0].grid(True)

# D. RMSE vs Zaman
axs1[1, 1].plot(rmse_mle, 'r-o', ms=4, alpha=0.6, label="MLE RMSE (Ham)")
axs1[1, 1].plot(rmse_kf, 'b-s', ms=4, alpha=0.8, label="Kalman RMSE (Filtreli)")
axs1[1, 1].axhline(y=std_r_limit, color='black', lw=2, ls='--', label="CRLB Limit")
axs1[1, 1].set_title("RMSE Karşılaştırması"); axs1[1, 1].legend(); axs1[1, 1].grid(True)

# --- 5. Görselleştirme: FİGÜR 2 (Filtre Teşhisi) ---
fig2, axs2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle("Kalman Filtresi Teşhis ve Kararlılık Analizi", fontsize=16)

# A. Mesafe İnovasyonu
for i, t in enumerate(target_defs):
    axs2[0, 0].plot(history[i]["innov_r"], color=t["color"], alpha=0.7, label=t['name'])
axs2[0, 0].axhline(0, color='black', ls='--')
axs2[0, 0].set_title("Mesafe İnovasyonu (Residuals)"); axs2[0, 0].legend(); axs2[0, 0].grid(True)

# B. Hız İnovasyonu
for i, t in enumerate(target_defs):
    axs2[0, 1].plot(history[i]["innov_v"], color=t["color"], alpha=0.7)
axs2[0, 1].axhline(0, color='black', ls='--')
axs2[0, 1].set_title("Hız İnovasyonu (Residuals)"); axs2[0, 1].grid(True)

# C. P Matrisi Yakınsaması (Mesafe)
for i, t in enumerate(target_defs):
    axs2[1, 0].plot(history[i]["P_r"], color=t["color"], lw=2)
axs2[1, 0].set_title("Mesafe Belirsizliği Yakınsaması ($P_{11}$)"); axs2[1, 0].set_ylabel("Varyans"); axs2[1, 0].grid(True)

# D. P Matrisi Yakınsaması (Hız)
for i, t in enumerate(target_defs):
    axs2[1, 1].plot(history[i]["P_v"], color=t["color"], lw=2)
axs2[1, 1].set_title("Hız Belirsizliği Yakınsaması ($P_{22}$)"); axs2[1, 1].set_ylabel("Varyans"); axs2[1, 1].grid(True)

plt.tight_layout()
plt.show()