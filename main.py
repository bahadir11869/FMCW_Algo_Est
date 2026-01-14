import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import os  # Klasör işlemleri için eklendi

# --- 1. PARAMETRELER ---
fs, T_chirp, B, fc = 2e6, 50e-6, 150e6, 24e9
N = int(fs * T_chirp)
slope = B / T_chirp
dt = 0.1
steps = 100

# --- 2. KALMAN FİLTRESİ ---
class KalmanTracker:
    def __init__(self, x0, P0, Q, R, dt):
        self.x = x0
        self.P = P0
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = Q
        self.R = R
        self.K = np.zeros((2,1))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0,0], self.x[1,0]

    def update(self, z):
        y = z - self.H @ self.x 
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.K = K
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0,0], self.x[1,0], y[0,0], np.sqrt(self.P[0,0])

# --- 3. SİNYAL İŞLEME VE CRLB ---
def calculate_crlb_range(snr_db):
    snr_lin = 10**(snr_db/10)
    sigma_f = (np.sqrt(12) * fs) / (2 * np.pi * np.sqrt(snr_lin * N * (N**2 - 1)))
    sigma_r = (c * sigma_f) / (2 * slope)
    return sigma_r

def generate_fmcw_signal(r_target, snr_db):
    t = np.linspace(0, T_chirp, N, endpoint=False)
    f_beat = (2 * slope * r_target) / c
    sig = np.exp(1j * 2 * np.pi * f_beat * t)
    pwr = 1.0 / (10**(snr_db/10))
    noise = (np.random.normal(0, np.sqrt(pwr/2), N) + 
             1j * np.random.normal(0, np.sqrt(pwr/2), N))
    return sig + noise

def extract_range_mle(signal):
    fft_data = np.abs(np.fft.fft(signal))[:N//2]
    peak_idx = np.argmax(fft_data)
    if 0 < peak_idx < len(fft_data) - 1:
        y1, y2, y3 = fft_data[peak_idx-1], fft_data[peak_idx], fft_data[peak_idx+1]
        denom = y1 - 2*y2 + y3
        delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0
        true_idx = peak_idx + delta
    else:
        true_idx = peak_idx
    return (true_idx / N) * fs * c / (2 * slope)

# --- 4. YARDIMCI FONKSİYONLAR ---
def calculate_rolling_rmse(errors, window_size=10):
    sq_errors = np.square(errors)
    kernel = np.ones(window_size) / window_size
    rolling_mean_sq = np.convolve(sq_errors, kernel, mode='same')
    return np.sqrt(rolling_mean_sq)

def print_report(title, logs, crlb_val, snr):
    rmse_mle = np.sqrt(np.mean(np.square(logs['err_pos_mle'])))
    rmse_map = np.sqrt(np.mean(np.square(logs['err_pos_map'])))
    rmse_vel_mle = np.sqrt(np.mean(np.square(logs['err_vel_mle'][5:])))
    rmse_vel_map = np.sqrt(np.mean(np.square(logs['err_vel_map'][5:])))
    
    improvement_percent = (1 - (rmse_map / rmse_mle)) * 100
    crlb_ratio = rmse_map / crlb_val
    innov_std = np.std(logs['innov'])
    avg_sigma = np.mean(logs['sigma'])
    consistency = innov_std / avg_sigma

    print(f"\n--- {title} ---")
    print(f"SNR (dB):              {snr}")
    print(f"CRLB (m):              {crlb_val:.6f}")
    print(f"RMSE Pos MLE (m):      {rmse_mle:.6f}")
    print(f"RMSE Pos MAP (m):      {rmse_map:.6f}")
    print(f"Improvement (%):       {improvement_percent:.4f}")
    print(f"CRLB Ratio (x):        {crlb_ratio:.4f}")
    print(f"RMSE Vel MLE (m/s):    {rmse_vel_mle:.6f}")
    print(f"RMSE Vel MAP (m/s):    {rmse_vel_map:.6f}")
    print(f"Consistency (NIS):     {consistency:.4f}")
    print("-" * 30)

# --- 5. SİMÜLASYON MOTORU ---
def run_scenario(snr, is_maneuver, scen_name):
    true_pos, true_vel = [], []
    curr_r, curr_v = 20.0, (5.0 if is_maneuver else 2.0)
    
    for i in range(steps):
        if is_maneuver and i == 40: curr_v = -5.0 
        curr_r += curr_v * dt
        true_pos.append(curr_r)
        true_vel.append(curr_v)
        
    if is_maneuver:
        tracker = KalmanTracker(np.array([[20], [5.0]]), np.eye(2)*1, np.eye(2)*0.1, np.array([[0.5]]), dt)
    else:
        tracker = KalmanTracker(np.array([[0], [0]]), np.eye(2)*50, np.eye(2)*0.01, np.array([[10.0]]), dt)

    logs = {'true_pos': true_pos, 'true_vel': true_vel, 'mle_pos': [], 'map_pos': [], 
            'map_vel': [], 'innov': [], 'sigma': [], 'err_pos_mle': [], 'err_pos_map': []}

    for i, r_gt in enumerate(true_pos):
        sig = generate_fmcw_signal(r_gt, snr)
        r_mle = extract_range_mle(sig)
        logs['mle_pos'].append(r_mle)
        logs['err_pos_mle'].append(r_mle - r_gt)
        
        tracker.predict()
        p, v, innov, sigma = tracker.update(r_mle)
        logs['map_pos'].append(p)
        logs['map_vel'].append(v)
        logs['innov'].append(innov)
        logs['sigma'].append(sigma)
        logs['err_pos_map'].append(p - r_gt)

    logs['mle_vel'] = np.diff(logs['mle_pos']) / dt
    true_vel_arr = np.array(true_vel)[1:]
    logs['err_vel_mle'] = logs['mle_vel'] - true_vel_arr
    logs['err_vel_map'] = np.array(logs['map_vel'])[1:] - true_vel_arr
    
    logs['roll_rmse_mle'] = calculate_rolling_rmse(logs['err_pos_mle'])
    logs['roll_rmse_map'] = calculate_rolling_rmse(logs['err_pos_map'])
    
    crlb_val = calculate_crlb_range(snr)
    print_report(scen_name, logs, crlb_val, snr)
    
    return logs, crlb_val

# Senaryoları Çalıştır
logs_s1, crlb_s1 = run_scenario(snr=-5.0, is_maneuver=False, scen_name="SENARYO 1")
logs_s2, crlb_s2 = run_scenario(snr=20.0, is_maneuver=True, scen_name="SENARYO 2")

# --- 6. GÖRSELLEŞTİRME VE KAYDETME ---

def plot_analysis_final(fig, logs, title_prefix, crlb_val, maneuver_line=False):
    axs = fig.subplots(2, 2)
    fig.suptitle(title_prefix, fontsize=16)
    
    # 1. Konum Hatası
    axs[0,0].plot(np.abs(logs['err_pos_mle']), 'r-', alpha=0.3, label='|MLE| (Anlık)')
    axs[0,0].plot(np.abs(logs['err_pos_map']), 'b-', lw=1.5, label='|MAP| (Anlık)')
    axs[0,0].set_title("1. Anlık Konum Hatası")
    axs[0,0].set_ylabel("Hata (m)")
    if maneuver_line: axs[0,0].axvline(40, color='y', ls='--', label='Manevra')
    axs[0,0].legend(loc='upper right'); axs[0,0].grid(True)
    axs[0,0].set_xlabel("Zaman Adımı")

    # 2. Hız Hatası
    axs[0,1].plot(np.abs(logs['err_vel_mle']), 'r-', alpha=0.3, label='|MLE|')
    axs[0,1].plot(np.abs(logs['err_vel_map']), 'b-', lw=2, label='|MAP|')
    axs[0,1].set_title("2. Anlık Hız Hatası")
    axs[0,1].set_ylabel("Hata (m/s)")
    axs[0,1].set_xlabel("Zaman Adımı")
    if maneuver_line: axs[0,1].axvspan(40, 55, color='yellow', alpha=0.3, label='Lag')
    if not maneuver_line: axs[0,1].set_ylim(0, 15)
    axs[0,1].legend(loc='upper right'); axs[0,1].grid(True)

    # 3. RMSE ve CRLB
    axs[1,0].plot(logs['roll_rmse_mle'], 'r--', lw=2, alpha=0.6, label='MLE Trendi')
    axs[1,0].plot(logs['roll_rmse_map'], 'b-', lw=3, label='MAP Trendi')
    axs[1,0].axhline(y=crlb_val, color='k', linestyle='-.', linewidth=2, label=f'CRLB (σ={crlb_val:.3f}m)')
    axs[1,0].set_title("3. RMSE Trendi vs Teorik Limit (CRLB)")
    axs[1,0].set_xlabel("Zaman Adımı")
    axs[1,0].set_ylabel("Ortalama Hata (m)")
    if maneuver_line: axs[1,0].axvline(40, color='y', ls='--')
    axs[1,0].legend(loc='upper right'); axs[1,0].grid(True)

    # 4. İnovasyon
    axs[1,1].plot(logs['innov'], 'g-', label='İnovasyon')
    axs[1,1].fill_between(range(steps), -3*np.array(logs['sigma']), 3*np.array(logs['sigma']), 
                          color='gray', alpha=0.2, label='±3σ')
    axs[1,1].set_title("4. Filtre Tutarlılığı")
    axs[1,1].set_ylabel("Ölçüm Farkı(m)");axs[1,1].set_xlabel("Zaman Adımı")
    if maneuver_line: axs[1,1].axvline(40, color='r', ls='--', label='Manevra')
    axs[1,1].legend(loc='upper right'); axs[1,1].grid(True)

# --- YENİ EKLENEN KAYDETME FONKSİYONU ---
def save_individual_subplots(fig, folder_name, base_filename, subplot_names):
    """
    Bir figürdeki subplotları tek tek ayrı dosyalar olarak kaydeder.
    """
    # Klasör yoksa oluştur
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Klasör oluşturuldu: {folder_name}")

    # Figürü çiz (Koordinatların oturması için gerekli)
    fig.canvas.draw()
    
    # Her bir eksen (subplot) için döngü
    # Not: Figürdeki eksen sırası genelde: [Sol-Üst, Sağ-Üst, Sol-Alt, Sağ-Alt] şeklindedir.
    for i, ax in enumerate(fig.axes):
        if i >= len(subplot_names): break # İsim listesi biterse dur
        
        # Subplot'un sınırlarını al (Bbox)
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        
        # Sınırları biraz genişlet (Eksen etiketleri kesilmesin diye)
        # expanded(width_factor, height_factor)
        expanded_extent = extent.expanded(1.15, 1.25)
        
        # Dosya yolu oluştur
        fname = f"{base_filename}_{subplot_names[i]}.png"
        full_path = os.path.join(folder_name, fname)
        
        # Sadece o bölgeyi kaydet
        fig.savefig(full_path, bbox_inches=expanded_extent, dpi=300)
        print(f"Kaydedildi: {full_path}")

# --- ÇİZİM VE KAYDETME ÇAĞRILARI ---

# 1. Genel Karşılaştırma Grafiği
fig1, axs1 = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle("PENCERE 1: Rota ve Hız Sonuçları", fontsize=16)

axs1[0,0].plot(logs_s1['true_pos'], 'k--', label='Gerçek'); axs1[0,0].plot(logs_s1['mle_pos'], 'r.', label='MLE', alpha=0.3); axs1[0,0].plot(logs_s1['map_pos'], 'b-', label='MAP')
axs1[0,0].set_title("S1: Konum"); axs1[0,0].legend(loc='upper right'); axs1[0,0].grid(True);axs1[0,0].set_xlabel("Zaman Adımı");axs1[0,0].set_ylabel("Konum(m)")

axs1[0,1].plot(logs_s1['true_vel'], 'k--', label='Gerçek'); axs1[0,1].plot(logs_s1['mle_vel'], 'r-', label='MLE', alpha=0.3); axs1[0,1].plot(logs_s1['map_vel'], 'b-', label='MAP')
axs1[0,1].set_title("S1: Hız"); axs1[0,1].set_ylim(-5, 10); axs1[0,1].grid(True);axs1[0,1].legend(loc='upper right')
axs1[0,1].set_xlabel("Zaman Adımı");axs1[0,1].set_ylabel("Hız(m/s)")

axs1[1,0].plot(logs_s2['true_pos'], 'k--', label='Gerçek'); axs1[1,0].plot(logs_s2['mle_pos'], 'r.', label='MLE', alpha=0.3); axs1[1,0].plot(logs_s2['map_pos'], 'b-', label='MAP')
axs1[1,0].set_title("S2: Konum"); axs1[1,0].legend(loc='upper right'); axs1[1,0].grid(True);axs1[1,0].set_xlabel("Zaman Adımı");axs1[1,0].set_ylabel("Konum(m)")

axs1[1,1].plot(logs_s2['true_vel'], 'k--', label='Gerçek'); axs1[1,1].plot(logs_s2['mle_vel'], 'r-', label='MLE', alpha=0.3); axs1[1,1].plot(logs_s2['map_vel'], 'b-', label='MAP')
axs1[1,1].set_title("S2: Hız"); axs1[1,1].grid(True);axs1[1,1].legend(loc='upper right')
axs1[1,1].set_xlabel("Zaman Adımı");axs1[1,1].set_ylabel("Hız(m/s)")

# 2. Senaryo 1 Analizi
fig2 = plt.figure(figsize=(14, 9))
plot_analysis_final(fig2, logs_s1, "PENCERE 2: Senaryo 1 Analizi (-5dB SNR)", crlb_s1, maneuver_line=False)

# 3. Senaryo 2 Analizi
fig3 = plt.figure(figsize=(14, 9))
plot_analysis_final(fig3, logs_s2, "PENCERE 3: Senaryo 2 Analizi (20dB SNR + Manevra)", crlb_s2, maneuver_line=True)

# --- KAYDETME İŞLEMİ ---
output_folder = "Radar_Analiz_Sonuclari"

# Figür 1'in alt grafiklerini kaydet
save_individual_subplots(fig1, output_folder, "Fig1", 
                         ["S1_Rota_Takibi", "S1_Hiz_Takibi", "S2_Rota_Takibi", "S2_Hiz_Takibi"])

# Figür 2'nin alt grafiklerini kaydet
save_individual_subplots(fig2, output_folder, "S1_Analiz", 
                         ["Konum_Hatasi", "Hiz_Hatasi", "RMSE_CRLB", "Inovasyon"])

# Figür 3'ün alt grafiklerini kaydet
save_individual_subplots(fig3, output_folder, "S2_Analiz", 
                         ["Konum_Hatasi", "Hiz_Hatasi", "RMSE_CRLB", "Inovasyon"])

print(f"\nTüm grafikler '{output_folder}' klasörüne ayrı ayrı kaydedildi.")
plt.show()