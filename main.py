import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. AYARLAR VE BAŞLANGIÇ DEĞERLERİ ---
Fs = 1000.0       # Örnekleme Frekansı (Hz)
N = 64            # FFT Boyutu (Düşük tutuldu ki "basamaklanma" belli olsun)
SNR_dB = 10       # Sinyal Gürültü Oranı
num_steps = 100   # Simülasyon adım sayısı

# Gerçek Hedef (Ground Truth): 
# Frekans 10.2'den başlayıp 15.8'e doğru lineer artıyor (Sabit hızla uzaklaşan hedef)
true_freqs = np.linspace(10.2, 15.8, num_steps) * (Fs/N) 
# Not: Burada "frekans" doğrudan hedef mesafesini temsil eder (FMCW prensibi).

# Sonuçları saklayacağımız listeler
fft_estimates = []
kalman_estimates = []

# --- KALMAN FİLTRESİ DEĞİŞKENLERİ ---
# Durum Vektörü [Frekans, Frekans_Değişim_Hızı]
x_est = np.array([true_freqs[0], 0]) 
P = np.eye(2) * 1.0              # Kovaryans matrisi
dt = 1.0                         # Zaman adımı (birim zaman)
F_mat = np.array([[1, dt],       # Geçiş Matrisi (Sabit Hız Modeli)
                  [0, 1]])
H = np.array([[1, 0]])           # Ölçüm Matrisi (Sadece konumu/frekansı ölçüyoruz)
Q = np.array([[0.01, 0],         # Süreç Gürültüsü (Sistem ne kadar oynak?)
              [0, 0.01]]) * 1.0 
R_cov = 0.01                     # Ölçüm Gürültüsü (MLE'ye ne kadar güveniyoruz?)

# --- 2. SİMÜLASYON DÖNGÜSÜ ---
print("Simülasyon çalışıyor...")

for t_step, true_f in enumerate(true_freqs):
    
    # A. Sinyal Oluşturma (Gürültülü)
    t = np.arange(N) / Fs
    # Karmaşık sinyal: e^(j*2pi*f*t)
    clean_sig = np.exp(1j * 2 * np.pi * true_f * t)
    
    # Gürültü ekle
    noise_power = 10**(-SNR_dB / 10)
    noise = (np.random.randn(N) + 1j * np.random.randn(N)) * np.sqrt(noise_power/2)
    signal = clean_sig + noise

    # --- YÖNTEM 1: Sadece FFT (Coarse) ---
    fft_vals = np.abs(np.fft.fft(signal))
    peak_idx = np.argmax(fft_vals)
    
    # FFT Frekans Çözünürlüğü (Bin Width)
    bin_width = Fs / N
    f_coarse = peak_idx * bin_width
    fft_estimates.append(f_coarse)

    # --- YÖNTEM 2: MLE + Kalman (Fine) ---
    
    # Adım 2.1: MLE (Fine Search)
    # Cost Function: Negatif Korelasyon
    def mle_cost(f_val):
        # Tahmini model
        model = np.exp(1j * 2 * np.pi * f_val * t)
        # Vektör çarpımı (Correlation)
        corr = np.abs(np.vdot(signal, model)) # Genliği normalize etmiyoruz, max'ı arıyoruz
        return -corr 

    # KRİTİK KISIM: Arama sınırları (Bounds)
    # FFT sonucunun sadece +/- 0.6 bin sağına soluna bak.
    # Bu, algoritmanın sonsuza ıraksamasını engeller.
    search_bounds = [(f_coarse - 0.6*bin_width, f_coarse + 0.6*bin_width)]
    
    res = minimize(mle_cost, f_coarse, method='L-BFGS-B', bounds=search_bounds, tol=1e-4)
    f_fine_mle = res.x[0]

    # Adım 2.2: Kalman Update
    # 1. Predict
    x_pred = F_mat @ x_est
    P_pred = F_mat @ P @ F_mat.T + Q
    
    # 2. Update (Measurement = f_fine_mle)
    y = f_fine_mle - (H @ x_pred)      # Innovation
    S = H @ P_pred @ H.T + R_cov       # Innovation Covariance
    K = P_pred @ H.T @ np.linalg.inv(S)# Kalman Gain
    x_est = x_pred + K @ y             # Updated State
    P = (np.eye(2) - K @ H) @ P_pred   # Updated Covariance
    
    kalman_estimates.append(x_est[0])

# --- 3. HATA HESAPLAMA (RMSE) ---
true_freqs = np.array(true_freqs)
fft_estimates = np.array(fft_estimates)
kalman_estimates = np.array(kalman_estimates)

err_fft = true_freqs - fft_estimates
err_kalman = true_freqs - kalman_estimates

rmse_fft = np.sqrt(np.mean(err_fft**2))
rmse_kalman = np.sqrt(np.mean(err_kalman**2))

print("-" * 40)
print(f"SONUÇLAR (N={N} nokta FFT için):")
print(f"FFT RMSE Hatası        : {rmse_fft:.4f} Hz")
print(f"MLE + Kalman RMSE Hatası: {rmse_kalman:.4f} Hz")
print(f"İyileştirme Oranı      : {rmse_fft / rmse_kalman:.1f} kat daha iyi")
print("-" * 40)

# --- 4. GRAFİK ÇİZİMİ ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Üst Grafik: Yörünge Takibi
ax1.plot(true_freqs, 'k--', label='Gerçek Konum (Ground Truth)', linewidth=2)
ax1.plot(fft_estimates, 'o-', label='Sadece FFT (Quantized)', alpha=0.5, markersize=4)
ax1.plot(kalman_estimates, 'r-', label='MLE + Kalman (Filtered)', linewidth=2)
ax1.set_ylabel('Frekans (Konum)')
ax1.set_title(f'FMCW Takip Performansı (N={N})')
ax1.legend()
ax1.grid(True)

# Alt Grafik: Hata Miktarı
ax2.plot(err_fft, 'o-', label='FFT Hatası', alpha=0.5)
ax2.plot(err_kalman, 'r-', label='MLE+Kalman Hatası', linewidth=2)
ax2.axhline(0, color='black', linewidth=1)
ax2.set_ylabel('Hata (Hz)')
ax2.set_xlabel('Zaman Adımı')
ax2.set_title('Gerçek Konuma Göre Hata Miktarı')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()