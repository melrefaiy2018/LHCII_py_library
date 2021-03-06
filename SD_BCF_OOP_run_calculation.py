import numpy as np

from SD_BCF_OOP import SpectralDensity, Spectral_OverDamped, Spectral_UnderDamped, CorrelationFunction, \
    Correlation_overDamped, Correlation_underDamped

# Constants:
# ----------
T = 295  # Unit : K
kBT = 0.69503 * T  # Unit: cm^-1
h_bar = 5308  # Unit: cm^-1 * fs
gamma = 100  # Unit: cm^-1
delta_t = 5  # Unit:fs
t_list = np.arange(delta_t, 1000, delta_t)  # Unit : fs
delta_omega = 1  # Unit: cm^-1
omega_list = np.arange(delta_omega, 1000, delta_omega)  # Unit: cm^-1
# omega_c = 500  # Unit: cm^-1
# lamd = 50  # Unit : cm^-1
omega_c = np.array([ [ 300 ], [ 518 ], [ 745 ] ])  # Unit: cm^-1
lamd = np.array([ [ 161 ], [ 16 ], [ 48 ] ])  # Unit : cm^-1

l = 0
N = 40

# # Object Instantiation:
# # ---------------------
SD = SpectralDensity(lamd, gamma, kBT, h_bar)
SD_over1 = Spectral_OverDamped(omega_list, omega_c[ 0 ], lamd, gamma, kBT, h_bar)  # the first over-damped
SD_under1 = Spectral_UnderDamped(omega_list, omega_c[ 0 ], lamd, gamma, kBT, h_bar)  # the first under-damped
SD_under2 = Spectral_UnderDamped(omega_list, omega_c[ 1 ], lamd, gamma, kBT, h_bar)  # the second under-damped
SD_under3 = Spectral_UnderDamped(omega_list, omega_c[ 2 ], lamd, gamma, kBT, h_bar)  # the second under-damped

# Mats_over = Mats_overDamped(omega_list, omega_c, t_list, l, N)
# Mats_under = Mats_underDamped(omega_list, omega_c, t_list, l, N)

Total_SD = SD_over1.calculate() + SD_under1.calculate() + SD_under2.calculate() + SD_under3.calculate()  # construct 1 over & 1 under-damped mode
Total_SD = Total_SD.reshape(999, 3)

# Plot SD:
# ========
# plt.figure()
# plt.plot(omega_list, Total_SD)
# plt.show()

Correlation = CorrelationFunction(lamd, gamma, kBT, h_bar)

corr_over1 = Correlation_overDamped(omega_list, omega_c, t_list, lamd, gamma, kBT, h_bar)
corr_under1 = Correlation_underDamped(omega_list, omega_c, t_list, lamd, gamma, kBT, h_bar)
corr_under2 = Correlation_underDamped(omega_list, omega_c, t_list, lamd, gamma, kBT, h_bar)
corr_under3 = Correlation_underDamped(omega_list, omega_c, t_list, lamd, gamma, kBT, h_bar)

Total_correlation = corr_over1.calculate() + corr_under1.calculate() + corr_under2.calculate() + corr_under3.calculate()
Total_correlation = Total_correlation.reshape(199, 3)

# Plot Correlation:
# ----------------
# plt.figure()
# plt.plot(t_list, Total_correlation)
# plt.show()
# Combine = Combine_Spectral(omega_list, omega_c, lamd, gamma, kBT, h_bar)
