import numpy as np

import SD_BCF_OOP as SCP

# Constants:
# ----------
T = 295  # Unit : K
kBT = 0.69503 * T  # Unit: cm^-1
h_bar = 5308  # Unit: cm^-1 * fs
gamma = 100  # Unit: cm^-1
lamd = 50  # Unit : cm^-1
delta_t = 5  # Unit:fs
t_list = np.arange(delta_t, 1000, delta_t)  # Unit : fs
delta_omega = 1  # Unit: cm^-1
omega_list = np.arange(delta_omega, 1000, delta_omega)  # Unit: cm^-1
omega_c = 500  # Unit: cm^-1
l = 0
N = 40

# Object Instantiation:
# ---------------------
SD = SCP.SpectralDensity()  # SD Object Instantiation
# C_t = SCP.CorrelationFunction(omega_list, lamd, gamma, omega_c, kBT)  # correlation Object Instantiation
# MatsubaraApproximation = SCP.MatsubaraApproximation(omega_list, lamd, gamma, omega_c)  # Matsubara Approx Object
# Instantiation

# storing list:
# -------------
BCF_under_damped_highTemp = [ ]
BCF_over_damped_highTemp = [ ]
BCF_under_damped_lowTemp = [ ]
BCF_over_damped_lowTemp = [ ]

# for t in t_list:
#     # High temp approximation:
#     # ------------------------
#     BCF_under_damped_highTemp.append(C_t.correlation_underDamped_mode(t))
#     BCF_over_damped_highTemp.append(C_t.correlation_overDamped_mode(t))
#     # Low temperature approximation:
#     # ------------------------------
#     BCF_over_damped_lowTemp.append(
#         C_t.correlation_overDamped_mode(t) - MatsubaraApproximation.Mats_overDamped(t, l, N))
#     BCF_under_damped_lowTemp.append(
#         C_t.correlation_underDamped_mode(t) - MatsubaraApproximation.Mats_underDamped(t, l, N))

# Total_BCF_highTemp = np.array(BCF_under_damped_highTemp) + np.array(BCF_over_damped_highTemp)
# Total_BCF_lowTemp = np.array(BCF_under_damped_lowTemp) + np.array(BCF_over_damped_lowTemp)

Total_SD = SCP.SdUnderDamped(omega_list, omega_c) + SCP.SdOverDamped(omega_c)  # construct 1 over & 1 under-damped mode

# Plotting:
# ---------
# plt.plot(t_list, Total_BCF_highTemp)
# plt.plot(t_list,Total_BCF_lowTemp)
# plt.plot(Total_SD)
# plt.plot(t_list, BCF_over_damped_lowTemp)
# plt.plot(t_list, BCF_under_damped_lowTemp)
# plt.show()

# fig, axs = plt.subplots(2)
# axs[ 0 ].plot(t_list, Total_BCF_highTemp, 'g')
# # axs[ 1 ].plot(t_list,Total_BCF_lowTemp, 'r')
# axs[ 1 ].plot(Total_SD, 'g--')
# plt.show()
