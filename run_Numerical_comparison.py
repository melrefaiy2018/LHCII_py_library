from Numerical_BCF import *
from SD_BCF_OOP import CorrelationFunction, \
    Correlation_overDamped, Correlation_underDamped

# Constants:
# ----------
T = 295  # Unit : K
kBT = 0.69503 * T  # Unit: cm^-1
h_bar = 5308  # Unit: cm^-1 * fs
gamma = 100  # Unit: cm^-1
delta_t = 5  # Unit:fs
# t_list = np.arange(delta_t, 1000, delta_t)  # Unit : fs
t_list = np.linspace(0.1, 1000, 999)  # Unit : fs

delta_omega = 1  # Unit: cm^-1
omega_list = np.arange(delta_omega, 1000, delta_omega)  # Unit: cm^-1
omega_c = 500  # Unit: cm^-1
lamd = 50  # Unit : cm^-1
# omega_c = np.array([ [ 300 ], [ 518 ], [ 745 ] ])  # Unit: cm^-1
# lamd = np.array([ [ 161 ], [ 16 ], [ 48 ] ])  # Unit : cm^-1

l = 0
N = 40

# Built the SD:
# ============
SD = SpectralDensity(lamd, gamma, kBT, h_bar)
SD_over = Spectral_OverDamped(omega_list, omega_c, lamd, gamma, kBT, h_bar)
SD.calculate()
# plotting SD:
# ============
# SD_over.plotting(omega_list)

# Build Correlation:
# ==================
Correlation_A = CorrelationFunction(lamd, gamma, kBT, h_bar)
Correlation_N = CorrelationFunction(lamd, gamma, kBT, h_bar)

corr_over_analytic = Correlation_overDamped(omega_list, omega_c, t_list, lamd, gamma, kBT, h_bar)

correlation_N = NumericalCorrelation(lamd, gamma, h_bar, kBT)
# corr_over_numerical = NumericalCorrOverDamped(omega_list, omega_c, t_list, lamd, gamma, h_bar, kBT)
corr_over_numerical = NumericalCorrOverDamped(omega_list, omega_c, delta_t, lamd, gamma, h_bar, kBT)

# Plotting BCF:
# =============
# corr_over_numerical.plotting(t_list)
# corr_over_analytic.plotting(t_list)


# UnderMode:
# ==========
corr_under_analytic = Correlation_underDamped(omega_list, omega_c, t_list, lamd, gamma, kBT, h_bar)
# corr_under_numerical = NumericalCorrUnderDamped(omega_list, omega_c, t_list, lamd, gamma, h_bar, kBT)
corr_under_numerical = NumericalCorrUnderDamped(omega_list, omega_c, delta_t, lamd, gamma, h_bar, kBT)

# Plotting BCF:
# =============
# corr_under_numerical.plotting(t_list)
# corr_under_analytic.plotting(t_list)


corr_under_numerical_lst = [ ]
for t in t_list:
    corr_under_numerical = NumericalCorrUnderDamped(omega_list, omega_c, t, lamd, gamma, h_bar, kBT)
    corr_under_numerical_lst.append(corr_under_numerical)

# # test
# BCF_over_lst = 0
# BCF_under_lst = 0
# for t in t_list:
#     BCF_over_lst += (corr_over_numerical.calculate())
#     BCF_under_lst += (corr_under_numerical.calculate())
