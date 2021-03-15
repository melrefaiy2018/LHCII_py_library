# This piece of code represents the comparison between Novoderezhkin and Kreisbeck Spectral Density followed by calculation of
# the correlation function for both cases:
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz

# Constants:
# ----------
T = 295  # Unit : K
kBT = 0.69503 * T  # Unit: cm^-1
h_bar = 5308  # Unit: cm^-1 * fs
gamma = 50  # Unit: cm^-1
delta_t = 5  # Unit:fs
t_list = np.arange(delta_t, 1000, delta_t)  # Unit : fs
delta_omega = 1  # Unit: cm^-1
omega_list = np.arange(delta_omega, 1000, delta_omega)  # Unit: cm^-1
omega_c = np.array([ 300, 518, 745 ])  # Unit: cm^-1
lamd = np.array([ 161, 16, 48 ])  # Unit : cm^-1
neu = np.array([ 40, 1000, 600 ])


# Novadri. Formula:
# -----------------
def J_underdamped_mode_fx(omega, omega_c, lamd, gamma):
    return (2 * lamd * gamma * omega * omega_c ** 2) / ((omega_c ** 2 - omega ** 2) ** 2 + (omega ** 2) * (gamma ** 2))


def J_overdamped_mode_fx(omega, lamd, gamma):
    return 2 * lamd * omega * gamma / (omega ** 2 + gamma ** 2)


def BCF_underdamped_real_Nov(omega, omega_c, t, lamd, gamma):
    SD = J_underdamped_mode_fx(omega, omega_c, lamd, gamma)
    coth = 1 / np.tanh(omega / (2 * kBT))
    return (1 / np.pi) * SD * (coth * np.cos(omega * t / h_bar))


def BCF_overdamped_real_Nov(omega, lamd, gamma, t):
    coth = 1 / np.tanh(omega / (2 * kBT))
    SD = J_overdamped_mode_fx(omega, lamd, gamma)
    return (1 / np.pi) * SD * (coth * np.cos(omega * t / h_bar))


# def BCF_overdamped_imaginary_Nov(omega, t, h_bar):
#     SD_over = J_overdamped_mode_fx(omega, lamd, gamma)
#     return (- 1 / np.pi) * np.sin(omega * t / h_bar) * SD_over
#
# def BCF_underdamped_imaginary_Nov(omega, t):
#     SD_under = J_underdamped_mode_fx(omega, omega_c, lamd, gamma)
#     return (- 1 / np.pi) * np.sin(omega * t / h_bar) * SD_under


# Kreisbeck Formula:
# ------------------

# Kreisbeck Spectral Density:

def JK_w(w):
    w_k = np.array([ 300, 518, 745 ])  # Unit: cm^-1
    lamd = np.array([ 161, 16, 48 ])  # Unit : cm^-1
    neu = np.array([ 40, 1000, 600 ])
    J1 = (h_bar * lamd * w / neu) / ((h_bar ** 2 / neu ** 2) + (w + w_k) ** 2)
    J2 = (h_bar * lamd * w / neu) / ((h_bar ** 2 / neu ** 2) + (w - w_k) ** 2)
    return J1 + J2


def BCF_underdamped_real_Kri(omega, omega_c, lamd, neu, t):
    def JK_w():
        J1 = (h_bar * lamd * omega / neu) / ((h_bar ** 2 / neu ** 2) + (omega + omega_c) ** 2)
        J2 = (h_bar * lamd * omega / neu) / ((h_bar ** 2 / neu ** 2) + (omega - omega_c) ** 2)
        return J1 + J2

    SD = JK_w()
    coth = 1 / np.tanh(omega / (2 * kBT))
    return (1 / np.pi) * SD * (coth * np.cos(omega * t / h_bar))


def BCF_overdamped_real_Kri(omega, omega_c, lamd, neu, t):
    coth = 1 / np.tanh(omega / (2 * kBT))

    def JK_w():
        J1 = (h_bar * lamd * omega / neu) / ((h_bar ** 2 / neu ** 2) + (omega + omega_c) ** 2)
        J2 = (h_bar * lamd * omega / neu) / ((h_bar ** 2 / neu ** 2) + (omega - omega_c) ** 2)
        return J1 + J2

    SD = JK_w()
    return (1 / np.pi) * SD * (coth * np.cos(omega * t / h_bar))


# List:
# =====
J_under = [ ]
J_over = [ ]
JK_lst = [ ]
Over_Correlation_R_Kri_list = [ ]
Over_Correlation_R_Nov_list = [ ]
Under_Correlation_R_Nov_list = [ ]
Under_Correlation_R_Kri_list = [ ]
Over_Correlation_Im_Nov_list = [ ]

# Calculating the Spectral Density:
# ---------------------------------
for w in omega_list:
    JK_lst.append(JK_w(w))  # spectral Density using Kreisbeck formula
    J_under.append(J_underdamped_mode_fx(w, omega_c, lamd, gamma))  # spectral Density using numerical methods
    J_over.append(J_overdamped_mode_fx(w, lamd, gamma))  # spectral Density using numerical methods
J_w_array = np.array(J_over) + np.array(J_under)

# Evaluate the Correlation function:
# ==================================
for t in t_list:
    # A : Novadr. case:
    # -----------------
    # over-damped mode / Real part:
    # =============================
    Over_integrand_R_Nov_list1 = [ BCF_overdamped_real_Nov(omega, lamd[ 0 ], gamma, t) for omega in omega_list ]
    Over_integrand_R_Nov_list2 = [ BCF_overdamped_real_Nov(omega, lamd[ 1 ], gamma, t) for omega in omega_list ]
    Over_integrand_R_Nov_list3 = [ BCF_overdamped_real_Nov(omega, lamd[ 2 ], gamma, t) for omega in omega_list ]
    Over_integrand_R_Nov_list = np.array(Over_integrand_R_Nov_list1) + np.array(Over_integrand_R_Nov_list2) + np.array(
        Over_integrand_R_Nov_list3)
    Over_Correlation_R_Nov_list.append(trapz(Over_integrand_R_Nov_list, omega_list, delta_omega))
    # # over-damped mode / Imaginary part:
    # # ==================================
    # Over_integrand_Im_Nov_list1 = [ BCF_overdamped_imaginary_Nov(omega, t, h_bar) for omega in omega_list ]
    # Over_Correlation_Im_Nov_list.append(trapz(Over_integrand_Im_Nov_list1, omega_list, delta_omega))

    # Under-damped mode / Real part:
    # ==============================
    Under_integrand_R_Nov_list1 = [ BCF_underdamped_real_Nov(omega, omega_c[ 0 ], t, lamd[ 0 ], gamma) for omega in
                                    omega_list ]
    Under_integrand_R_Nov_list2 = [ BCF_underdamped_real_Nov(omega, omega_c[ 1 ], t, lamd[ 1 ], gamma) for omega in
                                    omega_list ]
    Under_integrand_R_Nov_list3 = [ BCF_underdamped_real_Nov(omega, omega_c[ 2 ], t, lamd[ 2 ], gamma) for omega in
                                    omega_list ]
    Under_integrand_R_Nov_list = np.array(Under_integrand_R_Nov_list1) + np.array(
        Under_integrand_R_Nov_list2) + np.array(
        Under_integrand_R_Nov_list3)
    Under_Correlation_R_Nov_list.append(trapz(Under_integrand_R_Nov_list, omega_list, delta_omega))

    # B : Krisbeck case:
    # ------------------
    # over-damped mode / Real part:
    # =============================
    Over_integrand_R_Kri_list1 = [ BCF_overdamped_real_Kri(omega, omega_c[ 0 ], lamd[ 0 ], neu[ 0 ], t) for omega in
                                   omega_list ]
    Over_integrand_R_Kri_list2 = [ BCF_overdamped_real_Kri(omega, omega_c[ 1 ], lamd[ 1 ], neu[ 1 ], t) for omega in
                                   omega_list ]
    Over_integrand_R_Kri_list3 = [ BCF_overdamped_real_Kri(omega, omega_c[ 2 ], lamd[ 2 ], neu[ 2 ], t) for omega in
                                   omega_list ]
    Over_integrand_R_Kri_list = np.array(Over_integrand_R_Kri_list1) + np.array(Over_integrand_R_Kri_list2) + np.array(
        Over_integrand_R_Kri_list3)
    Over_Correlation_R_Kri_list.append(trapz(Over_integrand_R_Kri_list, omega_list, delta_omega))

    # Under-damped mode / Real part:
    # ==============================
    Under_integrand_R_Kri_list1 = [ BCF_underdamped_real_Kri(omega, omega_c[ 0 ], lamd[ 0 ], neu[ 0 ], t) for omega in
                                    omega_list ]
    Under_integrand_R_Kri_list2 = [ BCF_underdamped_real_Kri(omega, omega_c[ 1 ], lamd[ 1 ], neu[ 1 ], t) for omega in
                                    omega_list ]
    Under_integrand_R_Kri_list3 = [ BCF_underdamped_real_Kri(omega, omega_c[ 2 ], lamd[ 2 ], neu[ 2 ], t) for omega in
                                    omega_list ]
    Under_integrand_R_Kri_list = np.array(Under_integrand_R_Kri_list1) + np.array(
        Under_integrand_R_Kri_list2) + np.array(Under_integrand_R_Kri_list3)
    Under_Correlation_R_Kri_list.append(trapz(Under_integrand_R_Kri_list, omega_list, delta_omega))

# plotting Correlation:
# =====================
fig1 = plt.figure("Correlation for OverDamped mode")
fig1.suptitle("Correlation for OverDamped mode")
plt.plot(t_list, Over_Correlation_R_Kri_list, label="Kreisbeck")
plt.plot(t_list, Over_Correlation_R_Nov_list, '--', label="Novoderezhkin")
plt.xlabel(r'$\tau (fs)$', fontsize=14)
plt.ylabel(r'$\alpha(\tau) (cm-2)$', fontsize=14)
plt.legend()
fig1.savefig("/Users/48107674/Box/Reseach/2020/research/simulating_exciton_transport_in_LHCII_aggregates/graphs/Nov"
             "-Kriesbeck_comparison/Correlation_for_OverDamped_mode.png")

fig2 = plt.figure("Correlation for UnderDamped mode")
fig2.suptitle("Correlation for UnderDamped mode")
plt.plot(t_list, Under_Correlation_R_Kri_list, label="Kreisbeck")
plt.plot(t_list, Under_Correlation_R_Nov_list, '--', label="Novoderezhkin")
plt.xlabel(r'$\tau (fs)$', fontsize=14)
plt.ylabel(r'$\alpha(\tau) (cm-2)$', fontsize=14)
plt.legend()
fig2.savefig("/Users/48107674/Box/Reseach/2020/research/simulating_exciton_transport_in_LHCII_aggregates/graphs/Nov"
             "-Kriesbeck_comparison/Correlation_for_UnderDamped_mode.png")

fig3 = plt.figure("Total Correlation")
fig3.suptitle("Correlation for UnderDamped & OverDamped")
plt.plot(t_list, np.array(Over_Correlation_R_Kri_list) + np.array(Under_Correlation_R_Kri_list), label="Kreisbeck")
plt.plot(t_list, np.array(Over_Correlation_R_Nov_list) + np.array(Under_Correlation_R_Nov_list), '--',
         label="Novoderezhkin")
plt.legend()
fig3.savefig("/Users/48107674/Box/Reseach/2020/research/simulating_exciton_transport_in_LHCII_aggregates/graphs/Nov"
             "-Kriesbeck_comparison/Total Correlation.png")

# plotting Spectral Density:
# ==========================
fig4 = plt.figure("Spectral Density")
fig4.suptitle("Spectral Density")
plt.plot(omega_list, JK_lst, label="Kreisbeck.")
plt.plot(omega_list, J_w_array, '--', label="Novoderezhkin")
plt.xlabel(r'$w (cm^-1)$', fontsize=14)
plt.ylabel(r'$J(w) (cm-1)$', fontsize=14)
plt.legend()
plt.show()
fig4.savefig("/Users/48107674/Box/Reseach/2020/research/simulating_exciton_transport_in_LHCII_aggregates/graphs/Nov"
             "-Kriesbeck_comparison/Spectral_Density.png")
