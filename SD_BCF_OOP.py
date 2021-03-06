import matplotlib.pyplot as plt
import numpy as np


class SpectralDensity:
    """
       This class is used to build the spectral density for the over and under oscillation mode.
       """

    def __init__(self, lamd, gamma, kBT, h_bar):
        self.lamd = lamd
        self.gamma = gamma
        self.kBT = kBT
        self.h_bar = h_bar
        # self.omega = omega
        # self.omega_c = omega_c

    def __repr__(self):
        return "Spectral Density with paramters equal (lamd = {} ,gamma = {})".format(self.lamd, self.gamma)

    def calculate(self):
        pass

    def plotting(self, x_axis):
        return plt.plot(x_axis, self.calculate())  # ex: SpectralDensity.plotting(over,omega_list)


class Spectral_OverDamped(SpectralDensity):
    def __init__(self, omega, omega_c, lamd, gamma, kBT, h_bar):
        super().__init__(lamd, gamma, kBT, h_bar)
        self.omega = omega
        self.omega_c = omega_c

    # Class_methods:
    # ==============
    def calculate(self):
        return 2 * self.lamd * self.omega * self.gamma / (self.omega ** 2 + self.gamma ** 2)


class Spectral_UnderDamped(SpectralDensity):
    def __init__(self, omega, omega_c, lamd, gamma, kBT, h_bar):
        super().__init__(lamd, gamma, kBT, h_bar)
        self.omega = omega
        self.omega_c = omega_c

    def calculate(self):
        return (2 * self.lamd * self.gamma * self.omega * self.omega_c ** 2) / (
                (self.omega_c ** 2 - self.omega ** 2) ** 2 + (self.omega ** 2) * (self.gamma ** 2))


class CorrelationFunction(SpectralDensity):
    """
    This class is used to calculate the Bath Correlation Function for the over and under damped oscillation modes.
    """

    def __init__(self, lamd, gamma, kBT, h_bar):
        super().__init__(lamd, gamma, kBT, h_bar)

    def calculate(self):
        pass

    def plotting(self, x_axis):
        plt.plot(x_axis, self.calculate())  # ex: SpectralDensity.plotting(over,omega_list)


class Correlation_overDamped(Spectral_UnderDamped):
    def __init__(self, omega, omega_c, t, lamd, gamma, kBT, h_bar):
        super().__init__(omega, omega_c, lamd, gamma, kBT, h_bar)
        self.t = t

    def calculate(self):
        """
        This function is used to calculate the overdamped mode at high temperature approximation:
        Equation:
        bcf(t) = lambda_k * gamma_k * (coth(beta*gamma_k/2)-1j) * exp(-gamma_k*t)

        """
        beta = 1 / self.kBT
        cot = 1 / np.tan((self.gamma * beta) / 2)
        return self.lamd * self.gamma * (cot - 1j) * np.exp(- self.gamma * self.t / self.h_bar)


class Correlation_underDamped(Spectral_UnderDamped):
    def __init__(self, omega, omega_c, t, lamd, gamma, kBT, h_bar):
        super().__init__(omega, omega_c, lamd, gamma, kBT, h_bar)
        self.t = t

    def calculate(self):
        """
        This function is used to calculate the under-damped mode at high temperature approximation:
        FORMULA:
        bcf(t) = lambda_k * omega_k**2/(2*chi_k) * [coth(beta*(chi_k+1j*gamma_k/2)/2)-1] * exp(-(gamma_k/2-1j*chi_k)*t)
                +lambda_k * omega_k**2/(2*chi_k) * [-coth(beta*(-chi_k+1j*gamma_k/2)/2)+1] * exp(-(gamma_k/2+1j*chi_k)*t)

        """
        beta = 1 / self.kBT
        ki = np.sqrt(self.omega_c ** 2 - (self.gamma ** 2 / 4))

        coth_plus_analytical = 1 / np.tanh(beta * (ki + 1j * self.gamma / 2) / 2)
        coth_negative_analytical = 1 / np.tanh(beta * (- ki + 1j * self.gamma / 2) / 2)

        exp_plus_analytical = np.exp(-self.t * (self.gamma / 2 + 1j * ki) / self.h_bar)
        exp_negative_analytical = np.exp(-self.t * (self.gamma / 2 - 1j * ki) / self.h_bar)
        constant = self.lamd * self.omega_c ** 2 / (2 * ki)

        return constant * (coth_plus_analytical - 1) * exp_negative_analytical + (
                - coth_negative_analytical + 1) * exp_plus_analytical


class MatsubaraApproximation(CorrelationFunction):
    """
    This class is used to calculate the matsubura approximation for both the over and under oscillation modes.
    # Low Temp approximation: gamma * T > 1
    """

    def __init__(self, lamd, gamma, kBT, h_bar):
        super().__init__(lamd, gamma, kBT, h_bar)

    def calculate(self):
        pass


class Mats_overDamped(Correlation_overDamped):
    def __init__(self, omega, omega_c, t, l, N, lamd, gamma, kBT, h_bar):
        super().__init__(omega, omega_c, t, lamd, gamma, kBT, h_bar)
        self.l = l
        self.N = N

    def calculate(self):
        """
        This is the Matsubara approximation for the over-damped mode at the low temperature approximation.
        Equation:
        - 4 * lambda_k * gamma_k/beta * sum_l(eta_l * nu_l/( gamma_k**2-nu_l**2)*exp(-nu_l*t) )
        with beta = 1/temp
        """
        beta = 1 / self.kBT
        eta = 1
        # neu = 2 * np.pi * l / beta
        sigma_sum = np.array(
            [ (eta * 2 * np.pi * l_min / beta) * np.exp(- 2 * np.pi * l_min / beta * self.t / self.h_bar) / (
                    self.gamma ** 2 - (2 * np.pi * l_min / beta) ** 2) for l_min in range(self.l, self.N) ]).sum()
        return sigma_sum * 4 * self.lamd * self.gamma / beta


class Mats_underDamped(Correlation_underDamped):
    def __init__(self, omega, omega_c, t, l, N, lamd, gamma, kBT, h_bar):
        super().__init__(omega, omega_c, t, lamd, gamma, kBT, h_bar)
        self.l = l
        self.N = N

    def calculate(self):
        """
        This is the Matsubara approximation for the under-damped mode at the low temperature approximation.
        Equation:
        - 4 * lambda_k * gamma_k/beta *omega_k**2 * sum_l(eta_l * nu_l/((
        omega_k**2+nu_l**2)**2-gamma_k**2*nu_l**2)*exp(-nu_l*t) ) with beta = 1/temp and chi_k = np.sqrt(
        omega_k**2-gamma_k**2/4)
        """
        beta = 1 / self.kBT
        eta = 1
        # neu = 2 * np.pi * l / beta
        constant = 4 * self.lamd * self.gamma * self.omega_c ** 2 / beta
        sigma_sum = np.array(
            [ (eta * 2 * np.pi * l_min / beta) * np.exp((- 2 * np.pi * l_min / beta) * self.t / self.h_bar) / (
                    (self.omega_c ** 2 + (2 * np.pi * l_min / beta) ** 2) ** 2 - (
                    self.gamma ** 2 * (2 * np.pi * l_min / beta) ** 2)) for l_min in range(self.l, self.N) ]).sum()
        return constant * sigma_sum


class Combine_Spectral(Spectral_OverDamped, Spectral_UnderDamped):
    def __init__(self, omega, omega_c, lamd, gamma, kBT, h_bar):
        super().__init__(omega, omega_c, lamd, gamma, kBT, h_bar)

    def __add__(self, other):
        return self.calculate() + other.calculate()

# # Constants:
# # ----------
# T = 295  # Unit : K
# kBT = 0.69503 * T  # Unit: cm^-1
# h_bar = 5308  # Unit: cm^-1 * fs
# gamma = 100  # Unit: cm^-1
# delta_t = 5  # Unit:fs
# t_list = np.arange(delta_t, 1000, delta_t)  # Unit : fs
# delta_omega = 1  # Unit: cm^-1
# omega_list = np.arange(delta_omega, 1000, delta_omega)  # Unit: cm^-1
# # omega_c = 500  # Unit: cm^-1
# # lamd = 50  # Unit : cm^-1
# omega_c = np.array([ [ 300 ], [ 518 ], [ 745 ] ])  # Unit: cm^-1
# lamd = np.array([ [161], [16], [48] ])  # Unit : cm^-1
#
# l = 0
# N = 40
#
# # # Object Instantiation:
# # # ---------------------
# SD = SpectralDensity(lamd, gamma, kBT, h_bar)
# SD_over1 = Spectral_OverDamped(omega_list, omega_c[ 0 ])  # the first over-damped
# SD_over2 = Spectral_OverDamped(omega_list, omega_c[ 1 ])  # the second over-damped
# SD_under1 = Spectral_UnderDamped(omega_list, omega_c[ 0 ])  # the first under-damped
# SD_under2 = Spectral_UnderDamped(omega_list, omega_c[ 1 ])  # the second under-damped
# SD_under3 = Spectral_UnderDamped(omega_list, omega_c[ 2 ])  # the second under-damped
#
# # Mats_over = Mats_overDamped(omega_list, omega_c, t_list, l, N)
# # Mats_under = Mats_underDamped(omega_list, omega_c, t_list, l, N)
#
# Total_SD = SD_over1.calculate() + SD_under1.calculate() + SD_under2.calculate() + SD_under3.calculate()  # construct 1 over & 1 under-damped mode
# Total_SD = Total_SD.reshape(999, 3)
# # plt.plot(omega_list, Total_SD)
# # plt.show()
#
# Correlation = CorrelationFunction()
#
# corr_over1 = Correlation_overDamped(omega_list, 60, t_list)
# corr_over2 = Correlation_overDamped(omega_list, 70, t_list)
# corr_under1 = Correlation_underDamped(omega_list, omega_c[ 0 ], t_list)
# corr_under2 = Correlation_underDamped(omega_list, omega_c[ 1 ], t_list)
# corr_under3 = Correlation_underDamped(omega_list, omega_c[ 2 ], t_list)
#
# Total_correlation = corr_over1.calculate() + corr_over2.calculate() + corr_under1.calculate() + corr_under2.calculate() + corr_under3.calculate()
# Total_correlation = Total_correlation.reshape(199, 3)
#
#
# plt.plot(t_list, Total_correlation)
# plt.show()


# # storing list:
# # -------------
# BCF_under_damped_highTemp = [ ]
# BCF_over_damped_highTemp = [ ]
# BCF_under_damped_lowTemp = [ ]
# BCF_over_damped_lowTemp = [ ]

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
#
# Total_BCF_highTemp = np.array(BCF_under_damped_highTemp) + np.array(BCF_over_damped_highTemp)
# Total_BCF_lowTemp = np.array(BCF_under_damped_lowTemp) + np.array(BCF_over_damped_lowTemp)
#
# Total_SD = SD.J_overDamped_mode() + SD.J_underDamped_mode(omega_c)  # construct 1 over & 1 under-damped mode
#
# # Plotting:
# # ---------
# # plt.plot(t_list, Total_BCF_highTemp)
# # plt.plot(t_list,Total_BCF_lowTemp)
# # plt.plot(Total_SD)
# # plt.plot(t_list, BCF_over_damped_lowTemp)
# # plt.plot(t_list, BCF_under_damped_lowTemp)
# # plt.show()
#
# fig, axs = plt.subplots(2)
# axs[ 0 ].plot(t_list, Total_BCF_highTemp, 'g')
# # axs[ 1 ].plot(t_list,Total_BCF_lowTemp, 'r')
# axs[ 1 ].plot(Total_SD, 'g--')
# plt.show()


#
