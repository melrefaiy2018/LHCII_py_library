import matplotlib.pyplot as plt
import numpy as np


class SpectralDensity:
    """
       This class is used to build the spectral density for the over and under oscillation mode.
       """
    def __init__(self, omega, lamd, gamma):
        self.omega = omega
        self.lamd = lamd
        self.gamma = gamma
        # self.omega = omega
        # self.omega_c = omega_c

    def calculate(self):
        pass

    def plotting(self, x_axis):
        plt.plot(x_axis, self.calculate())  # ex: SpectralDensity.plotting(over,omega_list)


class SdOverDamped(SpectralDensity):
    def __init__(self, omega):
        super().__init__(omega, lamd, gamma)
        self.omega = omega


    # Class_methods:
    # ==============
    def calculate(self):
        return 2 * self.lamd * self.omega * self.gamma / (self.omega ** 2 + self.gamma ** 2)


class SdUnderDamped(SpectralDensity):
    def __init__(self, omega, omega_c):
        super().__init__(omega, lamd, gamma)
        self.omega = omega
        self.omega_c = omega_c


    def calculate(self):
        return (2 * self.lamd * self.gamma * self.omega * self.omega_c ** 2) / (
                (omega_c ** 2 - self.omega ** 2) ** 2 + (self.omega ** 2) * (self.gamma ** 2))

# =======================================
class CorrelationFunction:
    """
    This class is used to calculate the Bath Correlation Function for the over and under damped oscillation modes.
    """

    def __init__(self, lamd, gamma, kBT, h_bar):
        self.kBT = kBT
        self.h_bar = h_bar
        self.lamd = lamd
        self.gamma = gamma

    def calculate(self, omega, omaga_c, t):
        pass

    def plotting(self, x_axis):
        plt.plot(x_axis, self.calculate(self, omega, omega_c, t))  # ex: SpectralDensity.plotting(over,omega_list)


class COverDamped(CorrelationFunction):
    def __init__(self):
        super().__init__(lamd, gamma, kBT, h_bar)

    def calculate(self, omega, omaga_c, t):
        """
        This function is used to calculate the overdamped mode at high temperature approximation:
        Equation:
        bcf(t) = lambda_k * gamma_k * (coth(beta*gamma_k/2)-1j) * exp(-gamma_k*t)

        """
        beta = 1 / self.kBT
        cot = 1 / np.tan((gamma * beta) / 2)
        return lamd * gamma * (cot - 1j) * np.exp(-gamma * t / h_bar)


class CUnderDamped(CorrelationFunction):
    def __init__(self):
        super().__init__(lamd, gamma, omega_c, kBT)

    def calculate(self, omega, omega_c, t):
        """
        This function is used to calculate the under-damped mode at high temperature approximation:
        FORMULA:
        bcf(t) = lambda_k * omega_k**2/(2*chi_k) * [coth(beta*(chi_k+1j*gamma_k/2)/2)-1] * exp(-(gamma_k/2-1j*chi_k)*t)
                +lambda_k * omega_k**2/(2*chi_k) * [-coth(beta*(-chi_k+1j*gamma_k/2)/2)+1] * exp(-(gamma_k/2+1j*chi_k)*t)

        """
        beta = 1 / self.kBT
        ki = np.sqrt(omega_c ** 2 - (self.gamma ** 2 / 4))

        coth_plus_analytical = 1 / np.tanh(beta * (ki + 1j * self.gamma / 2) / 2)
        coth_negative_analytical = 1 / np.tanh(beta * (- ki + 1j * self.gamma / 2) / 2)

        exp_plus_analytical = np.exp(-t * (self.gamma / 2 + 1j * ki) / self.h_bar)
        exp_negative_analytical = np.exp(-t * (self.gamma / 2 - 1j * ki) / self.h_bar)
        constant = self.lamd * omega_c ** 2 / (2 * ki)

        return constant * ((coth_plus_analytical(beta, ki) - 1) * exp_negative_analytical(t, ki) + (
                - coth_negative_analytical(beta, ki) + 1) * exp_plus_analytical(t, ki))

class MatsubaraApproximation:
    """
    This class is used to calculate the matsubura approximation for both the over and under oscillation modes.
    # Low Temp approximation: gamma * T > 1
    """

    def __init__(self, omega, lamd, gamma, omega_c):
        super().__init__(omega, lamd, gamma, omega_c, kBT)

    def Mats_overDamped(self, t, l, N):
        """
        This is the Matsubara approximation for the over-damped mode at the low temperature approximation.
        Equation:
        - 4 * lambda_k * gamma_k/beta * sum_l(eta_l * nu_l/( gamma_k**2-nu_l**2)*exp(-nu_l*t) )
        with beta = 1/temp
        """
        beta = 1 / kBT
        eta = 1
        # neu = 2 * np.pi * l / beta
        sigma_sum = np.array(
            [ (eta * 2 * np.pi * l_min / beta) * np.exp(- 2 * np.pi * l_min / beta * t / h_bar) / (
                    gamma ** 2 - (2 * np.pi * l_min / beta) ** 2) for l_min in range(l, N) ]).sum()
        return sigma_sum * 4 * lamd * gamma / beta

    def Mats_underDamped(self, t, L_min, N):
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
            [ (eta * 2 * np.pi * l / beta) * np.exp((- 2 * np.pi * l / beta) * t / self.h_bar) / (
                    (self.omega_c ** 2 + (2 * np.pi * l / beta) ** 2) ** 2 - (
                    self.gamma ** 2 * (2 * np.pi * l / beta) ** 2)) for
              l in range(L_min, N) ]).sum()
        return constant * sigma_sum


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

# Example:
# =======
# overDamped = SdOverDamped(omega_list)
# SD = SpectralDensity
# SD.plotting(overDamped ,omega_list)
# underDamped = SdUnderDamped(omega_list, omega_c)
# SD.plotting(overDamped, omega_list) + SD.plotting(underDamped, omega_list)
