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

    # def __add__(self, other):
    #     return self.calculate() + other.calculate()


class Spectral_OverDamped(SpectralDensity):
    def __init__(self, omega, omega_c, lamd, gamma, kBT, h_bar):
        super().__init__(lamd, gamma, kBT, h_bar)
        self.omega = omega
        self.omega_c = omega_c

    # Class_methods:
    # ==============
    def calculate(self):
        return 2 * self.lamd * self.omega * self.gamma / (self.omega ** 2 + self.gamma ** 2)

    def __add__(self, other):
        return self.calculate() + other.calculate()


class Spectral_UnderDamped(SpectralDensity):
    def __init__(self, omega, omega_c, lamd, gamma, kBT, h_bar):
        super().__init__(lamd, gamma, kBT, h_bar)
        self.omega = omega
        self.omega_c = omega_c

    def calculate(self):
        return (2 * self.lamd * self.gamma * self.omega * self.omega_c ** 2) / (
                (self.omega_c ** 2 - self.omega ** 2) ** 2 + (self.omega ** 2) * (self.gamma ** 2))

    def __add__(self, other):
        return self.calculate() + other.calculate()


class CorrelationFunction(SpectralDensity):
    """
    This class is used to calculate the Bath Correlation Function for the over and under damped oscillation modes.
    """

    def __init__(self, lamd, gamma, kBT, h_bar):
        super().__init__(lamd, gamma, kBT, h_bar)

    def __resp__(self):
        pass
        
    def calculate(self):
        pass

    def plotting(self, x_axis):
        plt.plot(x_axis, self.calculate())  # ex: SpectralDensity.plotting(over,omega_list)


class Correlation_overDamped(Spectral_UnderDamped):
    def __init__(self, omega, omega_c, t, lamd, gamma, kBT, h_bar):
        super().__init__(omega, omega_c, lamd, gamma, kBT, h_bar)
        # self.t = t

    def calculate(self, t):
        """
        This function is used to calculate the overdamped mode at high temperature approximation:
        Equation:
        bcf(t) = lambda_k * gamma_k * (coth(beta*gamma_k/2)-1j) * exp(-gamma_k*t)

        """
        beta = 1 / self.kBT
        cot = 1 / np.tan((self.gamma * beta) / 2)
        return self.lamd * self.gamma * (cot - 1j) * np.exp(- self.gamma * t / self.h_bar)

    def __add__(self, other):
        return self.calculate() + other.calculate()

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

    def __add__(self, other):
        return self.calculate() + other.calculate()


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

