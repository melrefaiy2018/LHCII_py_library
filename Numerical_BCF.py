import matplotlib.pyplot as plt
import numpy as np


# from SD_BCF_OOP import SpectralDensity, Spectral_OverDamped, Spectral_UnderDamped, CorrelationFunction, \
#     Correlation_overDamped, Correlation_underDamped


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


# Correlation class
# =================
class NumericalCorrelation(SpectralDensity):
    def __init__(self, lamd, gamma, h_bar, kBT):
        super().__init__(lamd, gamma, kBT, h_bar)

    def calculate(self):
        pass

    def plotting(self, x_axis):
        plt.plot(x_axis, self.calculate())


class NumericalCorrUnderDamped(NumericalCorrelation):
    def __init__(self, omega, omega_c, t, lamd, gamma, h_bar, kBT):
        super().__init__(lamd, gamma, h_bar, kBT)
        self.omega = omega
        self.omega_c = omega_c
        self.t = t

    def calculate(self):
        coth = 1 / np.tanh(self.omega / (2 * self.kBT))
        return (1 / np.pi) * Spectral_UnderDamped.calculate() * (
                coth * np.cos(self.omega * self.t / self.h_bar))


class NumericalCorrOverDamped(NumericalCorrelation):
    def __init__(self, omega, omega_c, t, lamd, gamma, h_bar, kBT):
        super().__init__(lamd, gamma, h_bar, kBT)
        self.omega = omega
        self.omega_c = omega_c
        self.t = t

    def calculate(self):
        coth = 1 / np.tanh(self.omega / (2 * self.kBT))
        return (1 / np.pi) * Spectral_OverDamped.calculate() * (
                coth * np.cos(self.omega * self.t / self.h_bar))


# Imaginary part:
# ===============
class NumericalCorrImaUnderDamped(NumericalCorrelation):
    def __init__(self, omega, omega_c, t, lamd, gamma, h_bar, kBT):
        super().__init__(lamd, gamma, h_bar, kBT)
        self.omega = omega
        self.omega_c = omega_c
        self.t = t

    def calculate(self):
        (- 1 / np.pi) * np.sin(self.omega * self.t / self.h_bar) * (Spectral_UnderDamped.calculate())


class NumericalCorrImaOverDamped(NumericalCorrelation):
    def __init__(self, omega, omega_c, t, lamd, gamma, h_bar, kBT):
        super().__init__(lamd, gamma, h_bar, kBT)
        self.omega = omega
        self.omega_c = omega_c
        self.t = t

    def calculate(self):
        (- 1 / np.pi) * np.sin(self.omega * self.t / self.h_bar) * Spectral_OverDamped.calculate()
