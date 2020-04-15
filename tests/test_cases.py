"""
    Simple test module of finite_volumes
"""
import numpy as np
from numpy import cos, sin, pi

def linear_time(a, c, D):
    def u_real(x, t): return np.sin(x) + t
    def u_sec(x, t): return -np.sin(x)
    def u_bar(x_1_2, h, t): return 1/h * (np.cos(x_1_2) - np.cos(x_1_2+h)) + t
    def flux(x, t): return D * np.cos(x)
    def f(x, t): return c*u_real(x, t) + 1 - (- D * np.sin(x))
    def f_bar(x, h, t): return 1 + c*t - 1/h * (D+c) * (np.cos(x+h) - np.cos(x))
    def f_bar_sec(x, h, t): return 1/h * (D+c) * (np.cos(x+h) - np.cos(x))
    dico_ret = {'u':u_real,
            'u^(2)':u_sec,
            'Bar{u}':u_bar,
            'Ddu/dx':flux,
            'f':f,
            'Bar{f}':f_bar,
            'Bar{f}^(2)':f_bar_sec}
    return dico_ret

def cosinus_time(a, c, D):
    def u_real(x, t): return np.sin(x) + np.cos(t)
    def u_sec(x, t): return -np.sin(x)
    def u_bar(x_1_2, h, t): return 1/h * (np.cos(x_1_2) - np.cos(x_1_2+h)) + np.cos(t)
    def flux(x, t): return D * np.cos(x)
    def f(x, t): return c*u_real(x, t) - np.sin(t) - (- D * np.sin(x))
    def f_bar(x, h, t): return -np.sin(t)+c*np.cos(t) - 1/h * (D+c) * (np.cos(x+h) - np.cos(x))
    def f_bar_sec(x, h, t): return 1/h * (D+c) * (np.cos(x+h) - np.cos(x))
    dico_ret = {'u':u_real,
            'u^(2)':u_sec,
            'Bar{u}':u_bar,
            'Ddu/dx':flux,
            'f':f,
            'Bar{f}':f_bar,
            'Bar{f}^(2)':f_bar_sec}
    return dico_ret

def intricated_spacetime(a, c, D):
    def u_real(x, t): return np.sin(x + t)
    def u_sec(x, t): return -np.sin(x+t)
    def u_bar(x_1_2, h, t): return 1/h * (np.cos(x_1_2 + t) - np.cos(x_1_2+h + t))
    def flux(x, t): return D * np.cos(x + t)
    def f(x, t): return c*u_real(x, t) + np.cos(x+t) - ( - D * np.sin(x+t))
    def f_bar(x, h, t): return 1/h*(np.sin(x+h+t) - np.sin(x+t)) - 1/h * (D+c) * (np.cos(x+h+t) - np.cos(x+t))
    def f_bar_sec(x, h, t): return - 1/h*(np.sin(x+h+t) - np.sin(x+t)) + 1/h * (D+c) * (np.cos(x+h+t) - np.cos(x+t))
    dico_ret = {'u':u_real,
            'u^(2)':u_sec,
            'Bar{u}':u_bar,
            'Ddu/dx':flux,
            'f':f,
            'Bar{f}':f_bar,
            'Bar{f}^(2)':f_bar_sec}
    return dico_ret

def exp_space_quad_time(a, c, D):
    def u_real(x, t): return np.exp(x) * t**2
    def u_sec(x, t): return np.exp(x) * t**2
    def u_bar(x_1_2, h, t): return t**2/h * (np.exp(x_1_2+h) - np.exp(x_1_2))
    def flux(x, t): return D * np.exp(x) * t**2
    def f(x, t): return c*u_real(x, t) + 2*t*np.exp(x) - D * np.exp(x)*t**2
    def f_bar(x, h, t): return 2*t/h*(np.exp(x+h) - np.exp(x)) - 1/h * (D-c) * (np.exp(x+h) - np.exp(x))*t**2
    def f_bar_sec(x, h, t): return 2*t/h*(np.exp(x+h) - np.exp(x)) - 1/h * (D-c) * (np.exp(x+h) - np.exp(x))*t**2
    dico_ret = {'u':u_real,
            'u^(2)':u_sec,
            'Bar{u}':u_bar,
            'Ddu/dx':flux,
            'f':f,
            'Bar{f}':f_bar,
            'Bar{f}^(2)':f_bar_sec}
    return dico_ret


def const_space_quad_time(a, c, D):
    def u_real(x, t): return t**2 + np.zeros_like(x)
    def u_sec(x, t): return np.zeros_like(x)
    def u_bar(x_1_2, h, t): return t**2 +np.zeros_like(x_1_2)
    def flux(x, t): return 0 + np.zeros_like(x)
    def f(x, t): return 2*t + np.zeros_like(x)
    def f_bar(x, h, t): return 2*t + np.zeros_like(x)
    def f_bar_sec(x, h, t): return np.zeros_like(x)

    dico_ret = {'u':u_real,
            'u^(2)':u_sec,
            'Bar{u}':u_bar,
            'Ddu/dx':flux,
            'f':f,
            'Bar{f}':f_bar,
            'Bar{f}^(2)':f_bar_sec}
    return dico_ret

