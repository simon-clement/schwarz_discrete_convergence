#!/usr/bin/python3
"""
    This setup script is here to handle dependencies.
"""
import os

os.system('pip3 install scipy')
os.system('pip3 install matplotlib')
try:
    os.system('sudo apt install python3-tk gfortran libopenblas-dev liblapack-dev')
    os.system('curl https://sh.rustup.rs -sSf | sh')
    os.system('pip3 install cffi')
    os.system("echo \'# added by setup script of schwarz_tbc_finder\'" +
              ">> ~/.bashrc")
    os.system("echo \'export PATH=~/.cargo/bin:$PATH\'" + ">> ~/.bashrc")
except BaseException:
    print("Cannot install fast version.")
    print("The code will be able to run but slowly.")
