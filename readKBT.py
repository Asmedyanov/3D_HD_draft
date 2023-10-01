# =================================================================
# ReadKB :   Read file of KB, for example ('WATR7150.KBT')
#                       Rho - vector of water's density
#                       Temp - vector of temperature
#                       Pres - matrix if pressure
#                       Ener - matrix of energy
# calling the function: Rho=readKB('WATR7150.KBT') - function is returned
#                        only vector Rho
#                       [Rho,Temp]=readKB('WATR7150.KBT') - function is
#                       returned two vectores Rho, Temp.
#                       [Rho,Temp,Pres,Ener]= readKB('WATR7150.KBT') 
#                        - function is returned two vectores Rho, Temp and
#                        matrixes Pres,Ener
# ==================================================================
import numpy as np
import matplotlib.pyplot as plt

def readKB(filename):
    print(f"I read EOS {filename}")
    myfile = open(filename, 'r')
    myfile.readline()
    material = int(myfile.readline().split(' ')[-1])
    Rho0 = float(myfile.readline().split(' ')[-1])
    NRho = int(myfile.readline().split(' ')[-1])
    NTemp = int(myfile.readline().split(' ')[-1])
    Npress = NRho * NTemp
    Nenerg = Npress
    myfile.readline()
    Rho = []
    for i in range(NRho):
        Rho.append(float(myfile.readline().split(' ')[-1]))
    Rho = np.array(Rho)
    myfile.readline()
    Temp = []
    for i in range(NTemp):
        Temp.append(float(myfile.readline().split(' ')[-1]))
    Temp = np.array(Temp)
    myfile.readline()
    Pressure = []
    for i in range(Npress):
        value = float(myfile.readline())
        Pressure.append(value)
    Pressure = np.array(Pressure).reshape(NRho, NTemp)
    myfile.readline()
    Internal_energy = []
    for i in range(Nenerg):
        value = float(myfile.readline())
        Internal_energy.append(value)
    Internal_energy = np.array(Internal_energy).reshape(NRho, NTemp)
    plt.clf()
    plt.imshow((Internal_energy))
    plt.colorbar()
    plt.savefig("Internal_energy_com.png")
    plt.clf()
    plt.imshow((Pressure))
    plt.colorbar()
    plt.savefig("Pressure_com.png")
    plt.clf()

    return Rho*1.0e-3, Temp, Pressure*1.0e-11, Internal_energy