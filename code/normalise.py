



import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special

number = 10
n=np.zeros((number,1))
for i in range(number):
    n[i]=i+1
kl=np.zeros((number,1))
kl[0]=1.8751
for i in range(number-1):
    kl[i+1]=(n[i+1]-0.5)*math.pi

L=0.485
L2=0.33
E=210*10**9
p=7800
I=9*10**-10
A=3*10**-4
x=0.155
k=kl/L
k2=kl/L2

omegan=k*k*math.sqrt(E*I/(p*A))
omegan2=k2*k2*math.sqrt(E*I/(p*A))

freqarray=np.linspace(1,3500,3500)

Gyf=np.zeros(freqarray.shape,dtype=complex)
Gyft=np.zeros(freqarray.shape,dtype=complex)

D2=0.025
D2t=D2

D2sB=np.ones(k.shape)
D2sT=np.ones(k.shape)

for i in range(number):
    D1 = -D2sB[i] * ((np.cos(k[i] * L) + np.cosh(k[i] * L)) / (np.sin(k[i] * L) + np.sinh(k[i] * L))) ** -1
    result = integrate.quad(lambda x: (D1 * np.cos(k[i] * x) + D2sB[i]  * np.sin(k[i] * x) - D1 * np.cosh(k[i] * x) - D2sB[i] * np.sinh(k[i] * x))**2 *A *p , 0 , L)
    D2sB[i] = 1/ math.sqrt(result[0])
for i in range(number):
    D1 = -D2sT[i] * ((np.cos(k[i] * L2) + np.cosh(k[i] * L2)) / (np.sin(k[i] * L2) + np.sinh(k[i] * L2))) ** -1
    result = integrate.quad(lambda x: (D1 * np.cos(k[i] * x) + D2sT[i]  * np.sin(k[i] * x) - D1 * np.cosh(k[i] * x) - D2sT[i] * np.sinh(k[i] * x))**2 *A *p , 0 , L2)
    D2sT[i] = 1/ math.sqrt(result[0])

xs = np.linspace(0,L,200)
for i in range(4):
    D1= -D2sB[i] * ((np.cos(k[i]*L)+np.cosh(k[i]*L))/(np.sin(k[i]*L)+np.sinh(k[i]*L)))**-1
    ux1 = D1 * np.cos(k[i] * xs) + D2sB[i] * np.sin(k[i] * xs) - D1 * np.cosh(k[i] * xs) - D2sB[i] * np.sinh(k[i] * xs)
    plt.plot(xs,ux1 , Label = 'Mode ' + str(i+1))
#plt.ylim(-0.1,0.1)
plt.legend(loc='lower left')
plt.xlabel('x/ m')
plt.ylabel('Amplitude')
plt.title('Theoretical Mode Shapes')
plt.show()




zeta=0.005
for i in range(number):
    D1 = -D2sB[i] * ((np.cos(k[i] * L) + np.cosh(k[i] * L)) / (np.sin(k[i] * L) + np.sinh(k[i] * L))) ** -1
    ux = D1*np.cos(k[i]*x) + D2sB[i] *np.sin(k[i]*x) -D1 * np.cosh(k[i]*x) -D2sB[i] *np.sinh(k[i]*x)
    ux2=ux*ux
    Gyf +=ux2/(omegan[i]**2-freqarray**2 + 2*1j*omegan[i]*freqarray*zeta ) *1j*freqarray

    D1t = -D2sT[i]  * ((np.cos(k2[i] * L2) + np.cosh(k2[i] * L2)) / (np.sin(k2[i] * L2) + np.sinh(k2[i] * L2))) ** -1
    ux = D1t * np.cos(k2[i] * x) + D2sT[i]  * np.sin(k2[i] * x) - D1t * np.cosh(k2[i] * x) - D2sT[i]  * np.sinh(k2[i] * x)
    ux2 = ux * ux
    Gyft += ux2 / (omegan2[i] ** 2 - freqarray ** 2 + 2 * 1j * omegan2[i] * freqarray * zeta) *1j*freqarray
#Gyf = Gyf*10**3.1
#Gyft = Gyft*10**3.1
M= 5*10**-2 * 3.6*10**-2 * 1.5*10**-2 *p +0.04
M2=0.04
mtf=1/(freqarray**2 *M) *1j*freqarray
mtf2=1/(freqarray**2 *M2) *1j*freqarray

plt.plot(freqarray/(2*math.pi),20*np.log10(abs(Gyf)),Label = 'Lower Beam')
plt.plot(freqarray/(2*math.pi),20*np.log10(abs(Gyft)),Label = 'Upper Beam')
plt.plot(freqarray/(2*math.pi),20*np.log10(abs(Gyf*Gyft/(Gyf+Gyft))),Label = 'Coupled Beams')
massandtop=(Gyft*-mtf)/(Gyft-mtf)
massandbot=(Gyf*-mtf2)/(Gyf-mtf2)
plt.plot(freqarray/(2*math.pi),20*np.log10(abs(massandtop)),Label = 'Coupled Upper Beam and Mass')
plt.plot(freqarray/(2*math.pi),20*np.log10(abs(massandbot)),Label = 'Coupled Lower Beam and Mass')
plt.plot(freqarray/(2*math.pi),20*np.log10(abs((massandtop*massandbot)/(massandtop+massandbot) )) , Label = 'Coupled Beams and Both Masses')
plt.plot(freqarray/(2*math.pi),20*np.log10(mtf),Label = 'Top Mass')
plt.axvline(x=23, color = 'black' ,linestyle = '--', Label = 'Measured Resonant Frequencies')
plt.axvline(x=44, color = 'black',linestyle = '--')
plt.axvline(x=130, color = 'black',linestyle = '--')
plt.axvline(x=370, color = 'black',linestyle = '--')
plt.axvline(x=237, color = 'black',linestyle = '--')
plt.legend(loc='lower right')
plt.xlabel('Frequency/ Hz')
plt.ylabel('20Log|G(f)|')
plt.title('Theoretical Transfer Functions')
plt.ylim(-110,0)
plt.xlim(0,500)
plt.show()

zetas=[0.001,0.005,0.01,0.1]
for j in range(4):
    zeta = zetas[j]
    Gyf = np.zeros(freqarray.shape, dtype=complex)
    Gyft = np.zeros(freqarray.shape, dtype=complex)
    for i in range(number):
        D1 = -D2 * ((np.cos(k[i] * L) + np.cosh(k[i] * L)) / (np.sin(k[i] * L) + np.sinh(k[i] * L))) ** -1
        ux = D1 * np.cos(k[i] * x) + D2 * np.sin(k[i] * x) - D1 * np.cosh(k[i] * x) - D2 * np.sinh(k[i] * x)
        ux2 = ux * ux
        Gyf += ux2 / (omegan[i] ** 2 - freqarray ** 2 + 2 * 1j * omegan[i] * freqarray * zeta) * 1j * freqarray

        D1t = -D2t * ((np.cos(k2[i] * L2) + np.cosh(k2[i] * L2)) / (np.sin(k2[i] * L2) + np.sinh(k2[i] * L2))) ** -1
        ux = D1t * np.cos(k2[i] * x) + D2t * np.sin(k2[i] * x) - D1t * np.cosh(k2[i] * x) - D2t * np.sinh(k2[i] * x)
        ux2 = ux * ux
        Gyft += ux2 / (omegan2[i] ** 2 - freqarray ** 2 + 2 * 1j * omegan2[i] * freqarray * zeta) * 1j * freqarray
    Gyf = Gyf * 10 ** 3.1
    Gyft = Gyft * 10 ** 3.1
    M = 5 * 10 ** -2 * 3.6 * 10 ** -2 * 1.5 * 10 ** -2 * p + 0.04
    M2 = 0.04
    mtf = 1 / (freqarray ** 2 * M) * 1j * freqarray
    mtf2 = 1 / (freqarray ** 2 * M2) * 1j * freqarray
    massandtop = (Gyft * -mtf) / (Gyft - mtf)
    massandbot = (Gyf * -mtf2) / (Gyf - mtf2)
    plt.plot(freqarray / (2 * math.pi), 20 * np.log10(abs((massandtop * massandbot) / (massandtop + massandbot))),
             Label='Zeta = ' + str(zeta))
plt.legend(loc='lower right')
plt.xlabel('Frequency/ Hz')
plt.ylabel('20Log|G(f)|')
plt.title('Theoretical Transfer Functions Comparing Damping Ratio')
plt.ylim(-120,0)
plt.xlim(0,350)
plt.show()

