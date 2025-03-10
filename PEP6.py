import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt



# def results(Q11,Q12,Q21,Q22)
    
#     return Ccop, Cop, GHG, TAC

# for guess in [1,1000]:
#     Q = rand(4,1)
#     Q_alternativ = sp.LatinHypercube(d=2)
#     obj = results(Q)

# plot

TH1cold, TH1Hot = 25, 90
TH2cold, TH2Hot = 30, 80
TC1cold, TC1Hot = 15, 70
TC2cold,TC2Hot = 25, 60
mCp1H = 7.69
mCp2H = 8.00
mCp1C = 10.91
mCp2C = 11.43

H1req, H2req, C1req, C2req = 500,400,600,400

def logMeanTemp(Tleft,Tright):
    if Tleft <= 0 or Tright <= 0:
        # Handle the error appropriately: you might return a default value or raise an error
        return 1
    return (Tleft-Tright)/(np.log(Tleft/Tright))

Tlms = []

def calculations(Q11,Q12,Q21,Q22):

    if any(q < 0 for q in [Q11, Q12, Q21, Q22]):
        return None


    Qs = [Q11,Q12,Q21,Q22]
    Tlm = []
    A = []
    Atot = 0
    U = 0.075

    TH1A = TH1Hot-Q11/mCp1H
    TH1B = TH1A - Q12/mCp1H
    TH2A = TH2Hot-Q22/mCp2H
    TH2B = TH2A - Q21/mCp2H

    TC1A = TC1cold + Q21/mCp1C
    TC1B = TC1A + Q11/mCp1C
    TC2A = TC2cold + Q12/mCp1C
    TC2B = TC2A + Q22/mCp1C    

    if Q11 + Q12 > H1req or Q21 + Q22 > H2req:
        return None
    if Q11 + Q21 > C1req or Q12 + Q22 > C2req:
        return None
    
    # Check 2nd law of thermodynamics (hot stream must be hotter than cold stream)
    if Q11 > 0 and (TH1_hot <= TC1B or TH1A <= TC1A):
        return None
    if Q12 > 0 and (TH1A <= TC2B or TH1B <= TC2A):
        return None
    if Q21 > 0 and (TH2A <= TC1B or TH2B <= TC1A):
        return None
    if Q22 > 0 and (TH2_hot <= TC2B or TH2A <= TC2A):
        return None

    #Tlm for Q11
    if Q21 == 0:
        TCIn11 = TC1cold
    else:
        TCIn11 = TC1A
    Tlm.append(logMeanTemp(TH1A-TCIn11,TH1Hot-TC1B))

    #Tlm for Q12
    if Q11 == 0:
        THIn12 = TH1Hot
    else:
        THIn12 = TH1A
    Tlm.append(logMeanTemp(TH1B-TC2cold,THIn12-TC2A))

    #Tlm for Q21
    if Q22 == 0:
        THIn21 = TH2Hot
    else:
        THIn21 = TH2A
    Tlm.append(logMeanTemp(TH2B-TC1cold,THIn21-TC1A))

    #Tlm for Q22
    if Q12 == 0:
        TCIn22 = TC2cold
    else:
        TCIn22 = TC2A 

    Tlm.append(logMeanTemp(TH2A-TCIn22,TH2Hot-TC2B))

    for Q, Tlm in zip(Qs,Tlm):
        A.append(Q/(U*Tlm))
        Atot += Q/(U*Tlm)

    CW = H1req - Q11 - Q12 + H2req - Q21 - Q22
    LPS = C1req - Q11 - Q21 + C2req - Q12 - Q22

    OperationCost = (LPS*0.039 + CW*0.048)*20000*3600
    CapitalCost = float(15000000 + 11000*Atot)



    return OperationCost, CapitalCost

opeList = []
capList = []
numbber_of_exchangers = 0


for i in range(1000):
    Q11 = random.randint(10,500)
    Q12 = random.randint(10,500)
    Q21 = random.randint(10,500)
    Q22 = random.randint(10,500)
    opeCost, capCost = calculations(Q11,Q12,Q21,Q22)
    if opeCost and capCost is not None:
        opeList.append(opeCost)
        capList.append(capCost)
        numbber_of_exchangers += 1

print(numbber_of_exchangers)




plt.plot(opeList,capList,linestyle="",marker="o")
plt.show()    
