import numpy as np
import matplotlib.pyplot as plt


# mCp = np.array([50,10,70,15])
# T_in = np.array([200,300,90,40,30,240])
# T_fin = np.array([70,60,180,220,40,240])
# Q = []

# for i in range(len(mCp)):
#     Q.append(int(mCp[i] * (T_in[i] - T_fin[i])))

# print(Q)

# Q_needed = np.sum(Q)


hot_stream = {
    "H1": {
        "mCp": 50,
        "InletT": 200,
        "FinalT": 70,
    },
    "H2": {
        "mCp": 10,
        "InletT": 300,
        "FinalT": 60,
    }
}
cold_stream = {       
    "C1": {
        "mCp": 70,
        "InletT": 90,
        "FinalT": 180,
    },
    "C2": {
        "mCp": 15,
        "InletT": 40,
        "FinalT": 220,
    },
}


def calc_q(stream):
    q_list = []
    q_list_tot = []
    for duty in stream.values(): 
        t_diff = abs(duty['InletT'] - duty['FinalT'])
        q = duty['mCp'] * t_diff

        q_list.append(q)

        q_tot = 0
        for q_itr in q_list_tot:
            q_tot += q_itr
        q_tot += q

        q_list_tot.append(q_tot)

    return q_list, q_list_tot

q_hot, q_hot_tot  = calc_q(hot_stream)
q_cold, q_cold_tot  = calc_q(cold_stream)

def get_temp_list(stream):
    temp_list = []
    for temp in stream.values():
        temp_list.append(temp['FinalT'])
    return temp_list

hot_temps = get_temp_list(hot_stream)
cold_temps = get_temp_list(cold_stream)

#print(q_hot,q_hot_tot,hot_temps)

plt.figure(figsize=(8,8))

# plt.plot([q_hot_tot[0],0] ,[hot_stream["H1"]["InletT"],hot_stream["H1"]["FinalT"]] , marker='o', linestyle='-', color="purple", label="Hot Composite Curve")
# plt.plot([q_hot_tot[1],q_hot_tot[0]] ,[hot_stream["H2"]["InletT"],hot_stream["H2"]["FinalT"]] , marker='o', linestyle='-', color="purple", label="Hot Composite Curve")

# plt.plot([q_hot_tot[1],q_hot_tot[1] + q_cold_tot[0]] ,[cold_stream["C1"]["InletT"],cold_stream["C1"]["FinalT"]] , marker='o', linestyle='-', color="blue", label="cold Composite Curve")
# plt.plot([q_hot_tot[1] + q_cold_tot[0],q_hot_tot[1] + q_cold_tot[1]] ,[cold_stream["C2"]["InletT"],cold_stream["C2"]["FinalT"]] , marker='o', linestyle='-', color="blue", label="cold Composite Curve")

# #plt.plot(, , marker='o', linestyle='-', color="blue", label="Cold Composite Curve")
# plt.plot(1000,100,color="white")
# plt.show()

# all_temp = [hot_stream["H1"]["InletT"],hot_stream["H1"]["FinalT"],hot_stream["H2"]["InletT"],hot_stream["H2"]["FinalT"]]
# all_temp.sort()

# print(all_temp)





































# H1 = [T_,200]
# HQ1 = [0,Q[0]]
# H2 = [10,300]
# HQ2 = [HQ1[1],HQ1[1]+Q[1]]
# C1 = []

# plt.figure(figsize=(6,6))
# plt.plot(20000,1,color="white")
# plt.plot(HQ1,H1,color="red")
# plt.plot(HQ2,H2,color="red")
# plt.plot()

# plt.show()