# import numpy as np
# import matplotlib.pyplot as plt
#
# dat = np.loadtxt('test.dat')
# total_dofs = np.zeros((7, 5))
# dofs_per_second = np.zeros((7, 5))
# labels = []
#
# fig, ax = plt.subplots()
# for i in range(1,8):
#     indices = np.where(dat[:,2] == i)
#     total_dofs[i-1,:] = dat[indices,1]
#     dofs_per_second[i-1,:] = dat[indices,3]
#     labels.append(f"p = {i}")
#
# ax.plot(total_dofs, dofs_per_second, "-o", labels=labels)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt('bp1_hex_mfem.dat')
total_dofs = np.zeros((7, 6))
dofs_per_second = np.zeros((7, 6))
labels = []

fig, ax = plt.subplots()
colors = ["tab:green", "tab:cyan", "tab:purple", "tab:brown", "tab:orange", "tab:red", "tab:olive"]
for i in range(1, 8):
    indices = np.where(dat[:, 2] == i)
    print(len(indices[0]))
    print(dat[indices, 1])
    print(dat[indices, 3])
    total_dofs[i-1, :len(indices[0])] = dat[indices, 1]
    dofs_per_second[i-1, :len(indices[0])] = dat[indices, 3]
    label = f"p = {i}"
    ax.semilogx(dat[indices, 1][0], dat[indices, 3][0], label=label, color=colors[i-1], linestyle="-", marker="s")
    ax.set_title("MFEM Mass Operator, Partial Assembly, MatXVec")

ax.legend(prop={'size': 8})
ax.set_xlabel("DOFs [1]")
ax.set_ylabel("DOFs/sec [1/s]")
ax.legend()
plt.savefig("mfem_profile_mass_matrix.png")
plt.show()

