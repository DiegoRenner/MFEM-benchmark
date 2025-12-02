import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("bp1_hex_mfem.dat")
print("#modes #elmts #dofs seconds/pcg_iter")
print(data)

colors = ["#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#045a8d", "#023858"]
fig, ax = plt.subplots(1, 1)
fig.suptitle("Bake-off Problem 1, MFEM")
for nm in range(2, 9):
    mask = np.where(data[:, 0] == nm)
    print(data[mask, 2])
    print(mask)
    ax.semilogx(
        data[mask, 2][0],
        (data[mask, 2][0] / data[mask, 3][0]) * 1e6,
        label=f"nm={nm}",
        linestyle="-",
        marker="s",
    )  # , color=colors[nm-2])

ax.set_xlabel("unique DoFs [1]")
ax.set_ylabel("unique DoFs / second [1/s]")
ax.legend()

# for nm in range(2, 9):
#    mask = np.where(data[:, 0] == nm)
#    print(mask)
#    ax[1].semilogx(
#        data[mask, 3][0],
#        data[mask, 3][0] / data[mask, 4][0],
#        label=f"nm={nm}",
#        linestyle="-",
#        marker="s",
#    )  # , color=colors[nm-2])
#
# ax[1].set_xlabel("DoFs [1]")
# ax[1].set_ylabel("DoFs / second [1/s]")
plt.tight_layout()
plt.savefig("bp1_mfem.png")
plt.show()
plt.close()
