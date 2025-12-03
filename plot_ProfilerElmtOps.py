import numpy as np
import matplotlib.pyplot as plt

oper_names = ["Mass", "Stiffness", "Helmholtz"]

for oper_name in oper_names:
    file_name = "log_mfem_profilerElmtOps_" + oper_name + ".log"
    dat = np.loadtxt(file_name)
    total_dofs = np.zeros((7, 6))
    dofs_per_second = np.zeros((7, 6))
    labels = []

    fig, ax = plt.subplots()
    colors = [
        "tab:green",
        "tab:cyan",
        "tab:purple",
        "tab:brown",
        "tab:orange",
        "tab:red",
        "tab:olive",
    ]
    for i in range(1, 8):
        indices = np.where(dat[:, 2] == i)
        print(len(indices[0]))
        print(dat[indices, 1])
        print(dat[indices, 3])
        total_dofs[i - 1, : len(indices[0])] = dat[indices, 1]
        dofs_per_second[i - 1, : len(indices[0])] = dat[indices, 3]
        label = f"p = {i}"
        ax.semilogx(
            dat[indices, 1][0],
            dat[indices, 3][0],
            label=label,
            color=colors[i - 1],
            linestyle="-",
            marker="s",
        )
        ax.set_title("MFEM " + oper_name + " Operator, Partial Assembly, MatXVec")

    ax.legend(prop={"size": 8})
    ax.set_xlabel("DOFs [1]")
    ax.set_ylabel("DOFs/sec [1/s]")
    ax.legend()
    plt.savefig(oper_name + "_MFEM_GH200.png")
    plt.show()
