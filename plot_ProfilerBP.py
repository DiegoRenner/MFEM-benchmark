import numpy as np
import matplotlib.pyplot as plt

oper_names = ["Mass", "Helmholtz"]
for oper_name in oper_names:
    data = np.loadtxt("log_mfem_profilerBP_" + oper_name + ".log")
    print("#modes #elmts #dofs seconds/pcg_iter")
    print(data)

    colors = [
        "#d0d1e6",
        "#a6bddb",
        "#74a9cf",
        "#3690c0",
        "#0570b0",
        "#045a8d",
        "#023858",
    ]
    fig, ax = plt.subplots(1, 1)
    if oper_name == "Mass":
        fig.suptitle("Bake-off Problem 1, MFEM")
    else:
        fig.suptitle("Bake-off Problem 3, MFEM")
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
        )

    ax.set_xlabel("unique DoFs [1]")
    ax.set_ylabel("unique DoFs / second [1/s]")
    ax.legend()

    plt.tight_layout()
    if oper_name == "Mass":
        plt.savefig(oper_name + "_BP1_MFEM_GH200.png")
    else:
        plt.savefig(oper_name + "_BP3_MFEM_GH200.png")
    plt.show()
    plt.close()
