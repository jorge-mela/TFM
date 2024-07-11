import os
import sys

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./")
from libs.generales import obtener_dataset


norm = Normalize(vmin=-1, vmax=1, clip=True)


def obtener_dataset(path: str, nombreData: str):  # --> data: np.ndarray
    """
    Retorna un ndarray con los valores del dataset seleccionado.

    ## Parámetros
    - path: String con el path hasta el archivo .hd5f
    - nombreData: String con el nombre del dataset dentro del archivo
    """

    if not os.path.isfile(path):
        exStr = f"No such file or directory {path}"
        raise ValueError(exStr)

    f = h5py.File(path, "r")

    if nombreData not in f.keys():
        keys = f.keys()
        exStr = f"No such key ({nombreData}) in {path} dataset. Available keys: {keys}"
        f.close()
        raise ValueError(exStr)

    data = np.asarray(f[nombreData]).copy()
    f.close()

    return data


datos = obtener_dataset(
    "./Experimento_05/proyecciones/Deep_01_proyecciones_3_K1.hdf5", "muestras"
)

print(datos.shape)

# # Hacemos un clipping
# s = [slice(None)] * len(datos.shape)
# for i in range(datos.shape[-1]):
#     s[-1] = i
#     canal = datos[tuple(s)]
#     pr1  = np.percentile(canal, 1)
#     pr99 = np.percentile(canal, 99)
#     canal[canal<pr1] = pr1
#     canal[canal>pr99] = pr99
#     datos[tuple(s)] = canal

# Normalizamos entre -1 y 1
datos = MinMaxScaler((-1, 1)).fit_transform(datos)

proyecciones = obtener_dataset(
    "./Experimento_05/proyecciones/Deep_01_proyecciones_3_K1.hdf5", "espacioLatente"
)

labels = ["rotacional", "divergencia", "Ex", "Ey", "Txy", "bx", "by"]

s = [slice(None)] * len(datos.shape)

# for i in range(datos.shape[-1]):
#     s[-1] = i
#     fig = plt.figure(labels[i])
#     ax = fig.add_subplot(projection='3d')

#     ax.scatter(proyecciones[:,0], proyecciones[:,1], proyecciones[:,2], alpha=0.4, c=datos[tuple(s)], cmap="seismic")
#     ax.set_title(labels[i])
#     fig.colorbar(cm.ScalarMappable(norm=norm, cmap="seismic"), ax=ax)

#     ax.set_xlabel('Componente 1')
#     ax.set_ylabel('Componente 2')
#     ax.set_zlabel('Componente 3')

fig = plt.figure(1)
axes = []
for i in range(datos.shape[-1]):
    s[-1] = i
    axes.append(fig.add_subplot(2, 4, i + 1, projection="3d"))

    axes[i].scatter(
        proyecciones[:, 0],
        proyecciones[:, 1],
        proyecciones[:, 2],
        alpha=0.4,
        c=datos[tuple(s)],
        cmap="seismic",
    )
    axes[i].set_title(labels[i])

    axes[i].set_xlabel("Componente 1")
    axes[i].set_ylabel("Componente 2")
    axes[i].set_zlabel("Componente 3")

ax = fig.add_subplot(2, 4, 8)
ax.axis("off")

fig.colorbar(cm.ScalarMappable(norm=norm, cmap="seismic"), ax=ax)  # ax=axes[i]


def on_move(event):
    for ax in axes:
        if event.inaxes == ax:
            if ax.button_pressed in ax._rotate_btn:
                aux = [a for a in axes if a is not ax]
                for ax_2 in aux:
                    ax_2.view_init(elev=ax.elev, azim=ax.azim)
            elif ax.button_pressed in ax._zoom_btn:
                aux = [a for a in axes if a is not ax]
                for ax_2 in aux:
                    ax_2.set_xlim3d(ax.get_xlim3d())
                    ax_2.set_ylim3d(ax.get_ylim3d())
                    ax_2.set_zlim3d(ax.get_zlim3d())

    fig.canvas.draw_idle()


c1 = fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()
