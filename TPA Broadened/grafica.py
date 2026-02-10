import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(
        "TomaDeDatos50C-822nm-3V-100mHz_Triangular.txt",
        sep='\t',
        header=None,
        names=['nose','wavelength[nm]','counts[s]','nose2']
)
fig, axis = plt.subplots()
axis.scatter(df["wavelength[nm]"][1:],df["counts[s]"][1:])
axis.set_xlim(822.4680,822.4700)
axis.set(xlabel="wavelength[nm]",ylabel="counts per second",title="Counts vs Wavelength")
fig.savefig("prueba.png")

