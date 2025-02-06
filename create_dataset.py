import numpy as np
import pandas as pd

def create_dataset_old(P, T, Q, Angles, SurfType, SurfGeom, Skin, s2m, CO2, DateTimes, Result):
    # Combine the arrays column-wise
    data = np.column_stack((P, T, Q, Angles, SurfType, SurfGeom, Skin, s2m, CO2, DateTimes, Result))

    # Save to a csv file
    np.savetxt("dataset_rttov.csv", data, delimiter=",", fmt="%d", header="P, T, Q, Angles, SurfType, SurfGeom, Skin, s2m, GasUnits, CO2, DateTimes, Result", comments="")




def create_dataset(P, T, Q, Angles, SurfType, SurfGeom, Skin, s2m, CO2, DateTimes, Result):
    # Función para convertir arrays en listas si tienen varias columnas
    def format_array(array):
        if array.ndim == 1:
            return array.reshape(-1, 1)  # Convertir 1D a 2D columna
        elif array.shape[1] > 1:
            return np.array([str(list(row)) for row in array])  # Convertir cada fila en una lista como string
        return array  # Ya tiene (nprofiles, 1), dejar igual

    # Convertir todas las variables correctamente
    P = format_array(P)
    T = format_array(T)
    Q = format_array(Q)
    Angles = format_array(Angles)
    SurfType = format_array(SurfType)
    SurfGeom = format_array(SurfGeom)
    Skin = format_array(Skin)
    s2m = format_array(s2m)
    CO2 = format_array(CO2)
    DateTimes = format_array(DateTimes)
    Result = format_array(Result)

    # Verificar que todas las columnas tengan el mismo número de filas
    nrows = P.shape[0]
    arrays = [P, T, Q, Angles, SurfType, SurfGeom, Skin, s2m, CO2, DateTimes, Result]

    for i, arr in enumerate(arrays):
        if arr.shape[0] != nrows:
            print(f"⚠️ Error: El array en la posición {i} tiene {arr.shape[0]} filas, pero se esperaban {nrows}.")
            return

    # Crear DataFrame
    df = pd.DataFrame(
        np.column_stack(arrays),  # Concatenar horizontalmente
        columns=["P", "T", "Q", "Angles", "SurfType", "SurfGeom", "Skin", "s2m", "CO2", "DateTimes", "Result"]
    )

    # Guardar en CSV con comillas para manejar listas correctamente
    df.to_csv("dataset_rttov.csv", index=False, sep=",", encoding="utf-8", quoting=1)

    print("✅ Dataset guardado como dataset_rttov.csv")




