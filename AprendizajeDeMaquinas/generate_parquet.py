import pandas as pd


def generate_parquet():
    # Cargar los DataFrames desde archivos CSV
    df = pd.read_csv("recorridos.csv")
    df_weather = pd.read_csv("weather.csv")

    # Crear columna 'fecha' como datetime.date en ambos DataFrames
    df["fecha"] = pd.to_datetime(df["fecha_origen_recorrido"]).dt.date
    df_weather["fecha"] = pd.to_datetime(df_weather["date"]).dt.date

    # Agrupar por fecha y contar recorridos
    df_resumen = df.groupby("fecha").size().reset_index(name="cantidad_recorridos")

    # Calcular día de la semana (lunes=0, domingo=6) y ajustar a 1-7
    df_resumen["day_of_week"] = pd.to_datetime(df_resumen["fecha"]).dt.weekday + 1

    # Es fin de semana si day_of_week es 6 (sábado) o 7 (domingo)
    df_resumen["is_weekend"] = df_resumen["day_of_week"].isin([6, 7])

    df_resumen["day_of_month"] = pd.to_datetime(df_resumen["fecha"]).dt.day
    df_resumen["month"] = pd.to_datetime(df_resumen["fecha"]).dt.month

    # Unir con el DataFrame de clima por fecha
    df_final = pd.merge(df_resumen, df_weather[["fecha", "tavg", "prcp", "wspd"]], on="fecha", how="left")

    print(df_final.head())
    print(df_final.describe())
    print(df_final.info())

    # Guardar el DataFrame final en formato Parquet
    df_final.to_parquet("final.parquet", index=False)
    print(df_final.head())


if __name__ == "__main__":
    generate_parquet()
