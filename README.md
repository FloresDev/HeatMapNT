# Mapa de calor - Servicios de taxi

Aplicación web para visualizar **qué zonas tienen más y menos llamadas** de taxi a partir de un CSV, con **filtros por fechas, meses y días de la semana** y **comparativa entre dos días**.

## Qué hace

- **Mapa de calor**: muestra la densidad de servicios (más calor = más llamadas).
- **Filtros**: rango de fechas, meses, días de la semana.
- **Comparativa**: elige dos días y compara totales y mapa lado a lado; si hay columna de zona, tabla comparativa por zona.

## Formato del CSV

Compatible con **reportes tipo Ouigo/NORT** y CSVs con coordenadas.

### Opción A – Con coordenadas (lat/lon)

| Columna   | Ejemplo de nombre   | Descripción        |
|-----------|---------------------|--------------------|
| Fecha     | `fecha`, `date`     | Fecha del servicio |
| Latitud   | `lat_recogida`, `lat` | Coordenada recogida |
| Longitud  | `lon_recogida`, `lon` | Coordenada recogida |
| Zona      | opcional            | Para resúmenes     |
| Peso      | `cantidad`, `Pasajeros` | Opcional       |

### Opción B – Solo zona (estilo Ouigo)

Si el CSV **no tiene lat/lon** pero sí **Zona** (código de zona), el mapa usa el archivo de coordenadas por zona:

| Columna   | Nombre en el CSV   | Descripción |
|-----------|--------------------|-------------|
| Fecha/hora| `Fecha/Hora` o `Hora de Recogida` | Formato DD/MM/YYYY HH MM o DD/MM/YYYY HH:MM |
| Zona      | `Zona`             | Código (ej. ALC, APC2, MOS) |
| Peso      | `Pasajeros`, `Cantidad de cuenta` | Opcional |

En este caso hace falta el archivo **`data/zonas_coordenadas.csv`** con columnas `zona`, `lat`, `lon` (y opcional `nombre`). Ahí están ya ALC, APC2, MOS y otras zonas del área Madrid; puedes añadir filas para nuevos códigos de zona.

## Dónde ejecutarlo

### 1. En tu ordenador (recomendado para empezar)

```bash
cd /Volumes/DeveloperSSDD/IA/HeatMapNT
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`. Ahí puedes subir tu CSV o usar el de ejemplo.

### 2. En la nube (para compartir o usar desde cualquier sitio)

- **Streamlit Community Cloud** (gratis): sube el proyecto a GitHub y conecta el repo en [share.streamlit.io](https://share.streamlit.io). Ellos ejecutan la app y te dan una URL.
- **Otro hosting**: cualquier servidor donde puedas instalar Python y ejecutar `streamlit run app.py` (por ejemplo un VPS, Railway, Render, etc.).

## Estructura del proyecto

```
HeatMapNT/
├── app.py              # Aplicación Streamlit
├── requirements.txt
├── README.md
└── data/
    └── servicios_taxi_ejemplo.csv
```

## Notas

- El mapa se centra por defecto en Madrid; con tus datos se verán las zonas donde tengas lat/lon o las zonas definidas en `zonas_coordenadas.csv`.
- Si tu CSV usa otros nombres de columnas (en inglés, etc.), la app intenta detectarlos automáticamente.
- **CSV Ouigo**: sube el archivo (ej. `ouigo_enero.csv`); la app reconoce `Fecha/Hora`, `Zona`, `Pasajeros`, etc. Asegúrate de tener `data/zonas_coordenadas.csv` con todos los códigos de zona que uses (ALC, APC2, MOS, etc.).
