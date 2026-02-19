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

### Opcional: fijar unas pocas direcciones en el mapa

Si quieres que **una dirección concreta** (p. ej. "Conde de los Gaitanes") salga en su sitio en el mapa en lugar de en la zona genérica, puedes usar **`data/direcciones_coordenadas.csv`** con columnas `texto`, `lat`, `lon`, `nombre`. Si la columna **Recoger** del CSV contiene ese `texto`, se usan esas coordenadas. Sirve para 5–10 direcciones que añadas a mano; no hace falta geocodificar miles de registros.

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

## Estandarización de datos

- **Fechas**: se aceptan `Fecha / Hora` o `Fecha/Hora` en formato DD/MM/YYYY con hora (HH:MM o HH MM). Otras columnas de fecha/hora se detectan por nombre.
- **Zona**: se normaliza a mayúsculas y sin espacios; solo se pintan en el mapa las filas cuya zona existe en `data/zonas_coordenadas.csv`.
- **Servicios cancelados**: si el CSV tiene columna **Fuente** (p. ej. "Completed" / "Cancelled") o columnas como "Información de Cancelación", "Razon de Cancelación" o "Cancelled By", la app marca los cancelados y **por defecto los excluye** del mapa y de los totales. En el panel lateral puedes desmarcar "Excluir servicios cancelados" para incluirlos.
- **Totales**: "Servicios en vista" y "Total servicios" son siempre el **número de viajes (filas)**, no la suma de pasajeros u otro peso.

## Notas

- El mapa se centra por defecto en Madrid; con tus datos se verán las zonas donde tengas lat/lon o las zonas definidas en `zonas_coordenadas.csv`.
- Si tu CSV usa otros nombres de columnas (en inglés, etc.), la app intenta detectarlos automáticamente.
- **CSV Ouigo / NORT**: sube el archivo; la app reconoce `Fecha / Hora`, `Zona`, `Fuente` (Completed/Cancelled), etc. Asegúrate de tener `data/zonas_coordenadas.csv` con todos los códigos de zona que uses (ALC, APC2, MOS, etc.).
