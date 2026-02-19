"""
Mapa de calor de servicios de taxi.
Filtros por día/mes y comparativa entre días.
Coordenadas en el mapa: por Zona; si hay columna Recoger, se geocodifican las direcciones (Nominatim/OSM) para posicionar cada punto.
"""
import re
import time
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pathlib import Path
from typing import Optional, List, Tuple

try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# Configuración de página
st.set_page_config(
    page_title="Mapa de calor - Servicios taxi",
    page_icon="🚕",
    layout="wide",
)

# Rutas
DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo_nort_taxi.png"
DEFAULT_CSV = DATA_DIR / "servicios_taxi_ejemplo.csv"

# Nombres de columnas aceptados (alias) — incluye formato reportes Ouigo/NORT
DATE_COLS = ["Fecha / Hora", "Fecha/Hora", "fecha", "date", "fecha_recogida", "pickup_date"]
DATETIME_COLS = ["Fecha / Hora", "Fecha/Hora", "Hora de Recogida", "Hora completada"]  # columna única fecha+hora
TIME_COLS = ["hora", "time", "hora_recogida", "pickup_time"]
LAT_COLS = ["lat_recogida", "lat", "latitude", "pickup_lat", "start_lat"]
LON_COLS = ["lon_recogida", "lon", "longitude", "pickup_lon", "start_lon"]
ZONE_COLS = ["Zona", "zona", "zone", "borough", "district"]
WEIGHT_COLS = ["Pasajeros", "Cantidad de cuenta", "cantidad", "count", "pasajeros", "weight"]
# Estado del servicio: Fuente = "Completed" | "Cancelled" (reportes NORT/Autocab)
STATUS_COLS = ["Fuente", "Estado", "Status", "State"]
# Columnas que indican cancelación cuando tienen valor
CANCEL_INFO_COLS = ["Información de Cancelación", "Razon de Cancelación", "Cancelled By", "Razón de Cancelación"]
ZONAS_FILE = DATA_DIR / "zonas_coordenadas.csv"

# Valores que consideramos "cancelado" en columna estado (case-insensitive)
CANCELLED_STATUS_VALUES = {"cancelled", "canceled", "cancelado", "cancelada", "cancelled by", "no show", "no-show"}
# Zonas con mapa de calor más fino (jitter para repartir el calor)
ZONAS_MAS_FINO = {"ALC", "ANB", "DIV", "URBA", "ALG", "SAN", "SSR", "LMO", "LTB", "SCH"}
# Columna cliente/cuenta para filtrar por cliente
CLIENTE_COLS = ["Cuenta", "Cliente", "Código de Cuenta", "Cliente de cuenta"]
# Columna ID de servicio (para búsqueda y detalle)
ID_COLS = ["ID", "Id", "Número", "Nº", "Nº Servicio", "Servicio", "Job", "No", "Nº de servicio"]
# Columna nombre (pasajero/cliente) para búsqueda
NOMBRE_COLS = ["Nombre", "nombre", "Name", "Cliente nombre", "Pasajero", "Pasajero nombre"]
# Columna de dirección de recogida (para clasificar Hoteles vs Particulares por la dirección)
RECOGIDA_COLS = ["Recoger", "Dirección", "Dirección de recogida", "Pickup", "Recoger en", "Origen"]
# Palabras que identifican recogida en hotel (la dirección de recogida contiene alguna)
HOTEL_KEYWORDS = (
    "hotel", "hotels", "hilton", "nh ", "nh,", "nh hoteles", "melia", "meliá", "barceló", "barcelo",
    "ibis", "ac hotels", "state", "resort", "hostal", "hostel", "albergue", "palace", "marriott",
    "radisson", "tryp", "silken", "eurostars", "congress", "hosteler", "alojamiento", "aparthotel", "suite hotel",
)


def _normalize_cliente(s: str) -> str:
    """Clave normalizada para agrupar nombres muy parecidos."""
    if pd.isna(s) or not isinstance(s, str):
        return ""
    return " ".join(s.strip().lower().split())


def _build_canonical_client_map(series: pd.Series) -> dict:
    """Mapeo clave_normalizada -> nombre canónico (el más frecuente del grupo)."""
    s = series.astype(str).str.strip().replace(["nan", "None"], "")
    s = s[s != ""]
    if s.empty:
        return {}
    norm_to_originals: dict = {}
    for v in s.unique():
        if not v:
            continue
        key = _normalize_cliente(v)
        if not key:
            continue
        if key not in norm_to_originals:
            norm_to_originals[key] = []
        norm_to_originals[key].append(v)
    vc = s.value_counts()
    norm_to_canonical = {}
    for key, originals in norm_to_originals.items():
        best = max(originals, key=lambda x: vc.get(x, 0))
        norm_to_canonical[key] = best
    return norm_to_canonical


def _fmt_fecha(s: str) -> str:
    """Convierte '2026-01-01' a '01/01/2026'."""
    if not s or pd.isna(s):
        return str(s)
    try:
        parts = str(s).strip().split("-")
        if len(parts) == 3:
            return f"{parts[2]}/{parts[1]}/{parts[0]}"
    except Exception:
        pass
    return str(s)


def _fmt_entero(x: float) -> int:
    """Para mostrar conteos como entero (evita 3464.0000000000005)."""
    return int(round(float(x)))


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Devuelve la primera columna que exista en el DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_datetime_ouigo(series: pd.Series) -> pd.Series:
    """Parsea fechas tipo '01/01/2026 00 40' (DD/MM/YYYY H MM) o '01/01/2026 00:40'."""
    def fix(s):
        if pd.isna(s) or not isinstance(s, str):
            return s
        s = s.strip()
        if ":" in s:
            return s
        # Partes separadas por espacios: fecha + hora + minuto
        parts = [p for p in s.split() if p]
        if len(parts) >= 3 and "/" in (parts[0] or ""):
            date_part = parts[0]
            h, m = parts[-2], parts[-1]
            if len(h) <= 2 and len(m) <= 2 and h.isdigit() and m.isdigit():
                s = date_part + " " + h.zfill(2) + ":" + m.zfill(2)
        return s
    fixed = series.astype(str).map(fix)
    return pd.to_datetime(fixed, format="%d/%m/%Y %H:%M", errors="coerce")


def load_zonas_mapping() -> Optional[pd.DataFrame]:
    """Carga zona -> lat, lon desde data/zonas_coordenadas.csv."""
    if not ZONAS_FILE.exists():
        return None
    z = pd.read_csv(ZONAS_FILE)
    col_z = "zona" if "zona" in z.columns else z.columns[0]
    z[col_z] = z[col_z].astype(str).str.strip().str.upper()
    return z


# Límite de direcciones a geocodificar por sesión (Nominatim pide ~1 req/s)
GEOCODE_MAX_PER_RUN = 400
GEOCODE_DELAY_SEC = 1.1


def _geocode_one(address: str) -> Optional[Tuple[float, float]]:
    """Geocodifica una dirección con Nominatim (OpenStreetMap). Devuelve (lat, lon) o None."""
    if not GEOPY_AVAILABLE or not address or not str(address).strip():
        return None
    try:
        geolocator = Nominatim(user_agent="HeatMapNT-taxi-map/1.0")
        location = geolocator.geocode(str(address).strip(), timeout=10, exactly_one=True)
        if location:
            return (location.latitude, location.longitude)
    except Exception:
        pass
    return None


def _ensure_geocode_cache(
    df_filt: pd.DataFrame,
    recogida_col: str,
) -> dict:
    """Rellena la caché de geocodificación para las direcciones únicas en df_filt. Usa st.session_state['geocode_cache']."""
    if "_geocode_cache" not in st.session_state:
        st.session_state["_geocode_cache"] = {}
    cache = st.session_state["_geocode_cache"]
    unique_recoger = df_filt[recogida_col].astype(str).str.strip().replace(["nan", ""], pd.NA).dropna().unique().tolist()
    unique_recoger = [u for u in unique_recoger if u and len(u) > 5]
    to_geocode = [a for a in unique_recoger if a not in cache][:GEOCODE_MAX_PER_RUN]
    if not to_geocode:
        return cache
    progress = st.progress(0.0, text="Geocodificando direcciones en el mapa...")
    n = len(to_geocode)
    for i, addr in enumerate(to_geocode):
        result = _geocode_one(addr)
        if result:
            cache[addr] = result
        progress.progress((i + 1) / n, text=f"Geocodificando direcciones... {i + 1}/{n}")
        time.sleep(GEOCODE_DELAY_SEC)
    progress.empty()
    return cache


def _apply_geocode_to_df(df_filt: pd.DataFrame, recogida_col: str, cache: dict) -> None:
    """Sobrescribe _lat, _lon en df_filt con las coordenadas geocodificadas cuando la dirección está en cache."""
    if "_lat" not in df_filt.columns or "_lon" not in df_filt.columns:
        return
    recoger_series = df_filt[recogida_col].astype(str).str.strip()
    for addr, (lat, lon) in cache.items():
        mask = (recoger_series == addr)
        if mask.any():
            df_filt.loc[mask, "_lat"] = lat
            df_filt.loc[mask, "_lon"] = lon


def _read_csv_auto_encoding(source, path: Optional[Path] = None) -> pd.DataFrame:
    """Lee CSV probando utf-8-sig, utf-8 y latin-1."""
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    last_error = None
    for enc in encodings:
        try:
            if path is not None:
                df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
            else:
                df = pd.read_csv(source, encoding=enc, on_bad_lines="skip", low_memory=False)
            return df
        except Exception as e:
            last_error = e
    raise last_error or RuntimeError("No se pudo leer el CSV")


def load_and_prepare_data(uploaded_file=None, path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Carga CSV y estandariza datos. Soporta formato Ouigo/NORT/Autocab.
    Estandarización:
    - Fechas: columna Fecha/Hora o Fecha / Hora, formato DD/MM/YYYY HH:MM o HH MM.
    - Zona: mayúsculas, sin espacios; sin zona en archivo → se descarta la fila.
    - Cancelados: Fuente='Cancelled' o columnas Información/Razon/Cancelled By con valor → _cancelado=True.
    - Servicios en vista = número de filas (viajes), no suma de Pasajeros.
    """
    try:
        if uploaded_file is not None:
            df = _read_csv_auto_encoding(uploaded_file)
        elif path and path.exists():
            df = _read_csv_auto_encoding(None, path=path)
        else:
            return None
    except Exception as e:
        st.error(f"Error al leer el CSV: **{type(e).__name__}** — {e}")
        return None

    # Normalizar nombres de columnas (quitar espacios y BOM)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    datetime_col = find_column(df, DATETIME_COLS)
    fecha_col = find_column(df, DATE_COLS)
    hora_col = find_column(df, TIME_COLS)
    lat_col = find_column(df, LAT_COLS)
    lon_col = find_column(df, LON_COLS)
    zone_col = find_column(df, ZONE_COLS)
    weight_col = find_column(df, WEIGHT_COLS)

    if not datetime_col and not fecha_col:
        st.error("El CSV debe tener una columna de fecha/hora (ej: Fecha/Hora, fecha, Hora de Recogida).")
        return None

    df = df.copy()

    # Fecha y hora
    if datetime_col:
        if datetime_col in ("Fecha/Hora", "Fecha / Hora"):
            df["_datetime"] = _parse_datetime_ouigo(df[datetime_col])
        else:
            df["_datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
        df["_fecha"] = df["_datetime"]
    else:
        df["_fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")
        if hora_col:
            try:
                df["_datetime"] = pd.to_datetime(
                    df[fecha_col].astype(str) + " " + df[hora_col].astype(str),
                    errors="coerce"
                )
            except Exception:
                df["_datetime"] = df["_fecha"]
        else:
            df["_datetime"] = df["_fecha"]

    # Coordenadas: directas o por zona
    if lat_col and lon_col:
        df["_lat"] = pd.to_numeric(df[lat_col].astype(str).str.replace(",", "."), errors="coerce")
        df["_lon"] = pd.to_numeric(df[lon_col].astype(str).str.replace(",", "."), errors="coerce")
        df["_zona"] = df[zone_col].astype(str) if zone_col else ""
        df = df.dropna(subset=["_fecha", "_lat", "_lon"])
    elif zone_col and ZONAS_FILE.exists():
        zonas_df = load_zonas_mapping()
        if zonas_df is None or zonas_df.empty:
            st.error("Hay columna Zona pero no se encontró data/zonas_coordenadas.csv con columnas zona, lat, lon.")
            return None
        col_z = "zona" if "zona" in zonas_df.columns else zonas_df.columns[0]
        df["_zona"] = df[zone_col].astype(str).str.strip().str.upper()
        zonas_sel = zonas_df[[col_z, "lat", "lon"]].copy()
        zonas_sel = zonas_sel.rename(columns={"lat": "_lat", "lon": "_lon", col_z: "_zona_join"})
        df = df.merge(zonas_sel, left_on="_zona", right_on="_zona_join", how="left")
        df["_lat"] = pd.to_numeric(df["_lat"], errors="coerce")
        df["_lon"] = pd.to_numeric(df["_lon"], errors="coerce")
        zonas_en_csv = df["_zona"].dropna().unique().tolist()
        df = df.dropna(subset=["_fecha", "_lat", "_lon"])
        df = df.drop(columns=["_zona_join"], errors="ignore")
        if df.empty:
            st.warning(
                f"No quedaron filas con fecha y zona válida. **Zonas en tu CSV:** {zonas_en_csv[:20]}. "
                "Añade esas zonas en `data/zonas_coordenadas.csv` (columnas: zona, lat, lon)."
            )
            return None
    else:
        st.error(
            "El CSV debe tener columnas de latitud y longitud, o columna Zona junto con el archivo "
            "data/zonas_coordenadas.csv (zona, lat, lon)."
        )
        return None

    # Peso: Pasajeros, Cantidad de cuenta, etc.
    if weight_col:
        w = pd.to_numeric(df[weight_col].astype(str).str.replace(",", "."), errors="coerce").fillna(1)
        df["_weight"] = w.clip(lower=0.1)
    else:
        df["_weight"] = 1.0

    # --- Estandarización: servicio cancelado o no ---
    status_col = find_column(df, STATUS_COLS)
    df["_cancelado"] = False
    if status_col:
        estado = df[status_col].astype(str).str.strip().str.lower()
        df.loc[estado.isin(CANCELLED_STATUS_VALUES), "_cancelado"] = True
    for col in CANCEL_INFO_COLS:
        if col in df.columns:
            tiene_info = df[col].astype(str).str.strip()
            mask = (tiene_info != "") & (~tiene_info.isin(["nan", "None"]))
            df.loc[mask, "_cancelado"] = True
    # Normalización extra: zona como texto limpio
    if "_zona" in df.columns:
        df["_zona"] = df["_zona"].astype(str).str.strip().str.upper().str.replace("NAN", "", regex=False)

    # --- ID de servicio (para detalle y búsqueda) ---
    id_col = find_column(df, ID_COLS)
    if id_col:
        df["_id"] = df[id_col].astype(str).str.strip()
    else:
        df["_id"] = df.index.astype(str)  # fallback: índice como identificador
    # --- Nombre (pasajero/cliente) para búsqueda ---
    nombre_col = find_column(df, NOMBRE_COLS)
    if nombre_col:
        df["_nombre"] = df[nombre_col].astype(str).str.strip().replace(["nan", "None"], "")
    else:
        df["_nombre"] = ""
    # --- Cliente / cuenta: estandarización ---
    cliente_col = find_column(df, CLIENTE_COLS)
    if cliente_col:
        df["_cliente_raw"] = df[cliente_col].astype(str).str.strip().replace(["nan", "None"], "")
        norm_to_canonical = _build_canonical_client_map(df["_cliente_raw"])
        df["_cliente_canonical"] = df["_cliente_raw"].apply(
            lambda x: norm_to_canonical.get(_normalize_cliente(x), x) if x else ""
        )
        # "Con cuenta" = filas donde la columna Cuenta viene informada
        df["_tiene_cuenta"] = (df["_cliente_raw"] != "") & (df["_cliente_raw"].notna())
    else:
        df["_cliente_raw"] = ""
        df["_cliente_canonical"] = ""
        df["_tiene_cuenta"] = False
    # --- Hoteles: por dirección de recogida (ej. Eurostars Congress, Hotel NH...) ---
    recogida_col = find_column(df, RECOGIDA_COLS)
    if recogida_col:
        texto_recogida = df[recogida_col].astype(str).str.lower()
        df["_es_hotel"] = texto_recogida.str.contains(
            "|".join(re.escape(k) for k in HOTEL_KEYWORDS), na=False, regex=True
        )
    else:
        df["_es_hotel"] = False

    return df


def build_heatmap_data(df: pd.DataFrame) -> list:
    """Convierte el DataFrame a lista [lat, lon, weight] para HeatMap. En ZONAS_MAS_FINO aplica jitter para calor más fino."""
    out = []
    for i, row in df.iterrows():
        lat, lon = float(row["_lat"]), float(row["_lon"])
        w = float(row["_weight"])
        zona = (row.get("_zona") or "").strip().upper()
        if zona in ZONAS_MAS_FINO:
            # Jitter determinista (± ~0.012°) para repartir el calor y que se vea bien cada zona (Alcobendas, Sanse, etc.)
            h = hash((i, "lat")) % 2000
            lat += (h - 1000) / 80000.0
            h = hash((i, "lon")) % 2000
            lon += (h - 1000) / 80000.0
        out.append([lat, lon, w])
    return out


def create_heatmap(df: pd.DataFrame, center_lat: float = 40.42, center_lon: float = -3.70, zoom: int = 12) -> folium.Map:
    """Crea un mapa Folium con capa de calor."""
    data = build_heatmap_data(df)
    if not data:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
        return m
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
    HeatMap(data, radius=25, blur=20, max_zoom=14).add_to(m)
    return m


def main():
    st.title("🚕 Mapa de calor - Servicios de taxi")
    st.caption("Carga un CSV de servicios y explora qué zonas tienen más y menos llamadas. Filtra por fechas y compara días.")

    # Carga de datos
    st.subheader("Cargar datos")
    uploaded = st.file_uploader("Sube tu CSV de servicios (ej. Ouigo)", type=["csv"])
    path_custom = st.text_input(
        "O escribe la ruta completa a un CSV en tu Mac",
        placeholder="/Users/tu_usuario/.../ouigo_enero.csv",
        help="Pega la ruta del archivo para no tener que subirlo cada vez."
    )
    use_default = st.checkbox("Si no subes ni indicas ruta: usar CSV de ejemplo", value=False)

    if uploaded:
        df = load_and_prepare_data(uploaded_file=uploaded)
    elif path_custom:
        path = Path(path_custom.strip()).expanduser()
        if path.exists():
            df = load_and_prepare_data(path=path)
            if df is not None:
                st.success(f"Cargado: **{path.name}** ({len(df):,} filas)")
        else:
            st.error(f"No existe el archivo: {path}")
            df = None
    elif use_default and DEFAULT_CSV.exists():
        df = load_and_prepare_data(path=DEFAULT_CSV)
    else:
        df = None

    # Diagnóstico si no cargó
    if df is None or df.empty:
        if path_custom and Path(path_custom.strip()).expanduser().exists():
            with st.expander("Diagnóstico: qué hay en tu CSV"):
                try:
                    path = Path(path_custom.strip()).expanduser()
                    raw = pd.read_csv(path, nrows=100, encoding="utf-8-sig", on_bad_lines="skip")
                    raw.columns = raw.columns.str.strip().str.replace("\ufeff", "")
                    st.write("**Columnas detectadas:**", list(raw.columns[:20]))
                    zcol = "Zona" if "Zona" in raw.columns else ("zona" if "zona" in raw.columns else None)
                    if zcol:
                        st.write("**Zonas en el CSV:**", sorted(raw[zcol].dropna().astype(str).unique().tolist())[:25])
                    fecha_col = next((c for c in ["Fecha/Hora", "Hora de Recogida", "fecha"] if c in raw.columns), None)
                    if fecha_col:
                        st.write("**Ejemplo de fecha:**", raw[fecha_col].iloc[0])
                except Exception as e:
                    st.write("No se pudo inspeccionar:", e)
        elif uploaded is not None:
            st.caption("Si no cargó: revisa que el CSV tenga columnas **Fecha/Hora** y **Zona**, y que cada zona exista en `data/zonas_coordenadas.csv`.")
        st.info(
            "Sube un CSV con columnas: **Fecha/Hora** (o fecha+hora), **Zona** (o lat/lon). "
            "Si solo tienes Zona, usa el archivo data/zonas_coordenadas.csv (zona, lat, lon) para el mapa."
        )
        return

    df["_mes"] = df["_datetime"].dt.month
    df["_año"] = df["_datetime"].dt.year
    df["_día_semana"] = df["_datetime"].dt.dayofweek  # 0=lunes, 6=domingo
    df["_día_semana_nombre"] = df["_datetime"].dt.day_name()
    df["_fecha_str"] = df["_datetime"].dt.date.astype(str)
    df["_hora"] = df["_datetime"].dt.hour  # 0-23 para filtro por tramo horario

    min_date = df["_fecha"].min().date()
    max_date = df["_fecha"].max().date()

    # --- Panel lateral: logo y filtros ---
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=120)
    st.sidebar.header("Filtros")
    rango = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(rango, (list, tuple)) and len(rango) == 2:
        fecha_min, fecha_max = rango[0], rango[1]
    else:
        fecha_min, fecha_max = min_date, max_date
    st.sidebar.caption(f"Rango: {fecha_min.strftime('%d/%m/%Y')} – {fecha_max.strftime('%d/%m/%Y')}")

    MESES_ES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    meses = st.sidebar.multiselect("Meses (vacío = todos)", options=list(range(1, 13)), format_func=lambda x: MESES_ES[x - 1], placeholder="Elegir meses")
    días_semana = st.sidebar.multiselect("Días de la semana", options=list(range(7)), format_func=lambda x: ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][x], placeholder="Elegir días")

    # Filtro por hora: tramos de 1 hora (00:00-01:00, 01:00-02:00, ...)
    HORAS_ETIQUETAS = [f"{h:02d}:00-{h+1:02d}:00" if h < 23 else "23:00-24:00" for h in range(24)]
    horas_sel = st.sidebar.multiselect(
        "Tramo horario (vacío = todas las horas)",
        options=list(range(24)),
        format_func=lambda h: HORAS_ETIQUETAS[h],
        placeholder="Elegir horas (ej. 00:00-01:00)",
    )

    # Filtro por estado: incluir todos por defecto; poder filtrar por completados/cancelados
    filtro_estado = "todos"  # todos | solo_completados | solo_cancelados
    if "_cancelado" in df.columns:
        n_cancelados_tot = int(df["_cancelado"].sum())
        n_completados_tot = len(df) - n_cancelados_tot
        st.sidebar.caption(f"En el CSV: **{_fmt_entero(n_completados_tot)}** completados, **{_fmt_entero(n_cancelados_tot)}** cancelados")
        filtro_estado = st.sidebar.radio(
            "Mostrar servicios",
            options=["todos", "solo_completados", "solo_cancelados"],
            format_func=lambda x: {"todos": "Todos (completados + cancelados)", "solo_completados": "Solo completados", "solo_cancelados": "Solo cancelados"}[x],
            index=0,
            horizontal=False,
        )

    # Filtro por cliente/cuenta: tipo (Con cuenta / Hoteles / Particulares) y búsqueda por Cuenta y Nombre
    tipo_cliente = "todos"
    clientes_sel: List[str] = []
    if "_cliente_canonical" in df.columns:
        st.sidebar.subheader("Cliente / cuenta")
        tipo_cliente = st.sidebar.radio(
            "Tipo de cliente",
            options=["todos", "cuenta", "hoteles", "particulares"],
            format_func=lambda x: {
                "todos": "Todos",
                "cuenta": "Con cuenta (cuenta informada)",
                "hoteles": "Hoteles",
                "particulares": "Particulares",
            }[x],
            index=0,
            horizontal=False,
        )
        with st.sidebar.expander("¿Cómo se clasifica?"):
            st.caption(
                "**Con cuenta:** servicios con la columna **Cuenta** informada. "
                "**Hoteles:** dirección de recogida (Recoger) contiene hotel, eurostars, congress, hilton, nh, meliá, etc. "
                "**Particulares:** el resto (domicilio, empresa, etc.)."
            )
        unique_clientes = sorted(
            df["_cliente_canonical"].dropna().astype(str).replace("", pd.NA).dropna().unique().tolist()
        )
        unique_clientes = [c for c in unique_clientes if c and str(c).strip()]
        busqueda_cliente = st.sidebar.text_input(
            "Cliente (busca por Cuenta o Nombre)",
            placeholder="Escribe para ver coincidencias...",
            key="busca_cliente",
        )
        busq = (busqueda_cliente or "").strip().lower()
        if busq:
            # Coincidencias por Cuenta (canonical) o por Nombre
            if "_nombre" in df.columns:
                mask_cuenta = df["_cliente_canonical"].astype(str).str.lower().str.contains(re.escape(busq), na=False)
                mask_nombre = df["_nombre"].astype(str).str.lower().str.contains(re.escape(busq), na=False)
                clientes_con_busq = set(df.loc[mask_cuenta | mask_nombre, "_cliente_canonical"].dropna().astype(str))
                coincidencias = sorted([c for c in unique_clientes if c in clientes_con_busq])[:150]
            else:
                coincidencias = [c for c in unique_clientes if busq in c.lower()][:150]
        else:
            coincidencias = unique_clientes[:150]
        clientes_sel = st.sidebar.multiselect(
            "Coincidencias (elige uno o más)",
            options=coincidencias,
            default=[],
            placeholder="Abre el desplegable y elige" if coincidencias else "Escribe arriba para ver coincidencias",
            key="clientes_multiselect",
        )
        if not busq and len(unique_clientes) > 150:
            st.sidebar.caption("Escribe en el campo Cliente para acotar la lista.")

    df_filt = df[
        (df["_fecha"].dt.date >= fecha_min) &
        (df["_fecha"].dt.date <= fecha_max)
    ]
    if meses:
        df_filt = df_filt[df_filt["_mes"].isin(meses)]
    if días_semana:
        df_filt = df_filt[df_filt["_día_semana"].isin(días_semana)]
    if horas_sel:
        df_filt = df_filt[df_filt["_hora"].isin(horas_sel)]
    if "_cancelado" in df_filt.columns:
        if filtro_estado == "solo_completados":
            df_filt = df_filt[~df_filt["_cancelado"]]
        elif filtro_estado == "solo_cancelados":
            df_filt = df_filt[df_filt["_cancelado"]]
    if "_tiene_cuenta" in df_filt.columns and tipo_cliente == "cuenta":
        df_filt = df_filt[df_filt["_tiene_cuenta"]]
    if "_es_hotel" in df_filt.columns:
        if tipo_cliente == "hoteles":
            df_filt = df_filt[df_filt["_es_hotel"]]
        elif tipo_cliente == "particulares":
            df_filt = df_filt[~df_filt["_es_hotel"]]
    if "_cliente_canonical" in df_filt.columns and clientes_sel:
        df_filt = df_filt[df_filt["_cliente_canonical"].isin(clientes_sel)]

    # Coordenadas dinámicas por dirección: geocodificar Recoger (Nominatim/OSM) para el mapa; la clasificación sigue por Zona
    recogida_col = find_column(df, RECOGIDA_COLS)
    if (
        recogida_col
        and recogida_col in df_filt.columns
        and GEOPY_AVAILABLE
        and not df_filt.empty
        and "_lat" in df_filt.columns
    ):
        cache = _ensure_geocode_cache(df_filt, recogida_col)
        df_filt = df_filt.copy()
        _apply_geocode_to_df(df_filt, recogida_col, cache)
    elif recogida_col and not GEOPY_AVAILABLE:
        st.sidebar.caption("Instala **geopy** para que el mapa use coordenadas por dirección: `pip install geopy`")

    # Resumen: servicios y desglose completados/cancelados
    n_vista = len(df_filt)
    st.sidebar.metric("Servicios en vista", _fmt_entero(n_vista))
    if "_cancelado" in df_filt.columns and n_vista > 0:
        n_compl = int((~df_filt["_cancelado"]).sum())
        n_canc = int(df_filt["_cancelado"].sum())
        st.sidebar.caption(f"↳ {_fmt_entero(n_compl)} completados, {_fmt_entero(n_canc)} cancelados")
    st.sidebar.metric("Puntos en mapa", len(df_filt))

    # Tabs: Mapa | Comparativa | Detalle servicio
    tab_mapa, tab_comparativa, tab_detalle = st.tabs(["Mapa de calor", "Comparativa entre días", "Detalle de servicio"])

    with tab_mapa:
        if df_filt.empty:
            st.warning("No hay datos con los filtros seleccionados.")
        else:
            if "_cancelado" in df_filt.columns:
                nc, nk = int((~df_filt["_cancelado"]).sum()), int(df_filt["_cancelado"].sum())
                st.caption(f"Servicios en el mapa: **{_fmt_entero(nc)}** completados, **{_fmt_entero(nk)}** cancelados")
            m = create_heatmap(df_filt)
            st_folium(m, use_container_width=True, height=500)
            # Resumen por zona si existe
            zone_col = find_column(df, ZONE_COLS)
            if zone_col and "_zona" in df_filt.columns and df_filt["_zona"].astype(str).str.len().gt(0).any():
                st.subheader("Llamadas por zona (filtrado)")
                por_zona = df_filt.groupby("_zona", dropna=False).size().sort_values(ascending=False)
                st.bar_chart(por_zona)
                if "_cancelado" in df_filt.columns:
                    st.caption("Desglose por zona (completados / cancelados):")
                    por_zona_estado = df_filt.groupby(["_zona", "_cancelado"], dropna=False).size().unstack(fill_value=0)
                    por_zona_estado = por_zona_estado.rename(columns={False: "Completados", True: "Cancelados"}, errors="ignore")
                    por_zona_estado = por_zona_estado.sort_values(por_zona_estado.columns[0], ascending=False)
                    st.dataframe(por_zona_estado.astype(int), use_container_width=True)

    with tab_comparativa:
        fechas_únicas = sorted(df_filt["_fecha_str"].unique())
        if len(fechas_únicas) < 2:
            st.info("Se necesitan al menos dos fechas distintas (con los filtros actuales) para comparar.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                día_a = st.selectbox("Día A", fechas_únicas, format_func=_fmt_fecha, key="dia_a")
            with col_b:
                día_b = st.selectbox("Día B", fechas_únicas, format_func=_fmt_fecha, key="dia_b")

            df_a = df_filt[df_filt["_fecha_str"] == día_a]
            df_b = df_filt[df_filt["_fecha_str"] == día_b]

            st.subheader(f"Comparativa: {_fmt_fecha(día_a)} vs {_fmt_fecha(día_b)}")
            c1, c2, c3 = st.columns(3)
            n_a, n_b = len(df_a), len(df_b)
            c1.metric("Día A - Total servicios", _fmt_entero(n_a))
            c2.metric("Día B - Total servicios", _fmt_entero(n_b))
            diff = n_b - n_a
            c3.metric("Diferencia (B - A)", _fmt_entero(diff), delta=f"{_fmt_entero(diff)}")
            if "_cancelado" in df_a.columns:
                compl_a = int((~df_a["_cancelado"]).sum())
                canc_a = int(df_a["_cancelado"].sum())
                compl_b = int((~df_b["_cancelado"]).sum())
                canc_b = int(df_b["_cancelado"].sum())
                st.caption(f"**Día A:** {_fmt_entero(compl_a)} completados, {_fmt_entero(canc_a)} cancelados  —  **Día B:** {_fmt_entero(compl_b)} completados, {_fmt_entero(canc_b)} cancelados")

            map_col_a, map_col_b = st.columns(2)
            with map_col_a:
                st.caption(f"Mapa día A: {_fmt_fecha(día_a)}")
                st_folium(create_heatmap(df_a), use_container_width=True, height=400, key="comp_a")
            with map_col_b:
                st.caption(f"Mapa día B: {_fmt_fecha(día_b)}")
                st_folium(create_heatmap(df_b), use_container_width=True, height=400, key="comp_b")

            zone_col = find_column(df, ZONE_COLS)
            if zone_col and "_zona" in df_filt.columns and df_filt["_zona"].astype(str).str.len().gt(0).any():
                st.subheader("Comparativa por zona")
                za = df_a.groupby("_zona", dropna=False).size()
                zb = df_b.groupby("_zona", dropna=False).size()
                comp = pd.DataFrame({"Día A": za, "Día B": zb}).fillna(0)
                comp["Diferencia"] = comp["Día B"] - comp["Día A"]
                comp_show = comp.fillna(0).round(0).astype(int)
                st.dataframe(comp_show.sort_values("Día B", ascending=False), use_container_width=True)

    with tab_detalle:
        st.subheader("Buscar y ver detalle de un servicio")
        busca_id = st.text_input(
            "Buscar por ID (o parte del ID)",
            placeholder="Escribe el ID del servicio o parte de él...",
            key="busca_id_servicio",
        )
        if busca_id and busca_id.strip():
            id_busq = busca_id.strip()
            # Buscar en datos cargados (independiente de filtros del mapa)
            mask = df["_id"].astype(str).str.contains(re.escape(id_busq), case=False, na=False)
            detalle_df = df.loc[mask]
            if detalle_df.empty:
                st.info("No se encontró ningún servicio con ese ID.")
            else:
                n = len(detalle_df)
                st.caption(f"Se encontraron **{n}** servicio(s).")
                # Mostrar columnas originales del CSV + _lat, _lon, _zona, _fecha_str si existen
                cols_orig = [c for c in df.columns if not c.startswith("_")]
                cols_extra = [c for c in ["_id", "_fecha_str", "_zona", "_lat", "_lon", "_cliente_canonical", "_nombre", "_cancelado"] if c in detalle_df.columns]
                cols_show = ["_id"] + cols_orig + [c for c in cols_extra if c not in cols_orig and c != "_id"]
                cols_show = [c for c in cols_show if c in detalle_df.columns]
                st.dataframe(detalle_df[cols_show], use_container_width=True)
        else:
            st.caption("Escribe un ID (o parte) para ver el detalle del servicio. Se buscan coincidencias en la columna ID del CSV.")


if __name__ == "__main__":
    main()
