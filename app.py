"""
Mapa de calor de servicios de taxi.
Filtros por día/mes y comparativa entre días.
"""
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pathlib import Path
from typing import Optional, List

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
ZONAS_FILE = DATA_DIR / "zonas_coordenadas.csv"


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
    """Carga CSV y normaliza columnas para fecha, lat, lon (o zona) y peso. Soporta formato Ouigo/NORT."""
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

    return df


def build_heatmap_data(df: pd.DataFrame) -> list:
    """Convierte el DataFrame a lista [lat, lon, weight] para HeatMap."""
    return df[["_lat", "_lon", "_weight"]].values.tolist()


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

    meses = st.sidebar.multiselect("Meses (vacío = todos)", options=list(range(1, 13)), format_func=lambda x: pd.Timestamp(2000, x, 1).strftime("%B"))
    días_semana = st.sidebar.multiselect("Días de la semana", options=list(range(7)), format_func=lambda x: ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][x])

    df_filt = df[
        (df["_fecha"].dt.date >= fecha_min) &
        (df["_fecha"].dt.date <= fecha_max)
    ]
    if meses:
        df_filt = df_filt[df_filt["_mes"].isin(meses)]
    if días_semana:
        df_filt = df_filt[df_filt["_día_semana"].isin(días_semana)]

    # Resumen (números enteros)
    st.sidebar.metric("Servicios en vista", _fmt_entero(df_filt["_weight"].sum()))
    st.sidebar.metric("Puntos en mapa", len(df_filt))

    # Tabs: Mapa | Comparativa
    tab_mapa, tab_comparativa = st.tabs(["Mapa de calor", "Comparativa entre días"])

    with tab_mapa:
        if df_filt.empty:
            st.warning("No hay datos con los filtros seleccionados.")
        else:
            m = create_heatmap(df_filt)
            st_folium(m, use_container_width=True, height=500)
            # Resumen por zona si existe
            zone_col = find_column(df, ZONE_COLS)
            if zone_col and "_zona" in df_filt.columns and df_filt["_zona"].astype(str).str.len().gt(0).any():
                st.subheader("Llamadas por zona (filtrado)")
                por_zona = df_filt.groupby("_zona", dropna=False)["_weight"].sum().sort_values(ascending=False)
                st.bar_chart(por_zona)

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
            c1.metric("Día A - Total servicios", _fmt_entero(df_a["_weight"].sum()))
            c2.metric("Día B - Total servicios", _fmt_entero(df_b["_weight"].sum()))
            diff = df_b["_weight"].sum() - df_a["_weight"].sum()
            c3.metric("Diferencia (B - A)", _fmt_entero(diff), delta=f"{_fmt_entero(diff)}")

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
                za = df_a.groupby("_zona", dropna=False)["_weight"].sum()
                zb = df_b.groupby("_zona", dropna=False)["_weight"].sum()
                comp = pd.DataFrame({"Día A": za, "Día B": zb}).fillna(0)
                comp["Diferencia"] = comp["Día B"] - comp["Día A"]
                comp_show = comp.fillna(0).round(0).astype(int)
                st.dataframe(comp_show.sort_values("Día B", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
