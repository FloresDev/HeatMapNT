"""
Script offline para geocodificar direcciones de recogida y rellenar
`data/direcciones_coordenadas.csv` con:

    texto,lat,lon,nombre

La app de Streamlit usará este archivo para colocar ciertos puntos
con coordenadas reales, sin geocodificar nada en tiempo de ejecución.

Uso recomendado (desde la raíz del proyecto):

    python scripts/geocode_direcciones.py \\
        --csv "/Users/tu_usuario/Downloads/enero_NT_2026.csv" \\
        --columna-recoger "Recoger"

Puedes pasar varios `--csv` si quieres combinar meses distintos.
"""

import argparse
import csv
import difflib
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Set, Optional, Tuple

import pandas as pd
from geopy.geocoders import Nominatim


PROYECTO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROYECTO_DIR / "data"
DIRECCIONES_FILE = DATA_DIR / "direcciones_coordenadas.csv"

# Nominatim: 1 petición/segundo para respetar el servicio público
GEOCODE_DELAY_SEC = 1.1
USER_AGENT = "HeatMapNT-offline-geocode/1.0"

# Bounding box España (lon_sw, lat_sw, lon_ne, lat_ne) para priorizar resultados en España
VIEWBOX_ES = (-9.5, 35.5, 4.5, 43.8)


def _alternativas_busqueda(addr: str) -> List[str]:
    """Genera alternativas de búsqueda cuando la dirección completa no da resultado.
    Incluye uso del nombre de hotel/empresa (p. ej. "Hotel AC San Sebastián de los Reyes").
    """
    addr = addr.strip()
    if not addr:
        return []
    out = [addr]

    # Extraer posible nombre de hotel/empresa al inicio (antes de la primera coma)
    # Ej.: "HOTEL AC. Av. Cerro...", "OUIGO ESPAÑA, Calle...", "Hotel Catalonia Roma, Avinguda..."
    nombre_establecimiento: Optional[str] = None
    if "," in addr:
        primer_bloque = addr.split(",", 1)[0].strip()
        # Quitar paréntesis y posibles trozos de vía: nos quedamos con el "nombre puro"
        primer_bloque = re.sub(r"\(.*?\)", "", primer_bloque).strip()
        primer_bloque = re.split(
            r"\b(Calle|C\/|C\.|Avda\.?|Av\.?|Avenida|Plaza|Paseo|Ronda|P\.º|Pº)\b",
            primer_bloque,
            maxsplit=1,
        )[0].strip()
        # Que parezca un nombre: no es solo número, tiene letras, no es solo "Calle de X"
        if primer_bloque and re.search(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]{2,}", primer_bloque):
            if re.search(r"(?i)hotel|ouigo|hostal|hostel|ac\.?|catalonia|nh\s|meliá|iberostar|barceló|radisson|marriott|hilton|ibis|novotel", primer_bloque):
                nombre_establecimiento = primer_bloque
            elif not re.match(r"^(Calle|C\/|Av\.?|Avenida|Plaza|Paseo|Ronda)\s", primer_bloque, re.IGNORECASE):
                # Cualquier otro nombre al inicio (empresa, lugar) que no sea tipo de vía
                if len(primer_bloque) > 3:
                    nombre_establecimiento = primer_bloque

    # Quitar prefijos tipo "OUIGO ESPAÑA, ", "HOTEL AC. ", "Hotel X, " y probar el resto
    sin_prefijo = re.sub(
        r"^(OUIGO\s+ESPAÑA|HOTEL\s+AC\.?|HOTEL\s+[^,]+),\s*",
        "",
        addr,
        flags=re.IGNORECASE,
    ).strip()
    if sin_prefijo and sin_prefijo not in out:
        out.append(sin_prefijo)

    # Patrón "NOMBRE (Empresa) DIRECCIÓN, Ciudad" o "NOMBRE (Hotel) DIRECCIÓN" -> la calle va después del ")"
    emp_match = re.match(r"^[^,(]+\(\s*(?:Empresa|Hotel)\s*\)\s*(.+)", addr, re.IGNORECASE)
    if emp_match:
        despues = emp_match.group(1).strip()
        if despues and despues not in out:
            out.append(despues)
        # Normalizar AVENIDA DE X / Av. X y Nº para OSM
        norm = re.sub(r"\bAVENIDA\s+DE\s+", "Avenida de ", despues, flags=re.IGNORECASE)
        norm = re.sub(r"\bAV\.?\s+", "Avenida ", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\s+Nº\s*", " ", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\s{2,}", " ", norm)
        if norm != despues and norm not in out:
            out.append(norm)

    # Dirección dentro de paréntesis tipo "(C/ ARAGONESES 18)" -> intentar solo esa parte como calle
    par_match = re.search(r"\(([^)]*?)\)", addr)
    if par_match:
        dentro = par_match.group(1).strip()
        # No usar paréntesis que solo digan "Empresa" o "Hotel" (la dirección va fuera)
        if dentro and dentro.lower() not in ("empresa", "hotel"):
            # Normalizar C/ y Avda./Av. dentro de los paréntesis
            frag = dentro
            frag = re.sub(r"^(C\/|C\.)(\s*)", r"Calle ", frag, flags=re.IGNORECASE)
            frag = re.sub(r"^(Avda\.?|Av\.?)(\s*)", r"Avenida ", frag, flags=re.IGNORECASE)
            # Añadir el resto de la dirección fuera de paréntesis, si lo hay
            resto = addr[par_match.end():].strip(" ,")
            if resto:
                candidata_par = f"{frag}, {resto}"
            else:
                candidata_par = frag
            if candidata_par and candidata_par not in out:
                out.append(candidata_par)

    # Si hay nombre de hotel/empresa, buscar "Nombre, Ciudad, España"
    if nombre_establecimiento:
        for ciudad in ("Madrid", "Barcelona", "Alcobendas", "San Sebastián de los Reyes", "Collado Villalba", "Málaga", "Valencia", "Barajas"):
            if ciudad.lower() in addr.lower():
                candidato = f"{nombre_establecimiento}, {ciudad}, España"
                if candidato not in out:
                    out.append(candidato)
                break
        # También intentar solo "Nombre, España" por si Nominatim lo encuentra
        solo_nombre = f"{nombre_establecimiento}, España"
        if solo_nombre not in out:
            out.append(solo_nombre)
    # Sin números de portal (ej. "15-17" o ", 15") para que coincida la calle en OSM
    sin_numero = re.sub(r",\s*\d+[\s\-–—]*\d*\s*$", "", addr)
    sin_numero = re.sub(r"\s+\d+[\s\-–—]*\d*\s*$", "", sin_numero)
    if sin_numero and sin_numero != addr and sin_numero not in out:
        out.append(sin_numero)
    # Normalizar Av. / Avenida y Aguila -> Águila por si OSM tiene la variante
    variante = addr.replace("Av. ", "Avenida ").replace("Av ", "Avenida ")
    if "aguila" in variante.lower() and "águila" not in variante.lower():
        variante = re.sub(r"aguila", "Águila", variante, flags=re.IGNORECASE)
    if variante != addr and variante not in out:
        out.append(variante)
    # Si hay código postal tipo 28702, 28108, 28042... extraer "Ciudad, España"
    cp_match = re.search(r"\b(28\d{3}|08\d{3})\b", addr)
    if cp_match:
        # Buscar nombre de ciudad antes del CP (ej. "San Sebastián de los Reyes, 28702")
        antes_cp = addr[: cp_match.start()]
        partes = [p.strip() for p in antes_cp.split(",") if p.strip()]
        ciudad = None
        for parte in reversed(partes):
            if re.search(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]{2,}", parte):
                ciudad = parte
                break
        if ciudad and f"{ciudad}, España" not in out:
            out.append(f"{ciudad}, España")
    # Si la dirección contiene "Madrid" o "Barcelona" etc., intentar solo "Calle, Ciudad, España"
    for ciudad in ("Madrid", "Barcelona", "Alcobendas", "San Sebastián de los Reyes", "Málaga", "Valencia"):
        if ciudad.lower() in addr.lower() and f"{ciudad}, España" not in out:
            out.append(f"{ciudad}, España")
    return out


def leer_direcciones_csv(rutas_csv: List[Path], columna_recoger: str) -> Counter:
    contador = Counter()
    for ruta in rutas_csv:
        print(f"⏳ Leyendo direcciones de {ruta} ...")
        try:
            df = pd.read_csv(ruta, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
        except Exception as e:
            print(f"  ⚠️  Error leyendo {ruta}: {e}")
            continue
        if columna_recoger not in df.columns:
            print(f"  ⚠️  {ruta} no tiene columna '{columna_recoger}'. Columnas detectadas: {list(df.columns)[:10]}")
            continue
        serie = df[columna_recoger].astype(str).str.strip()
        for addr in serie:
            if addr and addr.lower() not in ("nan", "none"):
                contador[addr] += 1
        print(f"  ✔ {ruta}: {len(serie.dropna())} filas con dirección, {len(contador)} direcciones únicas acumuladas.")
    return contador


def leer_existentes() -> Set[str]:
    existentes: Set[str] = set()
    if DIRECCIONES_FILE.exists():
        print(f"📄 Leyendo direcciones ya geocodificadas de {DIRECCIONES_FILE} ...")
        with DIRECCIONES_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texto = (row.get("texto") or "").strip()
                if texto:
                    existentes.add(texto)
        print(f"  ✔ {len(existentes)} direcciones ya presentes.")
    return existentes


def _buscar_similar(addr: str, filas_existentes: List[dict], umbral: float = 0.9) -> Optional[dict]:
    """Devuelve una fila existente cuya columna 'texto' se parezca a `addr` con ratio >= umbral."""
    addr_norm = addr.strip().lower()
    if not addr_norm or not filas_existentes:
        return None
    mejor_fila: Optional[dict] = None
    mejor_score = umbral
    for fila in filas_existentes:
        texto = (fila.get("texto") or "").strip()
        if not texto:
            continue
        score = difflib.SequenceMatcher(None, addr_norm, texto.lower()).ratio()
        if score > mejor_score:
            mejor_score = score
            mejor_fila = fila
    return mejor_fila


def _geocode_una(geolocator: Nominatim, query: str) -> Optional[Tuple[float, float]]:
    """Intenta geocodificar una sola cadena; devuelve (lat, lon) o None.

    Nota: mantenemos la llamada lo más simple posible porque con
    `country_codes`/`viewbox` algunas direcciones estaban devolviendo
    sistemáticamente "Sin resultados" aunque Nominatim sí las resuelve.
    """
    try:
        loc = geolocator.geocode(
            query,
            timeout=15,
            exactly_one=True,
            # Si en el futuro hace falta, se puede volver a añadir:
            # viewbox=VIEWBOX_ES,
            # bounded=False,
            # country_codes="es",
        )
    except Exception as e:
        print(f"  ⚠️  Error geocodificando '{query}': {e}")
        return None
    if loc is None:
        return None
    lat, lon = loc.latitude, loc.longitude
    # Descartar resultados claramente fuera de España (ej. "1, España" que devuelve Argentina)
    lon_sw, lat_sw, lon_ne, lat_ne = VIEWBOX_ES
    lon_min, lon_max = sorted((lon_sw, lon_ne))
    lat_min, lat_max = sorted((lat_sw, lat_ne))
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        print(f"  ⚠️  Resultado fuera de España ({lat:.6f}, {lon:.6f}), descartando.")
        return None
    return (lat, lon)


def geocode_direcciones(unicas_ordenadas: List[str], ya_hechas: Set[str]) -> List[dict]:
    geolocator = Nominatim(user_agent=USER_AGENT)
    filas: List[dict] = []
    total = len(unicas_ordenadas)

    # Cargamos también las filas existentes completas para poder hacer matching aproximado (>=90%)
    existentes_detalle: List[dict] = []
    if DIRECCIONES_FILE.exists():
        with DIRECCIONES_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("texto") or "").strip():
                    existentes_detalle.append(row)

    for i, addr in enumerate(unicas_ordenadas, start=1):
        if addr in ya_hechas:
            print(f"[{i}/{total}] (saltando, ya existente) {addr}")
            continue

        # Intentar reutilizar coordenadas de una dirección muy parecida ya geocodificada
        fila_similar = _buscar_similar(addr, existentes_detalle, umbral=0.9)
        if fila_similar:
            try:
                lat = float(fila_similar.get("lat"))
                lon = float(fila_similar.get("lon"))
            except (TypeError, ValueError):
                lat = lon = None
            if lat is not None and lon is not None:
                filas.append(
                    {
                        "texto": addr,
                        "lat": lat,
                        "lon": lon,
                        "nombre": addr,
                    }
                )
                existentes_detalle.append(
                    {
                        "texto": addr,
                        "lat": lat,
                        "lon": lon,
                        "nombre": addr,
                    }
                )
                print(
                    f"[{i}/{total}] (reutilizando ~{fila_similar.get('texto', '').strip()!r}) "
                    f"{lat:.6f}, {lon:.6f}"
                )
                continue

        print(f"[{i}/{total}] Geocodificando: {addr}")
        coords = _geocode_una(geolocator, addr)
        usado_fallback = False
        if coords is None:
            for alternativa in _alternativas_busqueda(addr):
                if alternativa == addr:
                    continue
                time.sleep(GEOCODE_DELAY_SEC)
                coords = _geocode_una(geolocator, alternativa)
                if coords is not None:
                    usado_fallback = True
                    print(f"  ✔ (fallback: «{alternativa[:50]}…») {coords[0]:.6f}, {coords[1]:.6f}")
                    break
        if coords is None:
            print("  ⚠️  Sin resultados.")
        else:
            filas.append(
                {
                    "texto": addr,
                    "lat": coords[0],
                    "lon": coords[1],
                    "nombre": addr,
                }
            )
            if not usado_fallback:
                print(f"  ✔ {coords[0]:.6f}, {coords[1]:.6f}")
        time.sleep(GEOCODE_DELAY_SEC)
    return filas


def guardar_direcciones(nuevas: List[dict]) -> None:
    # Leemos lo que ya haya para no perder entradas anteriores
    existentes: List[dict] = []
    if DIRECCIONES_FILE.exists():
        with DIRECCIONES_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existentes.append(row)

    todas = existentes + nuevas
    if not todas:
        print("⚠️  No hay direcciones que guardar.")
        return

    DIRECCIONES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DIRECCIONES_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["texto", "lat", "lon", "nombre"])
        writer.writeheader()
        for row in todas:
            writer.writerow(row)
    print(f"💾 Guardado {len(todas)} filas en {DIRECCIONES_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Geocodificar direcciones de Recoger y rellenar data/direcciones_coordenadas.csv")
    parser.add_argument(
        "--csv",
        dest="csvs",
        action="append",
        required=True,
        help="Ruta a un CSV de servicios (puedes repetir la opción para varios archivos).",
    )
    parser.add_argument(
        "--columna-recoger",
        dest="col_recoger",
        default="Recoger",
        help="Nombre de la columna con la dirección de recogida (por defecto: Recoger).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Si >0, solo geocodifica las N direcciones más frecuentes.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignora y borra las direcciones ya geocodificadas (empieza de cero).",
    )
    args = parser.parse_args()

    rutas = [Path(p).expanduser() for p in args.csvs]
    contador = leer_direcciones_csv(rutas, args.col_recoger)
    if not contador:
        print("⚠️  No se encontraron direcciones válidas en los CSV indicados.")
        return

    print(f"📊 Direcciones únicas totales: {len(contador)}")
    if args.top > 0:
        comunes = [addr for addr, _ in contador.most_common(args.top)]
        print(f"🔎 Se geocodificarán solo las {len(comunes)} direcciones más frecuentes.")
        objetivo = comunes
    else:
        objetivo = list(contador.keys())

    # Si se pide reset, borramos el fichero de direcciones ya geocodificadas
    if args.reset and DIRECCIONES_FILE.exists():
        print(f"🧹 --reset activado: borrando {DIRECCIONES_FILE} para recalcular todo.")
        DIRECCIONES_FILE.unlink()

    ya_hechas = leer_existentes()
    nuevas = geocode_direcciones(objetivo, ya_hechas)
    guardar_direcciones(nuevas)


if __name__ == "__main__":
    main()

