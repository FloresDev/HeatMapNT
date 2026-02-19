# Revisión de coordenadas por zona

Coordenadas contrastadas con direcciones y referencias oficiales (Wikipedia, ayuntamientos, 123coordenadas, etc.) para las zonas que aparecen en el CSV.

## Zonas principales (con coordenadas verificadas)

| Código | Nombre | Dirección / Referencia | Lat | Lon | Notas |
|--------|--------|------------------------|-----|-----|--------|
| ALC, COB | Alcobendas | Centro / municipio norte Madrid | 40.5472 | -3.6414 | OK (alternativa centro 40.537, -3.637) |
| SSR, SAN | San Sebastián de los Reyes | Centro, plaza Constitución / Los Tempranales | 40.5474 | -3.6261 | Corregido (antes caía al sur de Madrid) |
| LMO | La Moraleja | Urbanización Alcobendas, A-1 | 40.515 | -3.651 | Corregido (antes -3.614) |
| LTB | Las Tablas | Barrio Valverde, Fuencarral-El Pardo | 40.504 | -3.672 | Corregido |
| SCH | San Chinarro | PAU Hortaleza, Madrid | 40.498 | -3.673 | OK |
| ALG | Algete | Centro municipio, 28110 | 40.5965 | -3.5016 | OK |
| ANB | Anabel | Calle Anabel Segura / Centro Cívico, Alcobendas 28108 (Distrito Urbanizaciones) | 40.545 | -3.638 | OK |
| DIV | Diversia | Av. Bruselas 21, 28109 Alcobendas (Kinépolis / centro ocio) | 40.543 | -3.6395 | Corregido |
| URBA | Urbanizaciones | Distrito Urbanizaciones Alcobendas/Sanse | 40.548 | -3.628 | OK |
| T4 | Barajas T4 | Terminal 4 Aeropuerto Adolfo Suárez | 40.4915 | -3.5917 | Corregido (lon -3.567 → -3.5917) |
| BAR | Barajas | Zona aeropuerto / distrito | 40.473 | -3.567 | OK |
| MAD | Madrid Centro | Referencia centro ciudad | 40.4168 | -3.7038 | OK |
| CNT | Centro | Cercanías / Sol área | 40.417 | -3.704 | OK |
| ATO, E.ATO | Atocha | Estación Puerta de Atocha | 40.407 | -3.691 | OK |
| CHA1, CHA2, CHAMB, E.CHA | Chamartín | Estación Chamartín-Clara Campoamor | 40.46 | -3.68 | OK |
| GET | Getafe | Centro municipio | 40.305 | -3.732 | OK |
| LEG | Leganés | Centro municipio | 40.326 | -3.764 | OK |
| MOS | Móstoles | Centro municipio | 40.3223 | -3.8644 | OK |
| FUE | Fuenlabrada | Centro municipio | 40.289 | -3.798 | Corregido (antes genérico Madrid) |
| ALCOR | Alcorcón | Centro | 40.349 | -3.824 | OK |
| TOR, VTOR | Torrejón de Ardoz | Centro / zona | 40.4615 | -3.4975 / 40.46, -3.50 | OK |
| APC1, APC1B, APC2 | Alcalá de Henares | Centro / Plaza Cervantes | 40.482 | -3.364 | OK |
| APC6, VILL | Collado Villalba | Centro municipio | 40.643 | -3.993 | Corregido |

## Zonas genéricas (40.42, -3.70)

Varias zonas del CSV (3C, 4VI, A1, AML, ARA, etc.) siguen con coordenadas genéricas de Madrid porque no hay una única dirección; son códigos internos o zonas amplias. Si en tus datos alguna corresponde a un lugar concreto, se puede sustituir en `zonas_coordenadas.csv` por sus lat/lon reales.

## Cómo revisar más zonas

1. Localizar el código de zona en el CSV de servicios (columna **Zona**).
2. Si tienes columna **Recoger** (dirección), buscar esa dirección en Google Maps / OpenStreetMap y tomar lat/lon.
3. Actualizar en `data/zonas_coordenadas.csv` la fila `zona,lat,lon,nombre` correspondiente.
