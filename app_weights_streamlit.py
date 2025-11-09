import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
import branca
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

BASE_DIR = os.path.dirname(__file__)
# Detecta automáticamente el CSV disponible para facilitar la demo
import glob
csv_files = glob.glob(os.path.join(BASE_DIR, '*.csv'))
CSV = csv_files[0] if csv_files else os.path.join(BASE_DIR, 'municipios_priorizados_2026.csv')
SHAPE_FILE = os.path.join(os.path.dirname(BASE_DIR), 'cartografia_censo2011_nacional', 'SECC_CPV_E_20111101_01_R_INE.shp')

st.set_page_config(page_title='Mapa interactivo – pesos de propensión', layout='wide')
st.title('Propensión 2026 – Ajuste dinámico de pesos')
st.caption('Mueve los deslizadores para recalcular y recolorear el mapa en tiempo real.')
st.caption(f"📊 Fuente de datos: {os.path.basename(CSV)}")

@st.cache_data(show_spinner=False, ttl=300)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f'No se encontró el CSV: {path}')
        st.stop()
    df = pd.read_csv(path, dtype={'CPRO': str, 'CMUN': str})
    df['CPRO'] = df['CPRO'].str.zfill(2)
    df['CMUN'] = df['CMUN'].str.zfill(3)
    # armonizar nombres
    if 'Poblacion_2026' in df.columns and 'Poblacion' not in df.columns:
        df = df.rename(columns={'Poblacion_2026':'Poblacion'})
    if 'renta_2026' in df.columns and 'renta_media' not in df.columns:
        df = df.rename(columns={'renta_2026':'renta_media'})
    if 'alquiler_2026' in df.columns and 'alquiler_m2' not in df.columns:
        df = df.rename(columns={'alquiler_2026':'alquiler_m2'})
    # asegurar columnas
    for c in ['Poblacion','renta_media','alquiler_m2','num_bancos']:
        if c not in df.columns:
            df[c] = np.nan
    # competencia: 1 si no hay bancos -> oportunidad
    df['competition_score'] = (df['num_bancos'].fillna(0) == 0).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_shapes(path: str) -> gpd.GeoDataFrame | None:
    """Carga shapefile si existe; devuelve None si no está disponible."""
    if not os.path.exists(path):
        return None
    try:
        g = gpd.read_file(path)
        g['CPRO'] = g['CPRO'].astype(str).str.zfill(2)
        g['CMUN'] = g['CMUN'].astype(str).str.zfill(3)
        try:
            if g.crs is None or '4326' not in str(g.crs):
                g = g.to_crs(epsg=4326)
        except Exception:
            pass
        return g
    except Exception:
        return None

def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    s = s.fillna(s.median())
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(1.0, index=s.index)
    return (s - mn) / (mx - mn)

# Load
cdf = load_csv(CSV)

# Componentes normalizados
cdf['pop_score'] = normalize(cdf['Poblacion'])
cdf['renta_score'] = 1 - normalize(cdf['renta_media'])   # renta baja = más social
cdf['alquiler_score'] = 1 - normalize(cdf['alquiler_m2']) # alquiler bajo = más social

# UI
with st.sidebar:
    st.header('Pesos (0–1)')
    w_pop = st.slider('Población', 0.0, 1.0, 0.25, 0.01)
    w_social = st.slider('Impacto social (renta baja)', 0.0, 1.0, 0.45, 0.01)
    w_rent = st.slider('Alquiler bajo', 0.0, 1.0, 0.25, 0.01)
    w_comp = st.slider('Oportunidad (competencia)', 0.0, 1.0, 0.05, 0.01)
    norm = st.checkbox('Normalizar para que sumen 1', value=False)
    palette = st.selectbox('Paleta', ['amarillos-rojos', 'azules', 'verdes'], index=0)
    st.markdown('---')
    st.subheader('Puntos')
    use_cluster = st.checkbox('Agrupar (MarkerCluster)', value=True, help='Los clusters muestran la SUMA de scores dentro del grupo.')
    show_numbers = st.checkbox('Mostrar número (puntuación) sobre cada punto', value=True, help='Visible cuando no hay cluster o cuando haces zoom y el cluster se abre.')
    show_breakdown = st.checkbox('Mostrar desglose en popup', value=True)
    point_radius = st.slider('Radio de punto', 4, 12, 7)

# normalizar si procede
weights = np.array([w_pop, w_social, w_rent, w_comp], dtype=float)
if norm and weights.sum() > 0:
    weights = weights / weights.sum()
    w_pop, w_social, w_rent, w_comp = weights.tolist()

# Preparar dataset: usar lat/lon del CSV si existen; si no, usar shapefile
has_coords = ('lat' in cdf.columns) and ('lon' in cdf.columns) and cdf['lat'].notna().any() and cdf['lon'].notna().any()
merged = None
if has_coords:
    merged = cdf[['CPRO','CMUN','NOMBRE','PROVINCIA','pop_score','renta_score','alquiler_score','competition_score','lat','lon']].dropna(subset=['lat','lon']).copy()
else:
    shapes = load_shapes(SHAPE_FILE)
    if shapes is None:
        st.error('No se encontró shapefile y el CSV no tiene columnas lat/lon. Añade columnas lat y lon al CSV o sube el shapefile para continuar.')
        st.stop()
    merged = shapes.merge(
        cdf[['CPRO','CMUN','NOMBRE','PROVINCIA','pop_score','renta_score','alquiler_score','competition_score']],
        on=['CPRO','CMUN'], how='inner'
    )

# score dinámico (multiplicado x1000 para valores grandes)
merged['score_dynamic'] = (
    merged['pop_score'] * w_pop +
    merged['renta_score'] * w_social +
    merged['alquiler_score'] * w_rent +
    merged['competition_score'] * w_comp
) * 1000.0

# contribuciones (para desgloses y punto numérico)
merged['contrib_pop'] = merged['pop_score'] * w_pop * 1000.0
merged['contrib_renta'] = merged['renta_score'] * w_social * 1000.0
merged['contrib_alquiler'] = merged['alquiler_score'] * w_rent * 1000.0
merged['contrib_comp'] = merged['competition_score'] * w_comp * 1000.0

# paletas: verde brillante (bajo) → amarillo → naranja → rojo intenso (alto)
if palette == 'amarillos-rojos':
    colors = ['#00ff00', '#9acd32', '#ffff00', '#ff8c00', '#ff0000']
elif palette == 'azules':
    colors = ['#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b']
else:
    colors = ['#f7fcf5','#c7e9c0','#7fcdbb','#41b6c4','#006d2c']

# Usar percentiles para distribuir colores equitativamente
p10 = float(merged['score_dynamic'].quantile(0.10))
p90 = float(merged['score_dynamic'].quantile(0.90))
vmin = p10
vmax = p90
colormap = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax)
colormap.caption = (
    f"Score = {w_pop:.2f}·Pobl + {w_social:.2f}·Social + {w_rent:.2f}·Alq + {w_comp:.2f}·Comp (p10-p90: {vmin:.0f}-{vmax:.0f})"
)

# mapa folium
m = folium.Map(location=[40.2, -3.7], zoom_start=6, tiles='CartoDB positron')

# (Eliminado) Capa de polígonos: sólo mostraremos puntos

# Coordenadas: si no vienen en CSV, calcular centroides del polígono
if not has_coords:
    # Centroides con reproyección métrica para mayor precisión
    try:
        centroids_metric = merged.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
        merged['lat'] = centroids_metric.y
        merged['lon'] = centroids_metric.x
    except Exception:
        merged['lat'] = merged.geometry.centroid.y
        merged['lon'] = merged.geometry.centroid.x

def popup_html(row):
    if not show_breakdown:
        return f"<b>{row['NOMBRE']}</b><br>Score: {row['score_dynamic']:.3f}"
    return (
        f"<b>{row['NOMBRE']}</b><br>Provincia: {row['PROVINCIA']}<br>"\
        f"Score total: <b>{row['score_dynamic']:.3f}</b><br>"\
        f"Población: {row['contrib_pop']:.3f}<br>"\
        f"Social (renta): {row['contrib_renta']:.3f}<br>"\
        f"Alquiler: {row['contrib_alquiler']:.3f}<br>"\
        f"Competencia: {row['contrib_comp']:.3f}"\
    )

# Colores escalados para puntos con mejor contraste
def point_color(score):
    # Asegurar que el color esté dentro del rango válido
    color = colormap(max(vmin, min(vmax, score)))
    return color

# Capa de puntos para TODOS los municipios
points_group = folium.FeatureGroup(name='Municipios (puntos)', show=True).add_to(m)
receiver = points_group
if use_cluster:
    # Clúster con icono personalizado que SUMA las puntuaciones leyendo el HTML del icono de cada marker
    icon_fn = r"""
    function(cluster){
      var markers = cluster.getAllChildMarkers();
      var sum = 0.0;
      for(var i=0;i<markers.length;i++){
        var html = markers[i].options && markers[i].options.icon && markers[i].options.icon.options && markers[i].options.icon.options.html;
        if(html){
          var m = html.match(/data-score=\"([0-9]+\.?[0-9]*)\"/);
          if(m){ sum += parseFloat(m[1]); }
        }
      }
      var label = sum.toFixed(0);
      return new L.DivIcon({html:'<div><span>'+label+'</span></div>', className:'marker-cluster marker-cluster-large', iconSize:new L.Point(40,40)});
    }
    """
    receiver = MarkerCluster(name='Municipios (cluster)', icon_create_function=icon_fn).add_to(points_group)

for _, r in merged.iterrows():
    if pd.isna(r['lat']) or pd.isna(r['lon']):
        continue
    # Color del fondo según score
    bg = point_color(r['score_dynamic'])
    if show_numbers:
        html_icon = f"""
        <div data-score=\"{r['score_dynamic']:.2f}\" style='background:{bg};color:#111;font-size:10px;font-weight:600;padding:4px 5px;min-width:34px;text-align:center;border-radius:18px;border:1px solid #222;'>
        {r['score_dynamic']:.0f}
        </div>
        """
    else:
        html_icon = f"""
        <div data-score=\"{r['score_dynamic']:.2f}\" style='background:{bg};width:{point_radius*2}px;height:{point_radius*2}px;border-radius:50%;border:1px solid #222;'></div>
        """
    folium.Marker(
        [r['lat'], r['lon']],
        popup=folium.Popup(popup_html(r), max_width=260),
        icon=folium.DivIcon(html=html_icon)
    ).add_to(receiver)

# --- Puntos estáticos objetivo (bancos a abrir) ---
TARGET_NOMBRES = ['Villanueva de la Serena', 'Vimianzo', 'Viator']
targets = merged[merged['NOMBRE'].isin(TARGET_NOMBRES)].copy()
if not targets.empty:
    # Agrupar por municipio y tomar el centroide promedio (solo 1 punto por municipio)
    targets_grouped = targets.groupby('NOMBRE', as_index=False).agg({
        'PROVINCIA': 'first',
        'lat': 'mean',
        'lon': 'mean',
        'score_dynamic': 'mean'
    })
    objetivos_group = folium.FeatureGroup(name='🏦 Objetivos (3 bancos)', show=True).add_to(m)
    for _, tr in targets_grouped.iterrows():
        # Icono distintivo: fondo morado y emoji banco (más pequeño)
        icon_html = f"""
        <div style='background:#6a00ff;color:#fff;font-size:11px;font-weight:700;padding:4px 6px;text-align:center;border-radius:12px;border:2px solid #000;box-shadow:0 0 8px rgba(106,0,255,0.5);'>
        🏦
        </div>
        """
        folium.Marker(
            [tr['lat'], tr['lon']],
            popup=folium.Popup(f"<b>🏦 {tr['NOMBRE']}</b><br>Provincia: {tr['PROVINCIA']}<br><b>Objetivo: Apertura banco</b><br>Score: {tr['score_dynamic']:.0f}", max_width=280),
            icon=folium.DivIcon(html=icon_html)
        ).add_to(objetivos_group)

colormap.add_to(m)
folium.LayerControl(collapsed=True).add_to(m)

# (Revertido) Capa con todos los puntos desactivada

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric('Municipios', f"{len(merged):,}")
c2.metric('Peso Población', f"{w_pop:.2f}")
c3.metric('Peso Social', f"{w_social:.2f}")
c4.metric('Peso Alquiler / Comp', f"{w_rent:.2f} / {w_comp:.2f}")

# Fuerza re-render del mapa al cambiar parámetros (clave única)
render_key = f"map-{w_pop:.3f}-{w_social:.3f}-{w_rent:.3f}-{w_comp:.3f}-{palette}-{int(use_cluster)}-{int(show_numbers)}-{point_radius}"
st_data = st_folium(m, width=None, height=700, key=render_key)

# Tabla top10
top10 = merged[['NOMBRE','PROVINCIA','score_dynamic']].sort_values('score_dynamic', ascending=False).head(10)
st.subheader('Top 10 municipios (score actual)')
st.dataframe(top10.reset_index(drop=True), width='stretch')

st.caption('Se muestran todos los municipios como puntos coloreados por score. El número (si está activado) refleja la puntuación dinámica y cambia al mover los pesos.')
