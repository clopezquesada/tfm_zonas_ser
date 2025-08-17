# app_parking.py
from shiny import App, ui, render, reactive
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random

# --- Función para calcular distancia haversine ---
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# --- UI ---
app_ui = ui.page_fluid(
    ui.h2("Mapa de Clusters de Aparcamiento"),
    
    ui.row(
        ui.column(3,
            ui.input_slider("radius", "Radio en km", min=0.1, max=5.0, step=0.1, value=1.0),
            ui.input_action_button("random_btn", "Generar nuevo punto aleatorio")
        ),
        ui.column(9,
            ui.output_plot("map_plot")
        )
    )
)

# --- Server ---
def server(input, output, session):
    
    # Datos de ejemplo: reemplaza esto por tu DataFrame gdf ya cargado
    gdf = pd.read_csv("gdf_final.csv")  # Asumimos que ya tiene latitud, longitud, facilidad_aparcamiento
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf['longitud'], gdf['latitud']), crs="EPSG:4326")
    
    # Variable reactiva para el punto aleatorio
    random_point = reactive.Value({
        "lat": random.uniform(40.410, 40.440),
        "lon": random.uniform(-3.725, -3.670)
    })
    
    # Actualizar punto aleatorio al pulsar el botón
    @reactive.Effect
    @reactive.event(input.random_btn)
    def _():
        random_point.set({
            "lat": random.uniform(40.410, 40.440),
            "lon": random.uniform(-3.725, -3.670)
        })
    
    # Generar mapa
    @output
    @render.plot
    def map_plot():
        rp = random_point.get()
        radius_km = input.radius()
        
        # Filtrar puntos dentro del radio
        gdf['distance_to_location'] = haversine_np(gdf['latitud'], gdf['longitud'], rp['lat'], rp['lon'])
        nearby_df = gdf[gdf['distance_to_location'] <= radius_km]
        
        color_discrete_map = {
            "Alta": "green",
            "Media": "red",
            "Baja": "orange"
        }
        
        fig = px.scatter_mapbox(
            nearby_df,
            lat="latitud",
            lon="longitud",
            color="facilidad_aparcamiento",
            zoom=11,
            height=600,
            mapbox_style="carto-positron",
            hover_data={
                "media_ocupacion": True,
                "numero_plazas": True
            },
            color_discrete_map=color_discrete_map
        )
        
        # Añadir el punto aleatorio
        fig.add_trace(go.Scattermapbox(
            lat=[rp['lat']],
            lon=[rp['lon']],
            mode='markers',
            marker=go.scattermapbox.Marker(size=14, color='black'),
            name="Punto aleatorio"
        ))
        
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(title="Clusters de Zonas de Aparcamiento")
        
        return fig

# --- Crear app ---
app = App(app_ui, server)
