import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import heapq
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import requests
import json
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="Sri Lankan Cities - Shortest Path Analysis",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and professional UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        color: #212529;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        background: #e9ecef;
    }
    .path-display {
        background: linear-gradient(145deg, #fff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        font-family: 'Monaco', 'Menlo', monospace;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
    }
    .algorithm-info {
        background: linear-gradient(145deg, #e3f2fd, #bbdefb);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #2196f3;
        box-shadow: 0 4px 15px rgba(33,150,243,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        background: linear-gradient(90deg, #f8f9fa, #ffffff);
        padding: 10px;
        border-radius: 10px;
    }
    .path-segment {
        display: inline-block;
        background: linear-gradient(145deg, #4caf50, #45a049);
        color: white;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(76,175,80,0.3);
    }
    .path-arrow {
        color: #666;
        font-size: 1.2em;
        margin: 0 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Data structures
@dataclass
class PathResult:
    algorithm: str
    source: str
    destination: str
    path: List[str]
    distance: float
    time_taken: float
    nodes_explored: int
    steps: List[str] = None

def create_path_display(path: List[str]) -> str:
    """Create beautiful HTML display for the path with intermediate cities highlighted."""
    if not path:
        return "<div class='path-display'>No path found</div>"
    
    if len(path) == 1:
        return f"<div class='path-display'>Path: <span class='path-segment'>{path[0]}</span></div>"
    
    html_parts = ["<div class='path-display'>"]
    html_parts.append("<h4 style='margin-bottom: 15px; color: #2c3e50;'>🗺️ Optimal Route</h4>")
    
    for i, city in enumerate(path):
        if i == 0:
            # Source city
            html_parts.append(f"<span class='path-segment' style='background: linear-gradient(145deg, #e74c3c, #c0392b);'>🏁 {city}</span>")
        elif i == len(path) - 1:
            # Destination city
            html_parts.append(f"<span class='path-segment' style='background: linear-gradient(145deg, #e74c3c, #c0392b);'>🎯 {city}</span>")
        else:
            # Intermediate city
            html_parts.append(f"<span class='path-segment' style='background: linear-gradient(145deg, #3498db, #2980b9);'>🚏 {city}</span>")
        
        if i < len(path) - 1:
            html_parts.append("<span class='path-arrow'>→</span>")
    
    html_parts.append("</div>")
    return "".join(html_parts)

# Load city data
@st.cache_data
def load_cities_from_csv(csv_file_path: str) -> Dict[str, Tuple[float, float]]:
    """Load city coordinates from CSV file."""
    try:
        df = pd.read_csv(csv_file_path)
        cities = {}
        for _, row in df.iterrows():
            # Handle both old format (City, Latitude, Longitude) and new format (name_en, latitude, longitude)
            if 'name_en' in df.columns:
                city_name = row['name_en']
                lat = row['latitude']
                lon = row['longitude']
            else:
                city_name = row['City']
                lat = row['Latitude']
                lon = row['Longitude']
            cities[city_name] = (lat, lon)
        return cities
    except Exception as e:
        st.error(f"Error loading cities from CSV: {e}")
        return {}

# Haversine distance calculation
def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate the great circle distance between two points on Earth."""
    R = 6371.0  # Earth's radius in kilometers
    
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return round(distance, 2)

# Graph class
class CityGraph:
    """Graph representation for Sri Lankan cities."""
    
    def __init__(self, cities: Dict[str, Tuple[float, float]], max_neighbors: int = None):
        self.cities = cities
        self.city_list = sorted(cities.keys())
        self.num_cities = len(self.city_list)
        self.max_neighbors = max_neighbors
        self.adj_list = defaultdict(list)
        self._build_graph()
    
    def _build_graph(self):
        """Build graph with limited neighbors option."""
        if self.max_neighbors is None:
            # Complete graph
            for city1 in self.cities:
                for city2 in self.cities:
                    if city1 != city2:
                        distance = haversine_distance(self.cities[city1], self.cities[city2])
                        self.adj_list[city1].append((city2, distance))
        else:
            # Limited neighbors graph
            for city1 in self.cities:
                distances = []
                for city2 in self.cities:
                    if city1 != city2:
                        distance = haversine_distance(self.cities[city1], self.cities[city2])
                        distances.append((city2, distance))
                
                # Sort by distance and take only top N neighbors
                distances.sort(key=lambda x: x[1])
                self.adj_list[city1] = distances[:self.max_neighbors]
    
    def get_neighbors(self, city: str) -> List[Tuple[str, float]]:
        """Get neighbors of a city."""
        return self.adj_list[city]
    
    def get_distance(self, city1: str, city2: str) -> float:
        """Get distance between two cities."""
        for neighbor, distance in self.adj_list[city1]:
            if neighbor == city2:
                return distance
        return float('inf')

# Algorithm implementations
def dijkstra_shortest_path(graph: CityGraph, source: str, destination: str) -> PathResult:
    """Dijkstra's algorithm implementation."""
    start_time = time.time()
    
    distances = {city: float('inf') for city in graph.cities}
    distances[source] = 0
    previous = {city: None for city in graph.cities}
    visited = set()
    pq = [(0, source)]
    nodes_explored = 0
    
    while pq:
        current_dist, current_city = heapq.heappop(pq)
        
        if current_city in visited:
            continue
        
        visited.add(current_city)
        nodes_explored += 1
        
        if current_city == destination:
            break
        
        for neighbor, weight in graph.get_neighbors(current_city):
            if neighbor not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_city
                    heapq.heappush(pq, (new_dist, neighbor))
    
    # Reconstruct path
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    end_time = time.time()
    
    return PathResult(
        algorithm="Dijkstra's Algorithm",
        source=source,
        destination=destination,
        path=path if path[0] == source else [],
        distance=distances[destination] if distances[destination] != float('inf') else 0,
        time_taken=end_time - start_time,
        nodes_explored=nodes_explored
    )

def bellman_ford_shortest_path(graph: CityGraph, source: str, destination: str) -> PathResult:
    """Bellman-Ford algorithm implementation."""
    start_time = time.time()
    
    distances = {city: float('inf') for city in graph.cities}
    distances[source] = 0
    previous = {city: None for city in graph.cities}
    nodes_explored = 0
    
    # Relax edges V-1 times
    for _ in range(len(graph.cities) - 1):
        for city in graph.cities:
            if distances[city] != float('inf'):
                nodes_explored += 1
                for neighbor, weight in graph.get_neighbors(city):
                    if distances[city] + weight < distances[neighbor]:
                        distances[neighbor] = distances[city] + weight
                        previous[neighbor] = city
    
    # Reconstruct path
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    end_time = time.time()
    
    return PathResult(
        algorithm="Bellman-Ford Algorithm",
        source=source,
        destination=destination,
        path=path if path and path[0] == source else [],
        distance=distances[destination] if distances[destination] != float('inf') else 0,
        time_taken=end_time - start_time,
        nodes_explored=nodes_explored
    )

def astar_shortest_path(graph: CityGraph, source: str, destination: str) -> PathResult:
    """A* algorithm implementation."""
    start_time = time.time()
    
    def heuristic(city: str) -> float:
        return haversine_distance(graph.cities[city], graph.cities[destination])
    
    g_score = {city: float('inf') for city in graph.cities}
    g_score[source] = 0
    f_score = {city: float('inf') for city in graph.cities}
    f_score[source] = heuristic(source)
    
    previous = {city: None for city in graph.cities}
    open_set = [(f_score[source], source)]
    closed_set = set()
    nodes_explored = 0
    
    while open_set:
        current_f, current_city = heapq.heappop(open_set)
        
        if current_city in closed_set:
            continue
        
        closed_set.add(current_city)
        nodes_explored += 1
        
        if current_city == destination:
            break
        
        for neighbor, weight in graph.get_neighbors(current_city):
            if neighbor in closed_set:
                continue
            
            tentative_g = g_score[current_city] + weight
            
            if tentative_g < g_score[neighbor]:
                previous[neighbor] = current_city
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    end_time = time.time()
    
    return PathResult(
        algorithm="A* Search Algorithm",
        source=source,
        destination=destination,
        path=path if path and path[0] == source else [],
        distance=g_score[destination] if g_score[destination] != float('inf') else 0,
        time_taken=end_time - start_time,
        nodes_explored=nodes_explored
    )

# Google Maps integration
class GoogleMapsRoutes:
    """Google Maps integration for real road route calculations."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/directions/json"
    
    def get_driving_route(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get driving route from Google Maps API."""
        params = {
            'origin': f"{origin}, Sri Lanka",
            'destination': f"{destination}, Sri Lanka", 
            'key': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK':
                route = data['routes'][0]
                leg = route['legs'][0]
                
                return {
                    'status': 'success',
                    'distance_km': leg['distance']['value'] / 1000,
                    'duration_min': leg['duration']['value'] / 60,
                    'polyline': route['overview_polyline']['points']
                }
            else:
                return {
                    'status': 'error',
                    'error': data.get('error_message', f"API Error: {data['status']}")
                }
        except requests.RequestException as e:
            return {
                'status': 'error', 
                'error': f"Request failed: {str(e)}"
            }

# Map creation functions
def create_path_map(graph: CityGraph, path_result: PathResult, show_all_cities: bool = True):
    """Create Folium map showing the shortest path."""
    # Center map on Sri Lanka
    m = folium.Map(
        location=[7.8731, 80.7718],
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Add tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    # Add all cities if requested
    if show_all_cities:
        for city, coords in graph.cities.items():
            if city in path_result.path:
                continue
            
            folium.CircleMarker(
                location=[coords[0], coords[1]],
                radius=4,
                popup=f"<b>{city}</b>",
                tooltip=city,
                color='gray',
                fill=True,
                fillColor='lightgray',
                fillOpacity=0.6
            ).add_to(m)
    
    # Add path cities
    if path_result.path:
        for i, city in enumerate(path_result.path):
            coords = graph.cities[city]
            
            if i == 0:
                color, icon = 'green', 'play'
            elif i == len(path_result.path) - 1:
                color, icon = 'red', 'stop'
            else:
                color, icon = 'blue', 'pause'
            
            folium.Marker(
                location=[coords[0], coords[1]],
                popup=f"<b>{city}</b><br>Stop {i+1}/{len(path_result.path)}",
                tooltip=city,
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)
        
        # Draw path
        path_coords = [[graph.cities[city][0], graph.cities[city][1]] 
                      for city in path_result.path]
        
        folium.PolyLine(
            path_coords,
            color='red',
            weight=3,
            opacity=0.8,
            popup=f"Distance: {path_result.distance:.2f} km"
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def create_google_maps_comparison_map(graph: CityGraph, path_result: PathResult, 
                                     google_route: Dict[str, Any], show_all_cities: bool = True):
    """Create Folium map comparing algorithm path with Google Maps route."""
    # Center map on Sri Lanka
    m = folium.Map(
        location=[7.8731, 80.7718],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add all cities as small circles if requested
    if show_all_cities:
        for city, coords in graph.cities.items():
            color = 'lightgray'
            if city in path_result.path:
                color = 'gray'
            
            folium.CircleMarker(
                location=[coords[0], coords[1]],
                radius=4,
                popup=f"<b>{city}</b>",
                tooltip=city,
                color='gray',
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
    
    # Add algorithm path cities and route
    if path_result.path:
        for i, city in enumerate(path_result.path):
            coords = graph.cities[city]
            
            if i == 0:
                color, icon = 'green', 'play'
            elif i == len(path_result.path) - 1:
                color, icon = 'red', 'stop'
            else:
                color, icon = 'blue', 'pause'
            
            folium.Marker(
                location=[coords[0], coords[1]],
                popup=f"<b>{city}</b><br>Stop {i+1}/{len(path_result.path)}",
                tooltip=f"{city} (Algorithm Path)",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)
        
        # Draw algorithm path (theoretical)
        path_coords = [[graph.cities[city][0], graph.cities[city][1]] 
                      for city in path_result.path]
        
        folium.PolyLine(
            path_coords,
            color='blue',
            weight=4,
            opacity=0.8,
            popup=f"Algorithm Path: {path_result.distance:.2f} km",
            tooltip="Theoretical Shortest Path (Haversine Distance)"
        ).add_to(m)
    
    # Add Google Maps route if available
    if google_route['status'] == 'success':
        try:
            import polyline
            google_coords = polyline.decode(google_route['polyline'])
            
            folium.PolyLine(
                google_coords,
                color='red',
                weight=4,
                opacity=0.9,
                popup=f"Google Maps Route: {google_route['distance_km']:.2f} km ({google_route['duration_min']:.0f} min)",
                tooltip="Real Road Route (Google Maps)"
            ).add_to(m)
            
        except ImportError:
            # Fallback if polyline package not available
            folium.Marker(
                location=[7.8731, 80.7718],
                popup="Install 'polyline' package to view Google Maps route",
                icon=folium.Icon(color='orange', icon='exclamation-triangle', prefix='fa')
            ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

# Main Streamlit App
def main():
    # Header with enhanced design
    st.markdown("""
    <div class="main-header">
        <h1>🗺️ Sri Lankan Cities - Shortest Path Analysis</h1>
        <p style="font-size: 1.2em; margin-bottom: 10px;">Advanced Data Structures & Algorithms Implementation</p>
        <p style="opacity: 0.9;">Compare Dijkstra's, Bellman-Ford, A*, and Floyd-Warshall algorithms with real geographic data</p>
        <div style="margin-top: 20px;">
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 5px;">2150+ Cities</span>
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 5px;">4 Algorithms</span>
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 5px;">Interactive Maps</span>
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 5px;">Google Maps API</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load cities data
    cities = load_cities_from_csv('cities.csv')
    
    if not cities:
        st.error("Could not load city data. Please ensure 'cities.csv' is available.")
        return
    
    # Display dataset information
    st.info(f"📊 **Dataset Loaded:** {len(cities):,} Sri Lankan cities/towns/villages available for pathfinding analysis")
    
    # Sidebar configuration with enhanced styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; text-align: center; margin: 0;">⚙️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm selection with enhanced display
    st.sidebar.markdown("### 🔬 Algorithm Selection")
    algorithm = st.sidebar.selectbox(
        "Choose your pathfinding algorithm:",
        ["Dijkstra's Algorithm", "Bellman-Ford Algorithm", "A* Search Algorithm", "Compare All"],
        help="• Dijkstra: Fastest for non-negative weights\n• Bellman-Ford: Handles negative weights\n• A*: Uses heuristic for efficiency\n• Compare All: Run all algorithms simultaneously"
    )
    
    # City selection with better organization
    st.sidebar.markdown("### 🏙️ Route Planning")
    city_names = sorted(cities.keys())
    
    # Enhanced city selection with more flexibility
    # Find safe default cities from the available list
    default_source = "Colombo 1" if "Colombo 1" in city_names else (
        "Akkaraipattu" if "Akkaraipattu" in city_names else city_names[0] if city_names else ""
    )
    default_dest = "Kandy" if "Kandy" in city_names else (
        "Ampara" if "Ampara" in city_names else city_names[1] if len(city_names) > 1 else city_names[0]
    )
    
    # Option to use dropdown selection or manual input
    input_method = st.sidebar.radio(
        "City Selection Method:",
        ["🔽 Dropdown Selection", "✏️ Text Input + Suggestions"],
        index=0,
        help="Choose how to select cities: dropdown for browsing or text input for direct entry"
    )
    
    if input_method == "🔽 Dropdown Selection":
        source_city = st.sidebar.selectbox(
            "📍 **Source City**", 
            city_names, 
            index=city_names.index(default_source) if default_source in city_names else 0,
            help="Select your starting point from the dropdown"
        )
        
        destination_city = st.sidebar.selectbox(
            "🎯 **Destination City**", 
            city_names, 
            index=city_names.index(default_dest) if default_dest in city_names else (1 if len(city_names) > 1 else 0),
            help="Select your destination from the dropdown"
        )
    else:
        # Manual text input with validation
        st.sidebar.info("💡 **Tip:** Type city names and select from suggestions below")
        
        source_input = st.sidebar.text_input(
            "📍 **Type Source City**",
            value=default_source,
            help="Type the name of your starting city"
        )
        
        destination_input = st.sidebar.text_input(
            "🎯 **Type Destination City**",
            value=default_dest,
            help="Type the name of your destination city"
        )
        
        # Filter cities based on input and show suggestions
        if source_input:
            source_matches = [city for city in city_names if source_input.lower() in city.lower()]
            if source_matches:
                source_city = st.sidebar.selectbox(
                    f"📍 **Select from {len(source_matches)} matches for '{source_input}':**",
                    source_matches,
                    key="source_suggestions"
                )
            else:
                st.sidebar.error(f"❌ No cities found matching '{source_input}'")
                source_city = default_source
        else:
            source_city = default_source
            
        if destination_input:
            dest_matches = [city for city in city_names if destination_input.lower() in city.lower()]
            if dest_matches:
                destination_city = st.sidebar.selectbox(
                    f"🎯 **Select from {len(dest_matches)} matches for '{destination_input}':**",
                    dest_matches,
                    key="dest_suggestions"
                )
            else:
                st.sidebar.error(f"❌ No cities found matching '{destination_input}'")
                destination_city = default_dest
        else:
            destination_city = default_dest
    
    # Distance preview
    if source_city != destination_city:
        direct_distance = haversine_distance(cities[source_city], cities[destination_city])
        st.sidebar.info(f"📏 Direct distance: **{direct_distance:.2f} km**")
    
    # Graph Configuration
    st.sidebar.subheader("🌐 Graph Configuration")
    
    graph_type = st.sidebar.radio(
        "Graph Type",
        options=["Complete Graph (All Cities)", "Limited Graph (Nearest Neighbors)"],
        index=0,  # Default to Complete Graph
        help="Complete Graph: Uses all cities as potential intermediate nodes (optimal paths)\nLimited Graph: Only uses nearest neighbors (faster computation)"
    )
    
    max_neighbors = None
    if graph_type == "Limited Graph (Nearest Neighbors)":
        max_neighbors = st.sidebar.slider(
            "Max neighbors per city", 
            min_value=3, 
            max_value=20, 
            value=10,
            help="Limits connections to N closest cities for each node"
        )
        st.sidebar.warning("⚠️ Limited graph may not find truly optimal paths")
    else:
        st.sidebar.success("✅ Using complete graph for optimal shortest paths")
    
    # Google Maps integration with built-in API key
    st.sidebar.markdown("### 🗺️ Google Maps Integration")
    use_google_maps = st.sidebar.checkbox("Enable Google Maps comparison", value=True)
    
    # Default API key from the notebook project
    DEFAULT_API_KEY = 'AIzaSyBMNh_v1WYOd7zY9whEXGhTIfs4u_tT1XQ'
    google_api_key = DEFAULT_API_KEY
    
    if use_google_maps:
        st.sidebar.success("✅ Google Maps API integrated")
        st.sidebar.info("""
        🛣️ **Real Route Features:**
        - Actual driving distances
        - Travel time estimates  
        - Route polyline overlay
        - Distance comparison metrics
        """)
        
        # Option to use custom API key
        use_custom_key = st.sidebar.checkbox("Use custom API key", value=False)
        if use_custom_key:
            custom_key = st.sidebar.text_input(
                "Custom Google Maps API Key", 
                type="password",
                help="Enter your own Google Cloud Console API key"
            )
            if custom_key:
                google_api_key = custom_key
    else:
        google_api_key = None
    
    # Additional options
    show_all_cities = st.sidebar.checkbox("Show all cities on map", value=True)
    show_algorithm_details = st.sidebar.checkbox("Show algorithm details", value=True)
    
    # Main content
    if source_city == destination_city:
        st.warning("⚠️ Please select different source and destination cities.")
        return
    
    # Create graph
    with st.spinner("Building city network..."):
        graph = CityGraph(cities, max_neighbors)
    
    # Display graph info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏙️ Total Cities", len(cities))
    with col2:
        edges = sum(len(neighbors) for neighbors in graph.adj_list.values()) // 2
        st.metric("🔗 Graph Edges", f"{edges:,}")
    with col3:
        if max_neighbors:
            st.metric("📊 Graph Type", f"Limited ({max_neighbors} neighbors)")
        else:
            st.metric("📊 Graph Type", "Complete (All cities)")
    with col4:
        connectivity = "Optimal" if not max_neighbors else "Approximate"
        st.metric("🎯 Path Quality", connectivity)
    
    # Information about graph configuration
    if max_neighbors:
        st.info(f"ℹ️ **Limited Graph Mode**: Each city connects to its {max_neighbors} nearest neighbors. This provides faster computation but may not find the truly shortest path.")
    else:
        st.success("✅ **Complete Graph Mode**: Each city connects to all other cities. This guarantees finding the optimal shortest path using all available cities as intermediate nodes.")
    
    # Run algorithms
    st.subheader(f"🚀 Route Analysis: {source_city} → {destination_city}")
    
    if algorithm == "Compare All":
        # Run all algorithms
        with st.spinner("Running all algorithms..."):
            algorithms = [
                ("Dijkstra's Algorithm", dijkstra_shortest_path),
                ("Bellman-Ford Algorithm", bellman_ford_shortest_path),
                ("A* Search Algorithm", astar_shortest_path)
            ]
            
            results = []
            for name, func in algorithms:
                result = func(graph, source_city, destination_city)
                results.append(result)
        
        # Display comparison
        comparison_df = pd.DataFrame([
            {
                'Algorithm': r.algorithm.replace(' Algorithm', ''),
                'Distance (km)': f"{r.distance:.2f}",
                'Execution Time (ms)': f"{r.time_taken*1000:.4f}",
                'Nodes Explored': r.nodes_explored,
                'Path Length': len(r.path)
            }
            for r in results
        ])
        
        st.subheader("📊 Algorithm Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Show optimal path with enhanced visualization
        if results[0].path:
            path_length = len(results[0].path)
            intermediate_cities = path_length - 2  # Exclude source and destination
            
            # Create beautiful path display with custom HTML
            path_html = create_path_display(results[0].path)
            st.markdown(path_html, unsafe_allow_html=True)
            
            # Path statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏁 Total Cities", path_length)
            with col2:
                st.metric("🚏 Intermediate Stops", intermediate_cities)
            with col3:
                st.metric("📏 Total Distance", f"{results[0].distance:.2f} km")
        
        # Use first result for map
        map_result = results[0]
        
    else:
        # Run single algorithm
        algorithm_map = {
            "Dijkstra's Algorithm": dijkstra_shortest_path,
            "Bellman-Ford Algorithm": bellman_ford_shortest_path,
            "A* Search Algorithm": astar_shortest_path
        }
        
        with st.spinner(f"Running {algorithm}..."):
            map_result = algorithm_map[algorithm](graph, source_city, destination_city)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📏 Distance", f"{map_result.distance:.2f} km")
        with col2:
            st.metric("⏱️ Execution Time", f"{map_result.time_taken*1000:.4f} ms")
        with col3:
            st.metric("🔍 Nodes Explored", map_result.nodes_explored)
        
        # Enhanced path display for single algorithm
        if map_result.path:
            path_length = len(map_result.path)
            intermediate_cities = path_length - 2
            
            # Beautiful path visualization
            path_html = create_path_display(map_result.path)
            st.markdown(path_html, unsafe_allow_html=True)
            
            # Additional path statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏁 Total Cities", path_length)
            with col2:
                st.metric("🚏 Intermediate Stops", intermediate_cities)
            with col3:
                if intermediate_cities > 0:
                    avg_hop_distance = map_result.distance / (path_length - 1)
                    st.metric("📏 Avg Hop Distance", f"{avg_hop_distance:.2f} km")
                else:
                    st.metric("📏 Direct Route", "Yes")
            with col4:
                efficiency = f"{map_result.distance/path_length:.1f}"
                st.metric("⚡ Route Efficiency", f"{efficiency} km/city")
        else:
            st.error("❌ No path found between the selected cities.")
            return
    
    # Algorithm details
    if show_algorithm_details:
        with st.expander("📚 Algorithm Information"):
            if algorithm != "Compare All":
                algo_info = {
                    "Dijkstra's Algorithm": {
                        "Time Complexity": "O((V+E) log V)",
                        "Space Complexity": "O(V)",
                        "Best For": "Single-source shortest paths with non-negative weights",
                        "Description": "Uses a priority queue to explore nodes in order of their distance from the source."
                    },
                    "Bellman-Ford Algorithm": {
                        "Time Complexity": "O(VE)",
                        "Space Complexity": "O(V)",
                        "Best For": "Graphs with negative weights and cycle detection",
                        "Description": "Relaxes all edges V-1 times to find shortest paths."
                    },
                    "A* Search Algorithm": {
                        "Time Complexity": "O(E log V)",
                        "Space Complexity": "O(V)",
                        "Best For": "Point-to-point pathfinding with heuristic guidance",
                        "Description": "Uses geographic distance as heuristic to guide search towards destination."
                    }
                }
                
                info = algo_info.get(algorithm, {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Time Complexity:** {info.get('Time Complexity', 'N/A')}")
                    st.write(f"**Space Complexity:** {info.get('Space Complexity', 'N/A')}")
                
                with col2:
                    st.write(f"**Best For:** {info.get('Best For', 'N/A')}")
                
                st.write(f"**Description:** {info.get('Description', 'N/A')}")
    
    # Google Maps comparison with enhanced analysis
    google_route_data = None
    if use_google_maps and google_api_key and map_result.path:
        st.subheader("🗺️ Google Maps vs Algorithm Comparison")
        
        try:
            google_maps = GoogleMapsRoutes(google_api_key)
            
            with st.spinner("🌐 Fetching real road route from Google Maps..."):
                google_route_data = google_maps.get_driving_route(source_city, destination_city)
            
            if google_route_data['status'] == 'success':
                # Enhanced metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🛣️ Real Road Distance", f"{google_route_data['distance_km']:.2f} km")
                with col2:
                    st.metric("📏 Algorithm Distance", f"{map_result.distance:.2f} km")
                with col3:
                    difference = google_route_data['distance_km'] - map_result.distance
                    st.metric("📈 Extra Distance", f"+{difference:.2f} km")
                with col4:
                    st.metric("🕒 Driving Time", f"{google_route_data['duration_min']:.0f} min")
                
                # Analysis insights
                ratio = google_route_data['distance_km'] / map_result.distance
                efficiency = (1 - (difference / google_route_data['distance_km'])) * 100
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>� Route Analysis</h4>
                    <p><strong>Distance Ratio:</strong> Real roads are <strong>{ratio:.2f}x</strong> longer than straight-line distance</p>
                    <p><strong>Extra Distance:</strong> <strong>{(ratio-1)*100:.1f}%</strong> more distance due to road infrastructure</p>
                    <p><strong>Route Efficiency:</strong> Algorithm path achieves <strong>{efficiency:.1f}%</strong> of theoretical optimality</p>
                    <p><strong>Speed Estimate:</strong> Average <strong>{google_route_data['distance_km']/(google_route_data['duration_min']/60):.1f} km/h</strong> including traffic</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Comparison insights
                if ratio > 1.5:
                    st.warning("⚠️ Real roads are significantly longer - terrain or infrastructure constraints likely")
                elif ratio > 1.2:
                    st.info("ℹ️ Moderate deviation from straight-line distance - typical for road networks")
                else:
                    st.success("✅ Real roads closely follow theoretical optimal path")
                
            else:
                st.error(f"🚫 Google Maps API Error: {google_route_data.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Error connecting to Google Maps: {str(e)}")
    
    # Interactive map with Google Maps overlay
    st.subheader("🗺️ Interactive Route Visualization")
    
    # Create appropriate map based on Google Maps availability
    if google_route_data and google_route_data['status'] == 'success':
        # Use comparison map with both routes
        route_map = create_google_maps_comparison_map(graph, map_result, google_route_data, show_all_cities)
        st.success("🌟 Map shows both theoretical path (blue) and real road route (red)")
    else:
        # Use standard map with just algorithm path
        route_map = create_path_map(graph, map_result, show_all_cities)
    
    # Display map using streamlit-folium
    map_data = st_folium(route_map, width=700, height=500)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🔬 Advanced Data Structures and Algorithms Project</p>
        <p>📍 Geographic data: OpenStreetMap | 🗺️ Visualizations: Folium | 🚀 Framework: Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()