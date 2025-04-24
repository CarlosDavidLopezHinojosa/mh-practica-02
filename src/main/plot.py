import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import tools.utils as utils
import tools.stats as stats

OUTPUT_FOLDER = "../plots/"

def generate_colors(n):
    """
    Genera una lista de colores únicos para `n` elementos.
    Args:
        n (int): Número de colores a generar.
    Returns:
        list: Lista de colores en formato hexadecimal.
    """
    return pc.qualitative.Set3[:n] if n <= len(pc.qualitative.Set3) else pc.qualitative.Plotly[:n]

def plot_nemenyi(nemenyi_result, labels):
    """
    Genera un gráfico de Nemenyi interactivo usando Plotly Figure Factory.
    """
    ranks = np.array(nemenyi_result['ranks'])
    cd = nemenyi_result['critical-distance']
    
    # Ordenar los métodos por sus rangos
    sorted_indices = np.argsort(ranks)
    sorted_ranks = ranks[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]
    colors = generate_colors(len(sorted_labels))

    # Crear la figura
    fig = go.Figure()
    
    # Agregar puntos para cada método
    for i, (rank, label) in enumerate(zip(sorted_ranks, sorted_labels)):
        fig.add_trace(go.Scatter(
            x=[rank],
            y=[1],
            mode='markers+text',
            text=[label],
            textposition='top center',
            marker=dict(size=10, color=colors[i]),
            name=label
        ))
    
    # Agregar línea horizontal
    fig.add_trace(go.Scatter(
        x=[min(sorted_ranks) - cd, max(sorted_ranks) + cd],
        y=[1, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ))
    
    # Dibujar el Critical Difference (CD)
    fig.add_trace(go.Scatter(
        x=[min(sorted_ranks), min(sorted_ranks) + cd],
        y=[1.2, 1.2],
        mode='lines+text',
        text=[f"CD = {cd:.2f}", ""],
        textposition='top center',
        line=dict(color='red', width=2),
        showlegend=False
    ))
    
    # Configuración del gráfico
    fig.update_layout(
        title='Gráfico de Nemenyi',
        xaxis_title='Rango promedio',
        yaxis=dict(showticklabels=False, showgrid=False),
        xaxis=dict(showgrid=True),
        plot_bgcolor='white'
    )
    
    return fig



def plot_bonferroni(bonferroni_results, labels=None, alpha=0.05):
    """
    Ejecuta la prueba de Bonferroni y genera un gráfico de barras interactivo de los p-valores ajustados.
    
    Args:
        data (np.array): Datos organizados con forma (algorithms, cases).
        control (int): Índice del grupo de control.
        labels (list, opcional): Lista de nombres para los algoritmos. Si no se provee, se usan índices.
        alpha (float, opcional): Nivel de significancia.
    
    Returns:
        fig (go.Figure): Figura interactiva con el gráfico de barras.
    """
    # Ejecutar la prueba de Bonferroni
    result = bonferroni_results
    adjusted_pvalues = result['adjusted-pvalues']
    
    # Definir etiquetas para las comparaciones (todos excepto el grupo control)
    n_algorithms = len(adjusted_pvalues)
    if labels is None:
        labels = [f'Algoritmo {i}' for i in range(n_algorithms)]
    comp_labels = [labels[i] for i in range(n_algorithms) if i != result['control']]
    
    # Colores: se asigna un color según si se rechaza o no H0 (p-valor ajustado ≤ α)
    colors = ['green' if p <= alpha else 'red' for p in adjusted_pvalues]
    
    # Crear el gráfico de barras
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_labels,
        y=adjusted_pvalues,
        marker_color=colors,
        text=[f"{p:.3f}" for p in adjusted_pvalues],
        textposition='auto'
    ))
    
    # Línea de referencia para el nivel de significancia
    fig.add_hline(
        y=alpha, 
        line_dash="dash", 
        line_color="black",
        annotation_text=f"α = {alpha}",
        annotation_position="bottom right"
    )
    
    # Configuración final del layout
    fig.update_layout(
        title="P-valores Ajustados (Bonferroni) One vs All",
        xaxis_title="Algoritmo",
        yaxis_title="P-valor ajustado",
        yaxis=dict(range=[0, max(max(adjusted_pvalues), alpha) * 1.1]),
        plot_bgcolor='white'
    )
    
    return fig

def plot_times(data, labels=None):
    """
    Genera un gráfico de barras interactivo para los tiempos de ejecución de diferentes algoritmos.
    
    Args:
        data (np.array): Datos organizados con forma (algorithms, cases).
        labels (list, opcional): Lista de nombres para los algoritmos. Si no se provee, se usan índices.
    
    Returns:
        fig (go.Figure): Figura interactiva con el gráfico de barras.
    """
    # Calcular el tiempo promedio por algoritmo
    mean_times = np.mean(data, axis=1)
    
    # Definir etiquetas para los algoritmos
    n_algorithms = len(mean_times)
    if labels is None:
        labels = [f'Algoritmo {i}' for i in range(n_algorithms)]
    colors = generate_colors(n_algorithms)

    # Crear el gráfico de barras
    fig = go.Figure()
    for i, (label, time) in enumerate(zip(labels, mean_times)):
        fig.add_trace(go.Bar(
            x=[label],
            y=[time],
            text=[f"{time:.2f}"],
            textposition='auto',
            marker_color=colors[i],
            name=label
        ))

    # Configuración final del layout
    fig.update_layout(
        title="Tiempos de Ejecución Promedio por Algoritmo",
        xaxis_title="Algoritmo",
        yaxis_title="Tiempo (s)",
        plot_bgcolor='white'
    )
    
    return fig

def plot_memory(data, labels=None):
    """
    Genera un gráfico de barras interactivo para el uso de memoria de diferentes algoritmos.
    
    Args:
        data (np.array): Datos organizados con forma (algorithms, cases).
        labels (list, opcional): Lista de nombres para los algoritmos. Si no se provee, se usan índices.
    
    Returns:
        fig (go.Figure): Figura interactiva con el gráfico de barras.
    """
    # Calcular el uso promedio de memoria por algoritmo
    mean_memory = np.mean(data, axis=1)
    
    # Definir etiquetas para los algoritmos
    n_algorithms = len(mean_memory)
    if labels is None:
        labels = [f'Algoritmo {i}' for i in range(n_algorithms)]
    colors = generate_colors(n_algorithms)

    # Crear el gráfico de barras
    fig = go.Figure()
    for i, (label, memory) in enumerate(zip(labels, mean_memory)):
        fig.add_trace(go.Bar(
            x=[label],
            y=[memory],
            text=[f"{memory:.2f}"],
            textposition='auto',
            marker_color=colors[i],
            name=label
        ))

    # Configuración final del layout
    fig.update_layout(
        title="Uso Promedio de Memoria por Algoritmo",
        xaxis_title="Algoritmo",
        yaxis_title="Memoria (MB)",
        plot_bgcolor='white'
    )
    
    return fig


def plot_convergences(data, labels=None):
    """
    Genera un gráfico de líneas interactivo para las convergencias de diferentes algoritmos.
    
    Args:
        data (np.array): Datos organizados con forma (algorithms, cases).
        labels (list, opcional): Lista de nombres para los algoritmos. Si no se provee, se usan índices.
    
    Returns:
        fig (go.Figure): Figura interactiva con el gráfico de líneas.
    """
    # Calcular la convergencia promedio por algoritmo
    mean_convergences = np.mean(data, axis=1)
    
    # Definir etiquetas para los algoritmos
    n_algorithms = len(mean_convergences)
    if labels is None:
        labels = [f'Algoritmo {i}' for i in range(n_algorithms)]
    colors = generate_colors(n_algorithms)

    # Crear el gráfico de líneas
    fig = go.Figure()
    for i, (label, convergence) in enumerate(zip(labels, data)):
        fig.add_trace(go.Scatter(
            x=np.arange(len(convergence)),
            y=convergence,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[i]),
            marker=dict(color=colors[i]),
            text=[f"{c:.2f}" for c in convergence],
            textposition='top center'
        ))

    # Configuración final del layout
    fig.update_layout(
        title="Convergencias por Algoritmo",
        xaxis_title="Generación",
        yaxis_title="Convergencia",
        plot_bgcolor='white'
    )
    return fig

def process_file(file_path):
    """
    Procesa un archivo JSON de resultados y genera gráficos de tiempo, memoria, convergencia y Nemenyi.
    Args:
        file_path (str): Ruta del archivo JSON.
        file_name (str): Nombre del archivo (sin extensión).
    """
    data = utils.load(file_path)
    labels = list(data.keys())

    # Ajustar el tamaño de los datos para que todos tengan la misma longitud
    size = min([len(data[label]['convergences']) for label in labels])
    times = [data[label]['time'][:size] for label in labels]
    memory = [data[label]['memory'][:size] for label in labels]
    convergences = [data[label]['convergences'][:size] for label in labels]

    # Generar gráficos
    fig_times = plot_times(np.array(times), labels)

    fig_memory = plot_memory(np.array(memory), labels)

    fig_convergences = plot_convergences(np.array(convergences), labels)

    nemenyi_result = stats.nemenyi(np.array(convergences))
    fig_nemenyi = plot_nemenyi(nemenyi_result, labels)

    return {
        "times": fig_times,
        "memory": fig_memory,
        "convergences": fig_convergences,
        "nemenyi": fig_nemenyi
    }


operator_files = {
    "selectors": "info/selectors.json",
    "crossings": "info/crossings.json",
    "mutations": "info/mutations.json",
    "replacements": "info/replacements.json"
}
