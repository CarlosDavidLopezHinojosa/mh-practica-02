import scipy.stats as hyp
import numpy as np

def mean(data, axis=None):
    """
    Calcula la media de un conjunto de datos.
    Args:
        data (list or np.array): Datos numéricos.
    Returns:
        float: Media de los datos.
    """
    return np.mean(data,axis=axis)

def stdev(data):
    """
    Calcula la desviación estándar de un conjunto de datos.
    Args:
        data (list or np.array): Datos numéricos.
    Returns:
        float: Desviación estándar de los datos.
    """
    return np.std(data)

def wilcoxon(data1, data2):
    """
    Realiza la prueba de Wilcoxon para muestras emparejadas.
    Args:
        data1 (list or np.array): Primer conjunto de datos.
        data2 (list or np.array): Segundo conjunto de datos.
    Returns:
        dict: Resultados de la prueba con p-valor y estadístico.
    """
    test = hyp.wilcoxon(data1, data2)
    return {'pvalue': test.pvalue.item(), 'stat-test': 'Wilcoxon'}

def friedman(data):
    """
    Realiza la prueba de Friedman para muestras relacionadas.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
    Returns:
        dict: Resultados de la prueba con p-valor y estadístico.
    """
    test = hyp.friedmanchisquare(*data)
    return {'pvalue': test.pvalue.item(), 'stat-test': 'Friedman'}  
    
def multitest(data, alpha=0.05):
    """
    Realiza la prueba de Friedman y, si es significativa, aplica pruebas post hoc.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
        alpha (float): Nivel de significancia.
    Returns:
        dict: Resultados de la prueba de Friedman y pruebas post hoc si es necesario.
    """
    reject = friedman(data)['pvalue'] <= alpha
    test = {'stat-test': 'Friedman', 'reject': reject}
    return test

def dualtest(data1, data2, alpha=0.05):
    """
    Realiza la prueba de Wilcoxon para comparar dos muestras.
    Args:
        data1 (list or np.array): Primer conjunto de datos.
        data2 (list or np.array): Segundo conjunto de datos.
        alpha (float): Nivel de significancia.
    Returns:
        dict: Resultados de la prueba de Wilcoxon con la decisión sobre H0.
    """
    reject = wilcoxon(data1, data2)['pvalue'] <= alpha
    return {'stat-test': 'Wilcoxon', 'reject': reject}

def statistical_test(data: np.array, alpha):
    """
    Determina y ejecuta la prueba estadística adecuada según el número de muestras.
    Args:
        data (np.array): Datos organizados en muestras.
        alpha (float): Nivel de significancia.
    Returns:
        dict: Resultados de la prueba estadística adecuada.
    """
    return dualtest(*data, alpha) if len(data) == 2 else multitest(data, alpha)

def critical_distance(data, alpha=0.05):
    """
    Calcula la distancia crítica para la prueba de Nemenyi.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
        alpha (float): Nivel de significancia.
    Returns:
        float: Distancia crítica calculada.
    """
    k = len(data)
    N = len(data[0])  # Número de muestras por grupo
    q_alpha = hyp.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6 * N))

def nemenyi(data):
    """
    Realiza la prueba post hoc de Nemenyi tras la prueba de Friedman.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
    Returns:
        dict: Matriz de p-valores resultante de la prueba de Nemenyi.
    """
    ranks = mean(hyp.rankdata(data, axis=0), axis=1)
    return {'post-hoc': 'Nemenyi', 'ranks': ranks, 'critical-distance': critical_distance(data)}

def bonferroni(data: np.array, control: int, alpha=0.05):
    """
    Realiza una comparación One vs All utilizando la corrección de Bonferroni.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
        control (int): Índice del grupo de control.
    Returns:
        dict: Resultados de la prueba de Bonferroni.
    """

    algorithms, cases = data.shape
    ranks = np.mean(hyp.rankdata(data, axis=0),axis=1)
    z_friedman = lambda i, j: np.abs(ranks[i] - ranks[j]) / np.sqrt(algorithms * (algorithms + 1) / (6 * cases))
    pvalues = 2 * hyp.norm.sf(np.array([z_friedman(control, i) for i in range(algorithms) if i != control]))
    ajusted_pvalues = np.minimum(pvalues * (algorithms - 1), 1)

    reject = ajusted_pvalues <= alpha
    return {'post-hoc': 'Bonferroni', 'reject': reject, 'ranks': ranks, 'adjusted-pvalues': ajusted_pvalues}  

import plotly.graph_objects as go

def plot_nemenyi(nemenyi_result, labels):
    """
    Genera un gráfico de Nemenyi interactivo usando Plotly Figure Factory.
    
    Args:
        nemenyi_result (dict): Resultados de la prueba de Nemenyi, que incluyen:
            - 'ranks': Rangos promedio de los métodos.
            - 'critical-distance': Distancia crítica calculada.
        labels (list): Nombres de los métodos correspondientes a los rangos.
    
    Returns:
        fig (plotly.graph_objects.Figure): Figura interactiva del gráfico de Nemenyi.
    """
    ranks = np.array(nemenyi_result['ranks'])
    cd = nemenyi_result['critical-distance']
    
    # Ordenar los métodos por sus rangos
    sorted_indices = np.argsort(ranks)
    sorted_ranks = ranks[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]
    
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
            marker=dict(size=10, color='blue')
            ,name=label
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



def plot_bonferroni(bonferroni_results, labels=None, alpha=0.05, control=0):
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
    comp_labels = [labels[i] for i in range(n_algorithms) if i != control]
    
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