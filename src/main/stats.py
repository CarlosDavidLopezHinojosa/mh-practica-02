import statistics as stats
import scipy.stats as hyp
import scikit_posthocs as sp
import numpy as np

def mean(data):
    """
    Calcula la media de un conjunto de datos.
    Args:
        data (list or np.array): Datos numéricos.
    Returns:
        float: Media de los datos.
    """
    return stats.mean(data)

def stdev(data):
    """
    Calcula la desviación estándar de un conjunto de datos.
    Args:
        data (list or np.array): Datos numéricos.
    Returns:
        float: Desviación estándar de los datos.
    """
    return stats.stdev(data)

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
    return {'pvalue': test.pvalue, 'statistic': test.statistic, 'test-type': 'Wilcoxon'}

def friedman(data):
    """
    Realiza la prueba de Friedman para muestras relacionadas.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
    Returns:
        dict: Resultados de la prueba con p-valor y estadístico.
    """
    test = hyp.friedmanchisquare(*data)
    return {'pvalue': test.pvalue, 'statistic': test.statistic, 'test-type': 'Friedman'}

def nemenyi(data):
    """
    Realiza la prueba post hoc de Nemenyi tras la prueba de Friedman.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
    Returns:
        dict: Matriz de p-valores resultante de la prueba de Nemenyi.
    """
    test = np.array(sp.posthoc_nemenyi_friedman(data))
    return {'pvalue': test, 'test-type': 'Nemenyi'}

def bonferroni(data):
    """
    Realiza la corrección de Bonferroni como prueba post hoc.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
    Returns:
        dict: Matriz de p-valores resultante de la prueba de Bonferroni.
    """
    test = np.array(sp.posthoc_conover(data, p_adjust='bonferroni'))
    return {'pvalue': test, 'test-type': 'Bonferroni'}

def multitest(data, alpha=0.05):
    """
    Realiza la prueba de Friedman y, si es significativa, aplica pruebas post hoc.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
        alpha (float): Nivel de significancia.
    Returns:
        dict: Resultados de la prueba de Friedman y pruebas post hoc si es necesario.
    """
    friedman_test = friedman(data)
    if friedman_test['pvalue'] > alpha:
        return {'pvalue': friedman_test['pvalue'], 'test-type': 'Friedman', 'result': 'Fail to reject H0'}
    else:
        nemenyi_test = nemenyi(data)
        bonferroni_test = bonferroni(data)
        return {'pvalue': friedman_test['pvalue'], 'test-type': 'Friedman', 'result': 'Reject H0', 'nemenyi': nemenyi_test, 'bonferroni': bonferroni_test}

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
    wilcoxon_test = wilcoxon(data1, data2)
    if wilcoxon_test['pvalue'] > alpha:
        return {'pvalue': wilcoxon_test['pvalue'], 'test-type': 'Wilcoxon', 'result': 'Fail to reject H0'}
    else:
        return {'pvalue': wilcoxon_test['pvalue'], 'test-type': 'Wilcoxon', 'result': 'Reject H0'}

def statistical_test(data: np.array, alpha):
    """
    Determina y ejecuta la prueba estadística adecuada según el número de muestras.
    Args:
        data (np.array): Datos organizados en muestras.
        alpha (float): Nivel de significancia.
    Returns:
        dict: Resultados de la prueba estadística adecuada.
    """
    if len(data) == 2:
        return dualtest(*data, alpha)
    else:
        return multitest(data, alpha)