# MH Práctica 2: Optimización Genética con Modelo de Islas

Este proyecto implementa un algoritmo genético basado en el modelo de islas para resolver problemas de optimización. El objetivo principal es encontrar los coeficientes que mejor se ajusten a un conjunto de datos, minimizando el error cuadrático medio (MSE). Además, se incluye una comparación con un modelo de regresión lineal tradicional.

## Características

- **Modelo de Islas**: Paralelización del algoritmo genético en múltiples islas.
- **Operadores Genéticos**: Métodos configurables de selección, cruce, mutación y reemplazo.
- **Visualización**: Gráficas interactivas para analizar los resultados.
- **Comparación**: Evaluación del rendimiento frente a un modelo de regresión lineal.

## Requisitos

Asegúrate de tener instaladas las siguientes dependencias. Puedes instalarlas ejecutando:

```bash
pip install -r requirements.txt
```

### Dependencias

- `numpy`
- `scipy`
- `scikit-posthocs`
- `streamlit`
- `matplotlib`
- `watchdog`

## Estructura del Proyecto

```
metaheuristicas/mh-practica-02/
├── src/
│   ├── main/
│   │   ├── algorithms/
│   │   │   ├── genetic.py          # Implementación del modelo de islas
│   │   │   ├── regression.py       # Ajuste de regresión polinómica
│   │   ├── functions/
│   │   │   ├── selection.py        # Métodos de selección
│   │   │   ├── crossing.py         # Métodos de cruce
│   │   │   ├── mutation.py         # Métodos de mutación
│   │   │   ├── replacement.py      # Métodos de reemplazo
│   │   ├── tools/
│   │   │   ├── utils.py            # Funciones auxiliares
│   │   │   ├── stats.py            # Pruebas estadísticas
│   │   ├── view.py                 # Interfaz de usuario con Streamlit
│   │   ├── plot.py                 # Gráficas con Matplotlib
│   ├── __init__.py
├── requirements.txt                # Dependencias del proyecto
├── start.sh                        # Script para iniciar la aplicación
├── .gitignore                      # Archivos ignorados por Git
└── README.md                       # Documentación del proyecto
```

## Uso

### Ejecutar la Aplicación

Para iniciar la interfaz de usuario, ejecuta el siguiente comando en la terminal:

```bash
bash start.sh
```

Esto configurará un entorno virtual, instalará las dependencias y ejecutará la aplicación en Streamlit.

### Configuración del Algoritmo

En la barra lateral de la aplicación, puedes configurar los siguientes parámetros:

- **Número de Islas**: Define cuántas islas se utilizarán en el modelo.
- **Tamaño de la Población por Isla**: Número de individuos en cada isla.
- **Número de Generaciones**: Cantidad de iteraciones del algoritmo.
- **Operadores Genéticos**: Métodos de selección, cruce, mutación y reemplazo.

### Resultados

La aplicación muestra:

1. **Coeficientes Encontrados**: Los valores óptimos obtenidos por el algoritmo genético.
2. **MSE**: Error cuadrático medio del modelo.
3. **Gráficas**:
   - Comparación entre predicciones y valores reales.
   - Representación de la función ajustada.

### Comparación con Regresión Lineal

Puedes ejecutar un modelo de regresión lineal para comparar su rendimiento con el algoritmo genético. La aplicación mostrará los coeficientes ajustados, el MSE y las gráficas correspondientes.

## Advertencia

El algoritmo genético puede ser intensivo en recursos. Configura los parámetros con cuidado para evitar un uso excesivo de memoria o tiempo de ejecución prolongado.
