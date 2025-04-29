# Práctica 02 - Metaheurísticas

Este repositorio contiene el código para la práctica 02 de la asignatura de Metaheurísticas. El objetivo es implementar y comparar algoritmos genéticos para resolver problemas de optimización.

## Estructura del Repositorio

- **`src/`**: Contiene el código fuente principal.
  - **`algorithms/`**: Implementaciones de algoritmos genéticos y de regresión.
  - **`functions/`**: Operadores genéticos como selección, cruce, mutación y reemplazo.
  - **`tools/`**: Utilidades, estadísticas y generación de gráficos.
  - **`scripts/`**: Scripts para medir y optimizar configuraciones.
  - **`view.py`**: Interfaz gráfica basada en Streamlit.
- **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.
- **`start.sh`**: Script para configurar el entorno y ejecutar la aplicación.
- **`end.sh`**: Script para limpiar el entorno de trabajo.

## Dependencias

El proyecto utiliza las siguientes dependencias, que se encuentran listadas en `requirements.txt`:

- `numpy`
- `scipy`
- `streamlit`
- `watchdog`
- `plotly`
- `scikit-optimize`

## Pasos para Iniciar la Práctica


1. **Configurar el Entorno Virtual**  
   Ejecute el script `start.sh` para crear un entorno virtual, instalar las dependencias y ejecutar la aplicación:
   ```bash
   ./start.sh
   ```

2. **Abrir la Interfaz**  
   La interfaz gráfica se abrirá automáticamente en su navegador. Si no ocurre, acceda manualmente a `http://localhost:8501`.

3. **Configurar y Ejecutar el Algoritmo**  
   - Configure los parámetros del algoritmo genético, como tamaño de población, generaciones y operadores genéticos.
   - Ejecute el algoritmo genético o comparelo con un modelo de regresión lineal.

4. **Limpiar el Entorno**  
   Una vez finalizada la práctica, ejecute el script `end.sh` para limpiar los archivos temporales y el entorno virtual:
   ```bash
   ./end.sh
   ```

## Interfaz Gráfica

La interfaz gráfica está implementada en `src/view.py` utilizando Streamlit. Permite:

- Configurar parámetros del algoritmo genético.
- Seleccionar operadores genéticos (selección, cruce, mutación, reemplazo).
- Ejecutar el modelo de islas o un algoritmo genético estándar.
- Comparar los resultados con un modelo de regresión lineal.

## Notas Adicionales

- Los datos de entrada se encuentran en `tools/utils.py` y se pueden modificar según sea necesario.
- Los resultados se visualizan en gráficos interactivos generados con Plotly.
- Consulte los scripts en `src/scripts/` para medir el rendimiento de los operadores genéticos.