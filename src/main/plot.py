import matplotlib.pyplot as plt
import streamlit as st
import utils
def plot_predictions(predictions, actuals):
    """
    Plots the predictions against the actual values.

    Args:
        predictions (list): List of predicted values.
        actuals (list): List of actual values.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(predictions)), predictions, label='Predictions', color='blue')
    plt.scatter(range(len(actuals)), actuals, label='Actuals', color='red')
    plt.title('Predictions vs Actuals')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    st.pyplot(plt)
    plt.close()

def plot_function(coeficients):
    """
    Plots the function based on the given coefficients.

    Args:
        coeficients (list): Coefficients of the function.
    """
    x = [i / 10 for i in range(-20, 14)]
    y = [utils.f(i, coeficients) for i in x]
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Function', color='green')
    plt.title('Function Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    st.pyplot(plt)
    plt.close()