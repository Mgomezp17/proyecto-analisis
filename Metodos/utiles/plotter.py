import numpy as np
import pandas as pd
import sympy as sp
from sympy import *
import plotly.graph_objects as go

def plot_fx_puntos(funcion,xinicial,xfinal,puntosx,puntosy,alias_punto):
    x = sp.Symbol('x')
    funcion_expr = parse_expr(funcion, local_dict={'x': x})
    intervalo_x = np.arange(xinicial, xfinal,0.1)
    fx = sp.lambdify(x, funcion_expr, 'numpy')
    intervalo_y = fx(intervalo_x)
    intervalo_x_completo = np.arange(xinicial, xfinal,0.1)
    intervalo_y_completo = fx(intervalo_x_completo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))
    fig.add_trace(go.Scatter(x=puntosx, y=puntosy, mode='markers', name=alias_punto))
    fig.update_layout(title=f'Función: {funcion} en intervalo [{xinicial}, {xfinal}]',
            xaxis_title='x',
            yaxis_title='f(x)')
    return fig

def fx_plot(funcion,xinicial,xfinal):
    x = sp.Symbol('x')
    funcion_expr = parse_expr(funcion, local_dict={'x': x})
    intervalo_x = np.arange(xinicial, xfinal,0.1)
    fx = sp.lambdify(x, funcion_expr, 'numpy')
    intervalo_y = fx(intervalo_x)
    intervalo_x_completo = np.arange(xinicial, xfinal,0.1)
    intervalo_y_completo = fx(intervalo_x_completo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))
    fig.update_layout(title=f'Función: {funcion} en intervalo [{xinicial}, {xfinal}]',
            xaxis_title='x',
            yaxis_title='f(x)')
    return fig
def spline_plot(fxs,puntosx,puntosy):
        try:
                x = sp.Symbol('x')
                fig = go.Figure()
                for i in range(len(fxs)):
                        funcion_expr = parse_expr(fxs[i], local_dict={'x': x})
                        intervalo_x = np.arange(puntosx[i], puntosx[i+1],0.1)
                        fx = sp.lambdify(x, funcion_expr, 'numpy')
                        intervalo_y = fx(intervalo_x)
                        intervalo_x_completo = np.arange(puntosx[i], puntosx[i+1],0.01)
                        intervalo_y_completo = fx(intervalo_x_completo)
                        fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name=f's(x{i})={fxs[i]}'))
                
                fig.update_layout(title=f'Funciones spline',
                        xaxis_title='x',
                        yaxis_title='f(x)')
                fig.add_trace(go.Scatter(x=puntosx, y=puntosy, mode='markers', name='Puntos dados'))
                return fig
        except Exception as e :
                print("Error plotting", e)