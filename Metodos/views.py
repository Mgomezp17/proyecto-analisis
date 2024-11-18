from django.shortcuts import render
import numpy as np
import pandas as pd
import sympy as sp
from sympy import *
from django.utils.safestring import mark_safe
import plotly.graph_objects as go
from .metodos.iterativos import metodo_gauss_seidel,metodo_jacobi,metodo_sor
from .utiles.saver import dataframe_to_txt,plot_to_png,text_to_txt
from .utiles.plotter import plot_fx_puntos,fx_plot,spline_plot



import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.conf import settings
import plotly.io as pio



def home(request):
    return render(request, 'All_methods.html')


def reglaFalsaView(request):
    context = {}
    
    if request.method == 'POST':
        try:
            fx = request.POST["funcion"]

            x0 = request.POST["a"]
            X0 = float(x0)

            xi = request.POST["b"]
            Xi = float(xi)

            tol = request.POST["tolerancia"]
            Tol = float(tol)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            tipo_error = request.POST.get('tipo_error')
            
            datos = reglaFalsa(X0, Xi, Niter, Tol, fx, tipo_error)

            if "errors" in datos and datos["errors"]:
                context['error_message'] = f'Hubo un error en el metodo Regla Falsa en: {datos["errors"]}'
                return render(request, 'error.html', context)
            
            if "results" in datos:
                df = pd.DataFrame(datos["results"], columns=datos["columns"])
                
                x = sp.symbols('x')
                funcion_expr = sp.sympify(fx)

                xi_copy = X0
                xs_copy = Xi

                intervalo_x = np.arange(xi_copy, xs_copy, 0.1)
                fx_func = sp.lambdify(x, funcion_expr, 'numpy')
                intervalo_y = fx_func(intervalo_x)

                
                grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')

                
                if os.path.exists(grafico_path):
                    os.remove(grafico_path)

              
                plt.figure(figsize=(10, 6))
                plt.plot(intervalo_x, intervalo_y, label='f(x)', color='blue')

                if datos.get("root") is not None:
                    plt.scatter([float(datos["root"])], [0], color='red', zorder=5, label='Raíz hallada')

                plt.title(f'Función: {fx} en intervalo [{xi_copy}, {xs_copy}]')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.grid(True)
                plt.legend()

                
                plt.savefig(grafico_path, format='svg')
                plt.close()  

               
                plot_html = f'<img src="/static/graficoa.svg" alt="Gráfico de la función">'

                
                dataframe_to_txt(df, f'Regla Falsa{fx}')
                
                context = {'df': df.to_html(), 'plot_html': plot_html, 'mensaje': f'La solución es: {datos["root"]}', 'nombre_metodo': "Regla Falsa"}
        except Exception as e:
            context = {'error_message': f'Hubo un error en el metodo Regla Falsa en: {str(e)}'}
            return render(request, 'error.html', context)

    return render(request, 'one_method.html', context)

def reglaFalsa(a, b, Niter, Tol, fx, tipo_error):

    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"],
        "results": [],
        "errors": [],
        'root': '0'
    }

  
    datos = list()
    x = sp.Symbol('x')
    i = 1
    cond = Tol
    error = 1.0000000

    Fun = sp.sympify(fx)
   
    xm = 0
    xm0 = 0
    Fx_2 = 0
    Fx_3 = 0
    Fa = 0
    Fb = 0

    try:
        if Fun.subs(x, a)*Fun.subs(x, b) > 0:
            output["errors"].append("La funcion no cambia de signo en el intervalo dado")
            return output
        while (error > cond) and (i < Niter):
            if i == 1:
                Fx_2 = Fun.subs(x, a)
                Fx_2 = Fx_2.evalf()
                Fa = Fx_2

                Fx_2 = Fun.subs(x, b)
                Fx_2 = Fx_2.evalf()
                Fb = Fx_2

                xm = (Fb*a - Fa*b)/(Fb-Fa)
                Fx_3 = Fun.subs(x, xm)
                Fx_3 = Fx_3.evalf()
                datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fx_3)])
            else:

                if (Fa*Fx_3 < 0):
                    b = xm
                else:
                    a = xm

                xm0 = xm
                Fx_2 = Fun.subs(x, a) 
                Fx_2 = Fx_2.evalf()
                Fa = Fx_2

                Fx_2 = Fun.subs(x, b)
                Fx_2 = Fx_2.evalf()
                Fb = Fx_2

                xm = (Fb*a - Fa*b)/(Fb-Fa) 

                Fx_3 = Fun.subs(x, xm) 
                Fx_3 = Fx_3.evalf()

                if tipo_error == "absoluto":
                    error = Abs(xm-xm0)
                    er = sp.sympify(error)
                    error = er.evalf()
                else:
                    error = Abs(xm-xm0)/xm
                    er = sp.sympify(error)
                    error = er.evalf()

                datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fx_3), '{:^15.7E}'.format(error)])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = datos
    output["root"] = xm
    return output


def puntoFijo(X0, Tol, Niter, fx, gx, tipo_error):
    output = {
        "columns": ["iter", "xi", "g(xi)", "f(xi)", "E"],
        "results": [],
        "errors": [],
        "root": None
    }

  
    x = sp.Symbol('x')
    i = 1
    error = 1.000
    Fx = sp.sympify(fx)
    Gx = sp.sympify(gx)

    xP = X0
    xA = 0.0

    Fa = Fx.subs(x, xP).evalf()
    Ga = Gx.subs(x, xP).evalf()

    datos = [[0, float(xP), float(Ga), float(Fa), None]]

    try:
        while error > Tol and i < Niter:
            Ga = Gx.subs(x, xP)
            xA = Ga.evalf()
            Fa = Fx.subs(x, xA).evalf()

            if tipo_error == "absoluto":
                error = abs(xA - xP)
            else:
                error = abs(xA - xP) / abs(xA)

            datos.append([i, float(xA), float(Ga), float(Fa), float(error)])
            xP = xA
            i += 1

        output["results"] = datos
        output["root"] = xA

   
        intervalo_x = np.linspace(X0 - 5, X0 + 5, 400)
        fx_func = sp.lambdify(x, Fx, 'numpy')
        gx_func = sp.lambdify(x, Gx, 'numpy')
        intervalo_y_f = fx_func(intervalo_x)
        intervalo_y_g = gx_func(intervalo_x)

     
        grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')

      
        if os.path.exists(grafico_path):
            os.remove(grafico_path)

   
        plt.figure(figsize=(10, 6))
        plt.plot(intervalo_x, intervalo_y_f, label='f(x)', color='blue')
        plt.plot(intervalo_x, intervalo_y_g, label='g(x)', color='green')

        if output["root"] is not None:
            plt.scatter([float(output["root"])], [0], color='red', zorder=5, label='Raíz hallada')

        plt.title(f'Funciones: f(x) = {fx} y g(x) = {gx} en intervalo [{X0 - 5}, {X0 + 5}]')
        plt.xlabel('x')
        plt.ylabel('f(x) / g(x)')
        plt.grid(True)
        plt.legend()

      
        plt.savefig(grafico_path, format='svg')
        plt.close()

    except BaseException as e:
        output["errors"].append("Error en los datos: " + str(e))
    
    return output



def puntoFijoView(request):
    if request.method == 'POST':
        try:
            fx = request.POST["funcion-F"]
            gx = request.POST["funcion-G"]
            
            
            try:
                x0 = float(request.POST["vInicial"])
            except ValueError:
                raise ValueError("El valor inicial (vInicial) no es un número válido.")
            
            try:
                tol = float(request.POST["tolerancia"])
            except ValueError:
                raise ValueError("La tolerancia no es un número válido.")
            
            try:
                niter = int(request.POST["iteraciones"])
            except ValueError:
                raise ValueError("El número de iteraciones no es un valor entero válido.")
            
            tipo_error = request.POST.get('tipo_error')

      
            datos = puntoFijo(x0, tol, niter, fx, gx, tipo_error)
            df = pd.DataFrame(datos["results"], columns=datos["columns"])

            
            root_value = datos["root"]

           
            plot_html = f'<img src="/static/graficoa.svg" alt="Gráfico de las funciones f(x) y g(x)">'
            
            dataframe_to_txt(df, f'PuntoFijo_{fx}')
            
            if datos["results"]:
                context = {
                    'df': df.to_html(),
                    'plot_html': plot_html,
                    'mensaje': f'La solución es: {root_value}', 'nombre_metodo': "Punto Fijo"
                }
                return render(request, 'one_method.html', context)

        except Exception as e:
            context = {'error_message': f'Hubo un error en el método de Punto Fijo: {str(e)}'}
            return render(request, 'error.html', context)

    return render(request, 'one_method.html')
            

def biseccion(request):
    if request.method == 'POST':
        mensaje = ""
        try:
           
            xi = request.POST['xi']
            xs = request.POST['xs']
            tol = request.POST['tol']
            niter = request.POST['niter']
            funcion = request.POST['funcion']
            tipo_error = request.POST.get('tipo_error')
            
            
            if not all([xi, xs, tol, niter, funcion, tipo_error]):
                xi = 1
                xs = 2
                tol = 0.0001
                niter = 100
                funcion = 'x**2 - 3'
                tipo_error = 'absoluto'
                mensaje += "Se tomaron los valores por defecto, por favor ingrese los valores de todos los campos <br> "
            else:
                xi = float(xi)
                xs = float(xs)
                tol = float(tol)
                niter = int(niter)

            xi_copy = xi
            xs_copy = xs
            hay_solucion = False
            solucion = 0
            fsolucion = 0

           
            try:
                x = sp.symbols('x')
                funcion_expr = sp.sympify(funcion)
            except Exception as e:
                raise ValueError(f"Error al interpretar la función: {e}")

            
            tabla = []
            absoluto = abs(xs - xi) / 2
            itera = 0
            xm = 0
            fi = funcion_expr.subs(x, xi).evalf()
            fs = funcion_expr.subs(x, xs).evalf()

            if fi == 0:
                s = xi
                absoluto = 0
                mensaje += "La solución es: " + str(s)
                hay_solucion = True
                solucion = xi
                fsolucion = fi
            elif fs == 0:
                s = xs
                absoluto = 0
                mensaje += "La solución es: " + str(s)
                hay_solucion = True
                solucion = xs
                fsolucion = fs
            elif fi * fs < 0:
                xm = (xi + xs) / 2
                fxm = funcion_expr.subs(x, xm).evalf()
                relativo = abs((xm - xi) / xm) if xm != 0 else abs((xi) / 0.0000001)

                if fxm == 0:
                    mensaje += "La solución es: " + str(xm)
                    hay_solucion = True
                    solucion = xm
                    fsolucion = fxm
                    tabla.append([xi, xm, xs, fi, fxm, fs, 0, relativo])
                else:
                    tabla.append([xi, xm, xs, fi, fxm, fs, absoluto, relativo])
                    error = absoluto if tipo_error == "absoluto" else relativo

                    while error > tol and itera < niter and fxm != 0:
                        if fi * fxm < 0:
                            xs = xm
                            fs = fxm
                        else:
                            xi = xm
                            fi = fxm

                        xaux = xm
                        xm = (xi + xs) / 2
                        fxm = funcion_expr.subs(x, xm).evalf()
                        absoluto = abs(xaux - xm)
                        relativo = abs((xm - xaux) / xm) if xm != 0 else abs((xaux) / 0.0000001)
                        itera += 1
                        error = absoluto if tipo_error == "absoluto" else relativo
                        tabla.append([xi, xm, xs, fi, fxm, fs, absoluto, relativo])

                    if fxm == 0:
                        s = xm
                        mensaje += "La solución es: " + str(s)
                        hay_solucion = True
                        solucion = xm
                        fsolucion = fxm
                    else:
                        if error < tol:
                            mensaje += "La solución aproximada es: " + str(xm)
                            hay_solucion = True
                            solucion = xm
                            fsolucion = fxm
                        else:
                            mensaje += "Se alcanzó el número máximo de iteraciones"

            else:
                mensaje += "No hay raíz en el intervalo, intente con otro intervalo"

            columnas = ['xi', 'xm', 'xs', 'f(xi)', 'f(xm)', 'f(xs)', 'Err abs ', 'Err Rel']
            df = pd.DataFrame(tabla, columns=columnas)
            df.index = np.arange(1, len(df) + 1)

           
            grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')

            
            if os.path.exists(grafico_path):
                os.remove(grafico_path)

            
            x_vals = np.linspace(xi_copy - 1, xs_copy + 1, 400)  
            y_vals = []
            for val in x_vals:
                y_val = funcion_expr.subs(x, val).evalf()
                if y_val.is_real:
                    y_vals.append(float(y_val))
                else:
                    y_vals.append(np.nan)

            
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label='f(x)', color='blue')
            plt.axhline(0, color='black', linewidth=1)
            plt.axvline(0, color='black', linewidth=1)

          
            if hay_solucion:
                plt.scatter([float(solucion)], [0], color='red', zorder=5, label='Raíz hallada')

            plt.title(f'Gráfico de la función: {funcion}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.grid(True)

          
            plt.savefig(grafico_path, format='svg')
            plt.close() 

            dataframe_to_txt(df, "biseccion")
            context = {'df': df.to_html(), 'mensaje': mark_safe(mensaje), 'nombre_metodo': "Bisección"}
            return render(request, 'one_method.html', context)

        except ValueError as ve:
            context = {'error_message': f"Error en los datos ingresados: {ve}"}
            return render(request, 'error.html', context)

        except Exception as e:
            context = {'error_message': f"Ocurrió un error: {e}"}
            return render(request, 'error.html', context)
    
    return render(request, 'one_method.html')

def secante(request):
    if request.method == 'POST':
        mensaje = ""
        try:
            
            x0 = request.POST.get('x0')
            x1 = request.POST.get('x1')
            tol = request.POST.get('tol')
            niter = request.POST.get('niter')
            funcion = request.POST.get('funcion')
            tipo_error = request.POST.get('tipo_error')

           
            if not x0 or not x1 or not tol or not niter or not funcion or not tipo_error:
                x0 = 1
                x1 = 2
                tol = 0.0001
                niter = 100
                funcion = 'x**2 - 3'
                tipo_error = 'absoluto'
                mensaje += "Se tomó el caso por defecto, por favor ingrese los valores de todos los campos <br> "

           
            try:
                x0 = float(x0)
                x1 = float(x1)
                tol = float(tol)
                niter = int(niter)
            except ValueError:
                raise ValueError("Los valores ingresados no son válidos. Asegúrese de que x0, x1, tol y niter sean números válidos.")

           
            x = symbols('x')
            funcion_expr = parse_expr(funcion, local_dict={'x': x})
            fx = lambda x: funcion_expr.subs(x, x)

           
            tabla = []
            itera = 1
            x2 = 0
            f0 = funcion_expr.subs(x, x0).evalf()
            f1 = funcion_expr.subs(x, x1).evalf()
            hay_solucion = False
            solucion = 0
            fsolucion = 0

            
            if f0 == 0:
                s = x0
                mensaje += "La solución es: " + str(s)
                hay_solucion = True
                solucion = x0
                fsolucion = f0
            elif f1 == 0:
                s = x1
                mensaje += "La solución es: " + str(s)
                hay_solucion = True
                solucion = x1
                fsolucion = f1
            else:
                x2 = x1 - ((f1 * (x0 - x1)) / (f0 - f1))
                f2 = funcion_expr.subs(x, x2).evalf()
                if x2 != 0:
                    relativo = abs((x2 - x1) / x2)
                else:
                    relativo = abs(x1) / 0.0000001
                absoluto = abs(x2 - x1)
                if f2 == 0:
                    mensaje += "La solución es: " + str(x2)
                    hay_solucion = True
                    solucion = x2
                    fsolucion = f2
                    tabla.append([itera, x0, x1, x2, f0, f1, f2, 0, "-"])
                else:
                    tabla.append([itera, x0, x1, x2, f0, f1, f2, abs(x2 - x1), "-"])
                    x0 = x1
                    f0 = f1
                    x1 = x2
                    f1 = f2
                    itera += 1
                    if tipo_error == "absoluto":
                        error = absoluto
                    else:
                        error = relativo
                    while itera < niter and tol < error and f1 != 0 and f0 != 0:
                        x2 = x1 - ((f1 * (x0 - x1)) / (f0 - f1))
                        f2 = funcion_expr.subs(x, x2).evalf()
                        if x2 != 0:
                            relativo = abs((x2 - x1) / x2) * 100
                        else:
                            relativo = abs(x1) / 0.0000001 * 100
                        if f2 == 0:
                            mensaje += "La solución es: " + str(x2) + "\n En la iteración " + str(itera)
                            hay_solucion = True
                            solucion = x2
                            fsolucion = f2
                            tabla.append([itera, x0, x1, x2, f0, f1, f2, 0, relativo])
                            break
                        absoluto = abs(x2 - x1)
                        tabla.append([itera, x0, x1, x2, f0, f1, f2, absoluto, relativo])
                        x0 = x1
                        f0 = f1
                        x1 = x2
                        f1 = f2
                        itera += 1
                        if tipo_error == "absoluto":
                            error = absoluto
                        else:
                            error = relativo
                    if f1 == 0:
                        mensaje += "La solución es: " + str(x1)
                        hay_solucion = True
                        solucion = x1
                        fsolucion = f1
                    if error < tol:
                        mensaje += "La solución aproximada que cumple la tolerancia es: " + str(x2) + " En la iteración " + str(itera) + " = x" + str(itera)
                        hay_solucion = True
                        solucion = x2
                        fsolucion = f2
                    elif itera == niter:
                        mensaje += "Se ha alcanzado el número máximo de iteraciones"

           
            columnas = ['i', 'xi-1', 'xi', 'xi+1', 'f(xi-1)', 'f(xi)', 'f(xi+1)', 'Err abs', 'Err Rel']
            df = pd.DataFrame(tabla, columns=columnas)

          
            fig, ax = plt.subplots()
            x_vals = np.linspace(x0 - 1, x1 + 1, 400)
            y_vals = [funcion_expr.subs(x, val).evalf() for val in x_vals]
            ax.plot(x_vals, y_vals, label='Función', color='blue')

           
            ax.scatter([x0, x1, solucion], [f0, f1, fsolucion], color='red', label='Raíces')

           
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.legend()

            ax.grid(True)

           
            plot_path = os.path.join('Metodos', 'static', 'graficoa.svg')
            plt.savefig(plot_path, format='svg')

           
            context = {
                'df': df.to_html(),
                'plot_path': plot_path,
                'mensaje': mark_safe(mensaje),
                'nombre_metodo': "Secante"
            }

           
            return render(request, 'one_method.html', context)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return render(request, 'error.html', {'error_message': error_message})

def newton(x0, Tol, Niter, fx, df, tipo_error):

    output = {
        "columns": ["N", "xi", "F(xi)", "E"],
        "results": [],
        "errors": [],
        "root": "0"
    }

    
    datos = list()
    x = sp.Symbol('x')
    Fun = sympify(fx)
    DerF = sympify(df)

    xn = []
    derf = []
    xi = x0 
    f = Fun.evalf(subs={x: x0})
    derivada = DerF.evalf(subs={x: x0}) 
    c = 0
    Error = 100
    xn.append(xi)

    try:
        datos.append([c, '{:^15.7f}'.format(x0), '{:^15.7f}'.format(f)])

        while Error > Tol and f != 0 and derivada != 0 and c < Niter:
            xi = xi - f / derivada
            derivada = DerF.evalf(subs={x: xi})
            f = Fun.evalf(subs={x: xi})
            xn.append(xi)
            c = c + 1
            if tipo_error == "absoluto":
                Error = abs(xn[c] - xn[c-1])
            else:
                Error = abs(xn[c] - xn[c-1]) / xn[c]
            derf.append(derivada)
            datos.append([c, '{:^15.7f}'.format(float(xi)), '{:^15.7E}'.format(float(f)), '{:^15.7E}'.format(float(Error))])

    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append("Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xi
    return output

def newtonView(request):
    context = {}
    datos = {}

    if request.method == 'POST':
        try:
            fx = request.POST["funcion"]
            derf = request.POST["derivada"]
            x0 = request.POST["punto_inicial"]
            X0 = float(x0)
            tol = request.POST["tolerancia"]
            Tol = float(tol)
            niter = request.POST["iteraciones"]
            Niter = int(niter)
            tipo_error = request.POST.get('tipo_error')

            datos = newton(X0, Tol, Niter, fx, derf, tipo_error)
        
            if "results" in datos:
                df = pd.DataFrame(datos["results"], columns=datos["columns"])
                
                
                x = sp.symbols('x')
                funcion_expr = sp.sympify(fx)

                xi_copy = X0
                xs_copy = X0  

                intervalo_x = np.arange(xi_copy - 10, xs_copy + 10, 0.1)
                fx_func = sp.lambdify(x, funcion_expr, 'numpy')
                intervalo_y = fx_func(intervalo_x)

               
                plt.figure(figsize=(10, 6))
                plt.plot(intervalo_x, intervalo_y, label='f(x)', color='blue')
                plt.axhline(0, color='black', linewidth=1)
                plt.axvline(0, color='black', linewidth=1)

                
                if datos.get("root") is not None:
                    plt.scatter([float(datos["root"])], [0], color='red', zorder=5, label='Raíz hallada')

                plt.title(f'Función: {fx} en intervalo [{xi_copy - 10}, {xs_copy + 10}]')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                plt.grid(True)

                0
                grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')

               
                if os.path.exists(grafico_path):
                    os.remove(grafico_path)

                
                plt.savefig(grafico_path, format='svg')
                plt.close() 

           
            dataframe_to_txt(df, f'Newton_txt {fx}')
            plot_html = f'<img src="/static/graficoa.svg" alt="Gráfico Newton">'

            context = {'df': df.to_html(), 'plot_html': plot_html, 'mensaje': f'La solución es: {datos["root"]}','nombre_metodo': "Newton" }

            return render(request, 'one_method.html', context)
        
        except Exception as e:
            context = {'error_message': f'Hubo un error en el método de Newton: {str(e)}'}
            return render(request, 'error.html', context)

    return render(request, 'one_method.html')

def raicesMultiples(fx, x0, tol, niter, tipo_error):
    output = {
        "columns": ["iter", "xi", "f(xi)", "E"],
        "iterations": niter,
        "errors": [],
        "results": [],
        "root": "0"
    }

    
    results = []
    x = sp.Symbol('x')
    ex = sp.sympify(fx)
    d_ex = sp.diff(ex, x)  
    d2_ex = sp.diff(d_ex, x)  

    ex_2 = ex.subs(x, x0).evalf()
    d_ex2 = d_ex.subs(x, x0).evalf()
    d2_ex2 = d2_ex.subs(x, x0).evalf()

    if d_ex2 == 0 or d2_ex2 == 0:
        output["errors"].append("Error: La primera o la segunda derivada son cero.")
        return output

    i = 0
    error = 1.0000
    results.append([i, '{:^15.7E}'.format(x0), '{:^15.7E}'.format(ex_2)])

    while (error > tol) and (i < niter):
        if i == 0:
            ex_2 = ex.subs(x, x0).evalf()
        else:
            d_ex2 = d_ex.subs(x, x0).evalf()
            d2_ex2 = d2_ex.subs(x, x0).evalf()

            if d_ex2 == 0 or d2_ex2 == 0:
                output["errors"].append("Error: La primera o la segunda derivada son cero.")
                return output

            xA = x0 - (ex_2 * d_ex2) / ((d_ex2) ** 2 - ex_2 * d2_ex2)
            ex_A = ex.subs(x, xA).evalf()

            if tipo_error == "absoluto":
                error = abs(xA - x0)
            else:
                error = abs(xA - x0) / abs(xA)
            error = error.evalf()

            ex_2 = ex_A
            x0 = xA

            results.append([i, '{:^15.7E}'.format(float(xA)), '{:^15.7E}'.format(float(ex_2)), '{:^15.7E}'.format(float(error))])
        i += 1

    output["results"] = results
    output["root"] = x0
    return output


def raicesMultiplesView(request):
    context = {}
    datos = {}

    if request.method == 'POST':
        try:
            Fx = request.POST["funcion"]
            X0 = float(request.POST["punto_inicial"])
            Niter = int(request.POST["iteraciones"])
            Tol = float(request.POST["tolerancia"])
            tipo_error = request.POST.get('tipo_error')

            datos = raicesMultiples(Fx, X0, Tol, Niter, tipo_error)

            
            if "results" in datos and datos["results"]:
                df = pd.DataFrame(datos["results"], columns=datos["columns"])

                
                x = sp.symbols('x')
                funcion_expr = sp.sympify(Fx)
                intervalo_x = np.arange(X0 - 10, X0 + 10, 0.1)
                fx_func = sp.lambdify(x, funcion_expr, 'numpy')
                intervalo_y = fx_func(intervalo_x)

                
                plt.figure(figsize=(10, 6))
                plt.plot(intervalo_x, intervalo_y, label='f(x)', color='blue')
                plt.axhline(0, color='black', linewidth=1)
                plt.axvline(0, color='black', linewidth=1)

                if datos.get("root") is not None:
                    plt.scatter([float(datos["root"])], [0], color='red', zorder=5, label='Raíz hallada')

                plt.title(f'Función: {Fx} en intervalo [{X0 - 10}, {X0 + 10}]')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                plt.grid(True)

               
                grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')

                
                if not os.path.exists(os.path.dirname(grafico_path)):
                    os.makedirs(os.path.dirname(grafico_path))

                
                if os.path.exists(grafico_path):
                    os.remove(grafico_path)

                
                plt.savefig(grafico_path, format='svg')
                plt.close()

             
                if os.path.exists(grafico_path):
                    print(f"Gráfico guardado en: {grafico_path}")
                else:
                    print("Hubo un problema al guardar el gráfico.")

            context = {'df': df.to_html(), 'mensaje': f'La solución es: {datos["root"]}', 'grafico_path': '/static/graficoa.svg', 'nombre_metodo': "Raices Multiples"}

            return render(request, 'one_method.html', context)

        except Exception as e:
            context = {'error_message': f'Error: {str(e)}'}
            return render(request, 'error.html', context)

    return render(request, 'one_method.html')
    

def iterativos(request):
    if request.method == 'POST':
        tamaño=request.POST['numero']
        metodo=request.POST['metodo_iterativo']
        tol=request.POST['tol']
        niter=request.POST['niter']
        if metodo=="sor":
            w=request.POST['w']
        try:
            matriz=[]
            tamaño=int(tamaño)
            tol=float(tol)
            niter=int(niter)
            if metodo=="sor":
                w=float(w)
            i=0
            for i in range(tamaño):
                row = []
                j=0
                for j in range(tamaño):
                    val = request.POST.get(f'matrix_cell_{i}_{j}')
                    row.append(int(val) if val else 0)
                matriz.append(row)
            vectorx=[]
            for i in range(tamaño):
                val = request.POST.get(f'vx_cell_{i}')
                vectorx.append(int(val) if val else 0)
            vectorb=[]
            for i in range(tamaño):
                val = request.POST.get(f'vb_cell_{i}')
                vectorb.append(int(val) if val else 0)
            nmatriz=np.array(matriz)
            nvectorx = np.array(vectorx).reshape(-1, 1)
            nvectorb = np.array(vectorb).reshape(-1, 1)
            if metodo=="jacobi":
                context=metodo_jacobi(nmatriz,nvectorx,nvectorb,tol,niter)
            elif metodo=="gauss_seidel":
                context=metodo_gauss_seidel(nmatriz,nvectorx,nvectorb,tol,niter)
            elif metodo=="sor":
                context=metodo_sor(nmatriz,nvectorx,nvectorb,tol,niter,w)

            dataframe_to_txt(pd.DataFrame(nmatriz),metodo+"_matrizA")
            dataframe_to_txt(pd.DataFrame(nvectorb),metodo+"_vectorb")
            dataframe_to_txt(context['tabla'],metodo+"_tabla")
            return render(request,'resultado_iterativo.html',context)

        except:
            context={'mensaje':'No se pudo realizar la operación'}
            return render(request,'resultado_iterativo.html',context)


def interpolacion(request):
    datos = request.POST
    mensaje = "" 
    
    if request.method == 'POST':
        
        try:
            metodo_interpolacion = request.POST.get('metodo_interpolacion')
            x_values = request.POST.getlist('x[]')
            y_values = request.POST.getlist('y[]')

           
            x_floats = [float(x) for x in x_values]
            y_floats = [float(y) for y in y_values]

            xi = x_floats
            fi = y_floats

            n = len(xi)
            if n<=1:
                context = {'error_message': f'Se deben ingresar 2 puntos o más'}
                return render(request, 'error.html', context)
            for i in range(n-1):
                if xi[i] >= xi[i+1]:
                    context = {'error_message': f'Los valores de x deben estar ordenados de menor a mayor'}
                    return render(request, 'error.html', context)

            if metodo_interpolacion == 'lagrange':
                try:
                    xi = x_floats
                    fi = y_floats

                   
                    
                    x = sp.Symbol('x')
                    polinomio = 0
                    divisorL = np.zeros(n, dtype=float)
                    iteraciones = []

                    for i in range(n):
                        numerador = 1
                        denominador = 1
                        numerador_str = ""
                        denominador_str = ""

                        for j in range(n):
                            if j != i:
                                numerador *= (x - xi[j])
                                denominador *= (xi[i] - xi[j])
                                numerador_str += f"(x - {xi[j]})"
                                if denominador_str:
                                    denominador_str += "*"
                                denominador_str += f"({xi[i]} - {xi[j]})"

                        terminoLi = numerador / denominador
                        polinomio += terminoLi * fi[i]
                        divisorL[i] = denominador

                        Li_str = f"{numerador_str} / {denominador_str}"
                        
                        
                        iteraciones.append({
                            'i': i,
                            'L_i': Li_str
                        })

                  
                    df_iteraciones = pd.DataFrame(iteraciones)

                    polisimple = polinomio.expand()

                    
                    a = np.min(xi)
                    b = np.max(xi)

                    
                    px = sp.lambdify(x, polisimple, 'numpy')

                    
                    intervalo_x_completo = np.arange(a, b, 0.1)
                    intervalo_y_completo = px(intervalo_x_completo)

                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(intervalo_x_completo, intervalo_y_completo, label='f(x)', color='blue')
                    plt.axhline(0, color='black', linewidth=1)
                    plt.axvline(0, color='black', linewidth=1)

                    if datos.get("root") is not None:
                        plt.scatter([float(datos["root"])], [0], color='red', zorder=5, label='Raíz hallada')

                    plt.title(f'Función: {polisimple} en intervalo [{a - 10}, {b + 10}]')
                    plt.xlabel('x')
                    plt.ylabel('f(x)')
                    plt.legend()
                    plt.grid(True)


      
                    

                    grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')


        
                    if not os.path.exists(os.path.dirname(grafico_path)):
                        os.makedirs(os.path.dirname(grafico_path))


       
                    if os.path.exists(grafico_path):
                        os.remove(grafico_path)


        
                    plt.savefig(grafico_path, format='svg')
                    plt.close()


        
                    polisimple_latex = sp.latex(polisimple)
                    mensaje = 'Polinomio de Lagrange:'
                    plot_html = None


                    context = {
                        'polinomio': polisimple_latex,
                        'df': df_iteraciones.to_html(classes='table table-striped', index=False),
                        'mensaje': mensaje,
                        'plot_html': plot_html,'nombre_metodo': "Lagrange"
                    }

                    dataframe_to_txt(df_iteraciones, "lagrange_iteraciones")
                    text_to_txt(str(polisimple),"lagrange_funcion")
                    return render(request, 'one_method.html', context)
                except Exception as e:
                    context = {'error_message': f'Hubo un error con lagrange en: {str(e)}'}
                    return render(request, 'error.html', context)

            elif metodo_interpolacion == 'newton':
                try:
                    xi = x_floats
                    fi = y_floats

                    
                    X = np.array(x_floats)
                    Y = np.array(y_floats)
                    n = len(X)

                  
                    D = np.zeros((n, n))
                    D[:, 0] = Y.T
                    for i in range(1, n):
                        for j in range(n - i):
                            D[j, i] = (D[j + 1, i - 1] - D[j, i - 1]) / (X[j + i] - X[j])
                    
                    Coef = D[0, :] 

                    
                    diff_table_df = pd.DataFrame(D)
                    diff_table_df_to_html = diff_table_df.to_html(classes='table table-striped', index=False)

                    
                    x = sp.Symbol('x')
                    newton_poly = Coef[0]
                    product_term = 1
                    for i in range(1, n):
                        product_term *= (x - X[i-1])
                        newton_poly += Coef[i] * product_term

                    
                    newton_poly_df = pd.DataFrame({'Polinomio': [str(newton_poly)]})
                    newton_poly_df_to_html = newton_poly_df.to_html(classes='table table-striped', index=False)

                   
                    x_vals = np.linspace(min(X), max(X), 100)
                    y_vals = [float(newton_poly.evalf(subs={x: val})) for val in x_vals]

                    plt.figure(figsize=(10, 6))

                    
                    plt.scatter(X, Y, color='red', label='Puntos originales')

                    
                    plt.plot(x_vals, y_vals, label='Polinomio de Newton', color='blue')

                    plt.title('Interpolación de Newton con Diferencias Divididas')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.grid(True)
                    plt.legend()

                   
                    grafico_path =os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')
                    if not os.path.exists(os.path.dirname(grafico_path)):
                        os.makedirs(os.path.dirname(grafico_path))

                    if os.path.exists(grafico_path):
                        os.remove(grafico_path)

                    plt.savefig(grafico_path, format='svg')
                    plt.close()

                   
                    mensaje = 'Polinomio de Newton:'
                    plot_html = None

                    context = {
                        'df': diff_table_df_to_html,
                        'df2': newton_poly_df_to_html,
                        'nombre_metodo': 'Newton Diferencias Divididas', 
                        'plot_html': plot_html
                    }

                    dataframe_to_txt(diff_table_df, 'Newton_diff_table')
                    dataframe_to_txt(newton_poly_df, 'Newton_poly')
                 
                    return render(request, 'one_method.html', context)
                except Exception as e:
                    context = {'error_message': f'Hubo un error con Newton Diferencias Divididas en: {str(e)}'}
                    return render(request, 'error.html', context)

            elif metodo_interpolacion == 'vandermonde':
                try:
                    x = sp.Symbol('x')
                    matriz_A = []
                    for i in range(n):
                        grado = n - 1
                        fila = []
                        for j in range(grado, -1, -1):
                            fila.append(xi[i] ** grado)
                            grado -= 1
                        matriz_A.append(fila)
                    vector_b = fi
                    nmatriz_A = np.array(matriz_A)
                    nvectorb = np.array(vector_b).reshape(-1, 1)
                    nvectora = np.linalg.inv(nmatriz_A) @ nvectorb
                    pol = ""
                    grado = n - 1
                    for i in range(n):
                        pol +=str(nvectora[i][0])
                        if i < n - 1:
                            if grado > 0:
                                pol += "*x**" + str(grado)
                            if (nvectora[i + 1][0] >= 0):
                                pol += "+"
                        grado -= 1


                    mA_html = pd.DataFrame(nmatriz_A).to_html()
                    vb_html = pd.DataFrame(nvectorb).to_html()
                    va_html = pd.DataFrame(nvectora).to_html()

                    x_vals = np.linspace(min(xi), max(xi), 100)
                    y_vals = [sum([nvectora[i][0] * (val ** (n - 1 - i)) for i in range(n)]) for val in x_vals]

        
                    plt.figure(figsize=(10, 6))

        
                    plt.scatter(xi, fi, color='red', label='Puntos originales')

        
                    plt.plot(x_vals, y_vals, label='Polinomio de Vandermonde', color='blue')

        
                    plt.title('Interpolación de Vandermonde')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.grid(True)
                    plt.legend()

        
                    grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')

        
                    if not os.path.exists(os.path.dirname(grafico_path)):
                        os.makedirs(os.path.dirname(grafico_path))

        
                    if os.path.exists(grafico_path):
                        os.remove(grafico_path)

        
                    plt.savefig(grafico_path, format='svg')
                    plt.close()
                    plot_html = None
                    mensaje = "Se logro dar con una solucion"
                    context = {
                        'vandermonde': True,
                        'ma': mA_html,
                        'vb': vb_html,
                        'va': va_html,
                        'polinomio': pol,
                        'nombre_metodo': "Vandermonde",
                        'mensaje': mensaje,
                        'plot_html': plot_html
                    }
                    return render(request, 'one_method.html', context)
                except Exception as e:
                    context = {'error_message': f'Hubo un error con vandermonde: {str(e)}'}
                    return render(request, 'error.html', context)
                

            elif metodo_interpolacion == 'splinel':

                xi = x_floats
                fi = y_floats    

               
                X = np.array(x_floats)
                Y = np.array(y_floats)
                n = len(X)
                
                
                x = sp.Symbol('x')
                px_tabla = []
                coef_list = []

                
                for i in range(1, n):
                    
                    numerador = Y[i] - Y[i-1]
                    denominador = X[i] - X[i-1]
                    m = numerador / denominador
                    
                    
                    coef_list.append([float(m), float(Y[i-1] - m * X[i-1])])
                    
                    
                    pxtramo = Y[i-1] + m * (x - X[i-1])
                    px_tabla.append(pxtramo)
                
                
                coef_df = pd.DataFrame(coef_list, columns=['Pendiente (m)', 'Intersección (b)'])
                coef_df_to_html = coef_df.to_html(classes='table table-striped', index=False)

                
                func_df = pd.DataFrame({'Función': [str(func) for func in px_tabla]})
                func_df_to_html = func_df.to_html(classes='table table-striped', index=False)

                
                funciones_evaluadas = []
                for i, func in enumerate(px_tabla):
                    x_vals = np.linspace(X[i], X[i+1], 100)
                    y_vals = [float(func.evalf(subs={x: val})) for val in x_vals]
                    funciones_evaluadas.append((x_vals, y_vals))
                
                plt.figure(figsize=(10, 6))

    
                plt.scatter(X, Y, color='red', label='Puntos originales')

   
                for i, (x_vals, y_vals) in enumerate(funciones_evaluadas):
                    plt.plot(x_vals, y_vals, label=f'Tramo {i+1}', color='blue')

    
                plt.title('Spline Lineal por Tramos')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
                plt.legend()

    
                grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')

    
                if not os.path.exists(os.path.dirname(grafico_path)):
                    os.makedirs(os.path.dirname(grafico_path))

    
                if os.path.exists(grafico_path):
                    os.remove(grafico_path)

   
                plt.savefig(grafico_path, format='svg')
                plt.close()
                plot_html= None

                context = {
                    'df': coef_df_to_html,
                    'df2': func_df_to_html,
                    'nombre_metodo': 'Spline Lineal', 
                    'plot_html': plot_html
                }

                dataframe_to_txt(coef_df, 'Spline_coef')
                dataframe_to_txt(func_df, 'Spline_func')
                
            elif metodo_interpolacion=="splinecu":
                x=xi
                y=fi
                A = np.zeros((4*(n-1), 4*(n-1)))
                b = np.zeros(4*(n-1))
                cua = np.square(x)
                cub = np.power(x, 3)
                c = 0
                h = 0
                d=3
                
                for i in range(n - 1):
                    A[i, c] = cub[i]
                    A[i, c + 1] = cua[i]
                    A[i, c + 2] = x[i]
                    A[i, c + 3] = 1
                    b[i] = y[i]
                    c += 4
                    h += 1
                
                c = 0
                for i in range(1, n):
                    A[h, c] = cub[i]
                    A[h, c + 1] = cua[i]
                    A[h, c + 2] = x[i]
                    A[h, c + 3] = 1
                    b[h] = y[i]
                    c += 4
                    h += 1
                
                c = 0
                for i in range(1, n - 1):
                    A[h, c] = 3 * cua[i]
                    A[h, c + 1] = 2 * x[i]
                    A[h, c + 2] = 1
                    A[h, c + 4] = -3 * cua[i]
                    A[h, c + 5] = -2 * x[i]
                    A[h, c + 6] = -1
                    b[h] = 0
                    c += 4
                    h += 1
                
                c = 0
                for i in range(1, n - 1):
                    A[h, c] = 6 * x[i]
                    A[h, c + 1] = 2
                    A[h, c + 4] = -6 * x[i]
                    A[h, c + 5] = -2
                    b[h] = 0
                    c += 4
                    h += 1
                
                A[h, 0] = 6 * x[0]
                A[h, 1] = 2
                b[h] = 0
                h += 1
                A[h, c] = 6 * x[-1]
                A[h, c + 1] = 2
                b[h] = 0
                val = np.linalg.inv(A).dot(b)
                Tabla = val.reshape((n - 1, d + 1))
                fxs=[]
                for i in range(n-1):
                    fx=""
                    for j in range(4):
                        fx+=str(Tabla[i,j])
                        if j<3:
                            fx+="*x"
                            if j<2:
                                fx+="**"+str(3-j)
                            if(Tabla[i,j+1]>=0):
                                fx+="+"
                    fxs.append(fx)
                tabla_df=pd.DataFrame(fxs)
                tabla_html=tabla_df.to_html(classes='table table-striped', index=False)

                plt.figure(figsize=(10, 6))

        
                for i in range(n - 1):
                    print(f"Graficando tramo {i + 1} con la función: {fxs[i]}") 
    
    
                    x_vals = np.linspace(xi[i], xi[i + 1], 100)
    
                    try:
       
                        y_vals = np.array([eval(fxs[i].replace('x', f'({x_val})')) for x_val in x_vals])
                    except Exception as e:
                        print(f"Error al evaluar la función para el tramo {i + 1}: {e}")
                        y_vals = np.zeros(100) 
    
   
                    plt.plot(x_vals, y_vals, label=f'Tramo {i + 1}')

       
                plt.scatter(xi, fi, color='red', label='Puntos originales')

        
                plt.title('Spline Cúbico por Tramos')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
                plt.legend()

       
                grafico_path = os.path.join(settings.BASE_DIR, 'Metodos/static/graficoa.svg')

        
                if not os.path.exists(os.path.dirname(grafico_path)):
                    os.makedirs(os.path.dirname(grafico_path))

       
                if os.path.exists(grafico_path):
                    os.remove(grafico_path)

        
                plt.savefig(grafico_path, format='svg')
                plt.close()
                plot_html = None


                context={
                    'nombre_metodo': "Spline cubico",
                    'df':tabla_html,
                    'coef': pd.DataFrame(A).to_html(classes='table table-striped', index=False),
                    'spline':True,
                    'plot_html':plot_html
                }
                dataframe_to_txt(pd.DataFrame(A), "spline_cubico_coef")
                dataframe_to_txt(tabla_df, "spline_cubico_funciones")
                
                

        
        except Exception as e:
            context = {'error_message': f'Ocurrió un error en la obtención de los datos, llene todos los campos bien, numeros enteros o decimales'}
            return render(request, 'error.html', context)

    return render(request, 'one_method.html', context)


