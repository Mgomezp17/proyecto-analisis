import os
import numpy as np
from tabulate import tabulate

def dataframe_to_txt(dataframe, filename):
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    dataframe = dataframe.map(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
    try:
        with open(f'resultados/{filename}.txt', 'w') as f:
            # Use tabulate to format the dataframe as a table
            table = tabulate(dataframe, headers='keys', tablefmt='pretty')
            f.write(table)
    except Exception as e:
        print("An error occurred while transcribing the dataframe:", str(e))
    
def plot_to_png(fig, filename):
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    try:
        fig.write_image(f'resultados/{filename}.png')
    except Exception as e:
        print("An error occurred while saving the plot as PNG:", str(e))
def text_to_txt(text,filename):
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    try:
        with open(f'resultados/{filename}.txt', 'w') as f:
            f.write(text)
    except Exception as e:
        print("An error occurred while saving the text as TXT:", str(e))
