El proceso de preprocesamiento se organiza en dos archivos principales:

1. **`eda.ipynb` (Jupyter Notebook)**:
    - Se utiliza para el **Análisis Exploratorio de Datos (EDA)**.
    - Permite explorar los datos de manera interactiva, facilitando la visualización de su distribución, la identificación de patrones y la prueba de distintos enfoques de limpieza y transformación.
2. **`process.py` (Python Script)**:
    - Aquí se implementa el **preprocesamiento definitivo** y estructurado.
    - Una vez validados los métodos adecuados de limpieza y transformación en el notebook, estos se trasladan al archivo `process.py`, donde se aplican de forma automática y repetible.
    - Este script está diseñado para integrarse al flujo principal gestionado por **`main.py`**, asegurando que el preprocesamiento sea parte de un pipeline automatizado.