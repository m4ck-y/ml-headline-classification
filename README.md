## Estructura del Proyecto

```
├── app/
│   └── main.py          # Aplicación principal
├── data/
│   └── data.json        # Dataset
└── requirements.txt     # Dependencias
```

## Requisitos

- Python 3.10.12(usado en el desarrollo)o superior
- pip (gestor de paquetes de Python)

## Instalación

1. **Crear entorno virtual:**
   ```bash
   python3 -m venv .venv
   ```

2. **Activar el entorno virtual:**
   
   En Windows:
   ```bash
   .venv\Scripts\activate
   ```
   
   En macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para ejecutar la aplicación:

```bash
python -m app.main 
```

## Desarrollo

- **Lenguaje**: Python 3.10.12
- **Sistema Operativo**: WSL2 Ubuntu 22.04.5 LTS x86_64
- **IDE**: VS Code

#### Dependencias

- **pandas**: Lectura, manipulación y análisis de datos
- **ipykernel**: Necesario para ejecutar Jupyter Notebooks. Se utiliza durante el Análisis Exploratorio de Datos (EDA) para pruebas interactivas, y una vez validadas las transformaciones, estas se trasladan a process.py para su procesamiento final.
- **plotly** :Librería de visualización interactiva de gráficos, utilizada para crear gráficos dinámicos y visualizaciones atractivas.
- **nbformat**: Librería necesaria para renderizar correctamente los gráficos interactivos de Plotly dentro de Jupyter Notebooks.
- **scikit-learn**: Toolkit principal para construir, entrenar y evaluar modelos.
- **imblearn**:Conjunto de herramientas especializado para manejar desbalanceo de clases.
- **nltk**: Biblioteca fundamental para el procesamiento de lenguaje natural (NLP). Se utiliza para tareas como tokenización, limpieza y normalización de texto antes del modelado.

Comando para actualizar el archivo de dependencias
```bash
pip freeze > requirements.txt
```