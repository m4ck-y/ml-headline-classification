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

### Entrenar el modelo

Para entrenar el modelo (solo necesario la primera vez):

```bash
python -m app.main 
```

### Ejecutar la API REST

Para ejecutar la API de clasificación:

```bash
python run_api.py
```

La API estará disponible en:
- **URL base**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc

### Probar la API

Para probar la API con ejemplos:

```bash
python test_api.py
```

### Endpoints disponibles

- `GET /` - Estado general de la API
- `GET /health` - Verificación de salud de los modelos
- `GET /categories` - Lista todas las categorías disponibles
- `POST /predict` - Clasifica un headline

#### Ejemplo de uso con curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"headline": "Breaking news: Stock market hits record high"}'
```

#### Respuesta esperada:

```json
{
  "headline": "Breaking news: Stock market hits record high",
  "predicted_category": "BUSINESS",
  "confidence": 0.85,
  "all_probabilities": {
    "BUSINESS": 0.85,
    "POLITICS": 0.10,
    "TECHNOLOGY": 0.05
  }
}
```

## Desarrollo

- **Lenguaje**: Python 3.10.12
- **Sistema Operativo**: WSL2 Ubuntu 22.04.5 LTS x86_64
- **IDE**: VS Code

#### Dependencias

- **pandas**: Lectura, manipulación y análisis de datos
- **ipykernel**: Necesario para ejecutar Jupyter Notebooks. Se utiliza durante el Análisis Exploratorio de Datos (EDA) para pruebas interactivas, y una vez validadas las transformaciones, estas se trasladan a process.py para su procesamiento final.
- **matplotlib**: Librería estándar para la visualización de datos en gráficos estáticos.
- **seaborn**: Biblioteca de visualización basada en Matplotlib
- **nbformat**: Librería necesaria para renderizar correctamente los gráficos interactivos de Plotly dentro de Jupyter Notebooks.
- **scikit-learn**: Toolkit principal para construir, entrenar y evaluar modelos.
- **imblearn**:Conjunto de herramientas especializado para manejar desbalanceo de clases.
- **nltk**: Biblioteca fundamental para el procesamiento de lenguaje natural (NLP). Se utiliza para tareas como tokenización, limpieza y normalización de texto antes del modelado.

Comando para actualizar el archivo de dependencias
```bash
pip freeze > requirements.txt
```