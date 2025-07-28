# 📰 Clasificador de Headlines con FastAPI

API REST moderna para clasificar headlines de noticias en diferentes categorías usando Machine Learning y procesamiento de lenguaje natural.

## 🚀 Características

- **API REST moderna** construida con FastAPI
- **Documentación automática** con Swagger UI
- **Clasificación de texto** usando Random Forest y TF-IDF
- **Balanceo de clases** con técnicas de oversampling
- **Procesamiento de texto** con NLTK
- **Validación de datos** con Pydantic
- **Gestión de modelos** con registro centralizado

## 📁 Estructura del Proyecto

```
├── app/
│   ├── main.py                    # 🎯 Punto de entrada de la API FastAPI
│   ├── api/
│   │   └── routes.py              # 🛣️  Rutas de la API (/predict, /categories)
│   ├── config/
│   │   └── env.py                 # ⚙️  Configuración de paths y variables
│   ├── core/
│   │   ├── lifecycle.py           # 🔄 Gestión del ciclo de vida de la app
│   │   ├── loader.py              # 📥 Carga de modelos entrenados
│   │   └── registry.py            # 📋 Registro global de modelos
│   ├── models/
│   │   ├── models.py              # 🤖 Entrenamiento y evaluación de modelos
│   │   └── utils.py               # 🔧 Utilidades y decoradores
│   ├── preprocessing/
│   │   ├── process.py             # 📊 Procesamiento principal del dataset
│   │   ├── process_text.py        # 📝 Tokenización y vectorización
│   │   ├── process_dataframe.py   # 🗂️  Manipulación de DataFrames
│   │   ├── process_category.py    # 🏷️  Procesamiento de categorías
│   │   ├── prepare.py             # 🛠️  Preparación de datos
│   │   └── eda.ipynb              # 📈 Análisis exploratorio de datos
│   └── schemas/
│       └── __init__.py            # 📋 Modelos Pydantic (request/response)
├── data/
│   ├── data.json                  # 📄 Dataset original
│   ├── data_processed.csv         # ✅ Dataset procesado
│   └── tmp_minimized.csv          # 🗜️  Dataset temporal minimizado
├── models/
│   ├── model.pkl                  # 🎯 Modelo Random Forest entrenado
│   ├── vectorizer.pkl             # 🔤 Vectorizador TF-IDF
│   └── label_encoder.pkl          # 🏷️  Codificador de etiquetas
└── requirements.txt               # 📦 Dependencias del proyecto
```

## 🛠️ Tecnologías Utilizadas

### Framework Web
- **FastAPI** `0.115.6` - Framework web moderno y rápido para construir APIs
- **Uvicorn** `0.34.0` - Servidor ASGI de alto rendimiento
- **Pydantic** - Validación de datos y serialización (incluido en FastAPI)

### Machine Learning
- **scikit-learn** `1.6.1` - Algoritmos de ML y métricas de evaluación
- **joblib** - Serialización eficiente de modelos

### Procesamiento de Datos
- **pandas** `2.3.1` - Manipulación y análisis de datos
- **nltk** `3.9.1` - Procesamiento de lenguaje natural
- **numpy** `2.2.6` - Operaciones numéricas

### Visualización y Análisis
- **matplotlib** `3.10.3` - Gráficos estáticos
- **seaborn** `0.13.2` - Visualización estadística
- **ipykernel** `6.30.0` - Soporte para Jupyter Notebooks

## 📋 Requisitos

- **Python** 3.10.12 o superior
- **pip** (gestor de paquetes de Python)

## 🔧 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <repository-url>
   cd headline-classifier
   ```

2. **Crear entorno virtual:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activar el entorno virtual:**
   
   **Windows:**
   ```bash
   .venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Descargar recursos de NLTK (primera vez):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 🚀 Uso

### 1. Ejecutar la API

```bash
uvicorn app.main:app --reload
```

### 2. Acceder a la API

La API estará disponible en:
- **URL base**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc

### 3. Probar la API

Puedes probar la API directamente desde la documentación interactiva en http://localhost:8000/docs o usando cURL como se muestra en los ejemplos.

## 📡 Endpoints Disponibles

### `POST /predict`
Clasifica un headline en una categoría.

**Request:**
```json
{
  "headline": "Breaking news: Stock market hits record high"
}
```

**Response:**
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

### `GET /categories`
Obtiene todas las categorías disponibles.

**Response:**
```json
{
  "categories": {
    "BUSINESS": 0,
    "POLITICS": 1,
    "TECHNOLOGY": 2,
    "SPORTS": 3,
    "ENTERTAINMENT": 4
  }
}
```

## 🧪 Ejemplo de Uso con cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"headline": "New smartphone released with advanced camera features"}'
```

## 🔄 Flujo de Procesamiento

Este flujo fue definido y explorado en detalle en `app/preprocessing/eda.ipynb`, donde se realiza el análisis exploratorio y diseño del pipeline de clasificación.

1. **Preprocesamiento**: Limpieza y normalización del texto
2. **Tokenización**: Separación en tokens usando NLTK
3. **Vectorización**: Conversión a vectores TF-IDF
4. **Predicción**: Clasificación usando Random Forest
5. **Post-procesamiento**: Cálculo de confianza y probabilidades

## 🛠️ Desarrollo

### Entorno de Desarrollo
- **Lenguaje**: Python 3.10.12
- **Sistema Operativo**: WSL2 Ubuntu 22.04.5 LTS x86_64
- **IDE**: VS Code

### Comandos Útiles

**Actualizar dependencias:**
```bash
pip freeze > requirements.txt
```

**Ejecutar en modo desarrollo:**
```bash
uvicorn app.main:app --reload
```

## 📊 Arquitectura del Modelo

- **Algoritmo**: Random Forest Classifier
- **Vectorización**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Balanceo**: Oversampling para clases minoritarias
- **Evaluación**: Classification report y accuracy score

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.