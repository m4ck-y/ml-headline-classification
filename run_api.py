#!/usr/bin/env python3
"""
Script para ejecutar la API de clasificación de headlines
"""

import uvicorn
import os
import sys

def check_models():
    """Verifica que los modelos necesarios existan"""
    required_files = [
        "models/random_forest.pkl",
        "models/vectorizer.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Error: Faltan los siguientes archivos de modelo:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Ejecuta primero el entrenamiento del modelo:")
        print("   python -m app.main")
        return False
    
    print("✅ Todos los archivos de modelo están presentes")
    return True

def main():
    print("🚀 Iniciando API de Clasificación de Headlines")
    print("=" * 50)
    
    # Verificar modelos
    if not check_models():
        sys.exit(1)
    
    print("🌐 Iniciando servidor en http://localhost:8000")
    print("📖 Documentación disponible en http://localhost:8000/docs")
    print("🔍 Para probar la API, ejecuta: python test_api.py")
    print("\n⏹️  Presiona Ctrl+C para detener el servidor")
    print("=" * 50)
    
    # Ejecutar la API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Recarga automática en desarrollo
        log_level="info"
    )

if __name__ == "__main__":
    main()