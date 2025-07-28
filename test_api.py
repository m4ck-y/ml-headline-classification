import requests
import json

# URL base de la API
BASE_URL = "http://localhost:8000"

def test_api():
    """Prueba la API con algunos ejemplos"""
    
    # Verificar que la API esté funcionando
    print("🔍 Verificando estado de la API...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ API Status: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a la API. ¿Está ejecutándose?")
        return
    
    # Verificar salud de la API
    print("\n🏥 Verificando salud de la API...")
    health = requests.get(f"{BASE_URL}/health")
    print(f"✅ Health Check: {health.json()}")
    
    # Obtener categorías disponibles
    print("\n📋 Obteniendo categorías disponibles...")
    categories = requests.get(f"{BASE_URL}/categories")
    print(f"✅ Categorías: {categories.json()}")
    
    # Ejemplos de headlines para probar
    test_headlines = [
        "Breaking: Stock market reaches new record high",
        "Scientists discover new species in Amazon rainforest",
        "Local team wins championship after dramatic final",
        "New smartphone released with advanced camera features",
        "President announces new economic policies"
    ]
    
    print("\n🧪 Probando predicciones...")
    for headline in test_headlines:
        print(f"\n📰 Headline: '{headline}'")
        
        # Hacer la predicción
        payload = {"headline": headline}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"🎯 Categoría predicha: {result['predicted_category']}")
            print(f"🎲 Confianza: {result['confidence']:.2%}")
            
            # Mostrar top 3 probabilidades
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            print("📊 Top 3 probabilidades:")
            for category, prob in sorted_probs:
                print(f"   {category}: {prob:.2%}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_api()