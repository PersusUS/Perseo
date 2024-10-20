import requests
import json
import re
import time

# Configura tu clave de API de OpenRouter
openrouter_api_key = 'API-KEY'

# Actualización: Solo se utilizará nousresearch en la lista de modelos
discussion_model_ids = ['nousresearch/hermes-3-llama-3.1-405b:free', 'liquid/lfm-40b', 'google/gemini-flash-1.5-exp', 'meta-llama/llama-3.1-70b-instruct:free',
                        'google/gemma-2-9b-it:free', 'meta-llama/llama-3.1-405b-instruct:free']

# ID del modelo evaluador
evaluator_model_id = 'nousresearch/hermes-3-llama-3.1-405b:free'

# ID del primer modelo externo
initial_model_id = 'nousresearch/hermes-3-llama-3.1-405b:free'

# Umbral de calidad para la respuesta (ajustado a 0.95)
QUALITY_THRESHOLD = 0.95

# Límite de iteraciones para alertar al usuario
ALERT_ITERATIONS = 20

# Número máximo de reintentos en caso de error de timeout
MAX_RETRIES = 3

# Timeout para la solicitud HTTP (en segundos)
REQUEST_TIMEOUT = 30

def make_request_with_retries(url, headers, payload, retries=MAX_RETRIES):
    """Función auxiliar para realizar solicitudes HTTP con reintentos."""
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response
            else:
                raise Exception(f"Error en la solicitud: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            print(f"Timeout en el intento {attempt + 1} de {retries}. Reintentando...")
            time.sleep(2)  # Esperar 2 segundos antes de reintentar
    raise Exception(f"Error: Se alcanzó el número máximo de reintentos ({retries})")

def get_model_response(model_id, input_text, instruction=""):
    """Llama al modelo especificado usando la API de OpenRouter con reintentos."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"{instruction}\n{input_text}" if instruction else input_text

    # Estructura de mensajes que se envían a la API
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Realizamos la solicitud con reintentos
    response = make_request_with_retries(url, headers, payload)

    return response.json()['choices'][0]['message']['content'].strip()

def evaluate_response(response):
    """Evalúa la calidad de la respuesta usando un modelo externo, extrayendo solo el valor numérico."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    
    eval_prompt = f"Evalúa la siguiente respuesta en una escala de 0 a 1:\n\n{response}"
    
    payload = {
        "model": evaluator_model_id,
        "messages": [
            {
                "role": "user",
                "content": eval_prompt
            }
        ]
    }

    # Realizamos la solicitud con reintentos
    response = make_request_with_retries(url, headers, payload)

    # Extraer la puntuación usando regex (buscando el primer número decimal entre 0 y 1)
    response_text = response.json()['choices'][0]['message']['content'].strip()
    score_match = re.search(r'(\d\.\d)', response_text)
    if score_match:
        return float(score_match.group(1))
    else:
        raise ValueError(f"No se pudo extraer una puntuación numérica de la respuesta: {response_text}")

def rephrase_response(response):
    """Reescribe la respuesta para mayor claridad."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    
    rephrase_prompt = f"Reescribe esta respuesta de manera clara y estructurada:\n\n{response}"
    
    payload = {
        "model": initial_model_id,
        "messages": [
            {
                "role": "user",
                "content": rephrase_prompt
            }
        ]
    }

    # Realizamos la solicitud con reintentos
    response = make_request_with_retries(url, headers, payload)

    return response.json()['choices'][0]['message']['content'].strip()

def improve_response_in_discussion(initial_input):
    """Pasa la respuesta por el grupo de discusión de modelos para mejorarla iterativamente."""
    current_input = initial_input
    improvement_count = 0

    for model_id in discussion_model_ids:
        instruction = "Por favor, mejora la respuesta, corrigiendo errores, añadiendo información relevante y afinando la calidad."
        response = get_model_response(model_id, current_input, instruction)
        quality_score = evaluate_response(response)
        
        improvement_count += 1

        if quality_score >= QUALITY_THRESHOLD:
            # Si la respuesta cruza el umbral, se reescribe y retorna
            return rephrase_response(response), improvement_count
        else:
            # Si no cruza el umbral, la respuesta se mejora y pasa al siguiente modelo
            current_input = response

        # Notificar al usuario si se alcanzan muchas iteraciones
        if improvement_count >= ALERT_ITERATIONS:
            print(f"Alerta: La respuesta ha sido mejorada más de {ALERT_ITERATIONS} veces.")

    # Si no se supera el umbral tras pasar por todos los modelos, devolvemos lo mejor que se pudo hacer
    return current_input, improvement_count

def process_question(question):
    """Flujo principal para procesar la pregunta del usuario."""
    # Primer modelo externo responde inicialmente a la pregunta
    initial_response = get_model_response(initial_model_id, question)
    initial_score = evaluate_response(initial_response)

    if initial_score >= QUALITY_THRESHOLD:
        # Si la respuesta inicial es suficientemente buena, la reescribimos y retornamos
        return rephrase_response(initial_response), 0
    else:
        # Si la calidad inicial no es suficiente, la pasamos al grupo de discusión
        final_response, improvement_count = improve_response_in_discussion(initial_response)
        
        return final_response, improvement_count

if __name__ == '__main__':
    question = input("Por favor, introduce tu pregunta: ")
    response, improvement_count = process_question(question)
    print(f"Respuesta final: {response}")
    print(f"Número de revisiones: {improvement_count}")
