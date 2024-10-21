import requests
import json
import re
import time


openrouter_api_key = 'API-KEY'

discussion_model_ids = ['nousresearch/hermes-3-llama-3.1-405b:free', 'liquid/lfm-40b', 'google/gemini-flash-1.5-exp', 'meta-llama/llama-3.1-70b-instruct:free',
                        'google/gemma-2-9b-it:free']

evaluator_model_id = 'nousresearch/hermes-3-llama-3.1-405b:free'

initial_model_id = 'nousresearch/hermes-3-llama-3.1-405b:free'


QUALITY_THRESHOLD = 0.95
ALERT_ITERATIONS = 20
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

def make_request_with_retries(url, headers, payload, retries=MAX_RETRIES):
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
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"{instruction}\n{input_text}" if instruction else input_text

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = make_request_with_retries(url, headers, payload)

    return response.json()['choices'][0]['message']['content'].strip()

def evaluate_response(response):
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

    response = make_request_with_retries(url, headers, payload)

    response_text = response.json()['choices'][0]['message']['content'].strip()
    score_match = re.search(r'(\d\.\d)', response_text)
    if score_match:
        return float(score_match.group(1))
    else:
        raise ValueError(f"No se pudo extraer una puntuación numérica de la respuesta: {response_text}")

def rephrase_response(response):
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

    response = make_request_with_retries(url, headers, payload)

    return response.json()['choices'][0]['message']['content'].strip()

def improve_response_in_discussion(initial_input):
    current_input = initial_input
    improvement_count = 0

    for model_id in discussion_model_ids:
        instruction = "Por favor, mejora la respuesta, corrigiendo errores, añadiendo información relevante y afinando la calidad."
        response = get_model_response(model_id, current_input, instruction)
        quality_score = evaluate_response(response)
        
        improvement_count += 1

        if quality_score >= QUALITY_THRESHOLD:
            return rephrase_response(response), improvement_count
        else:
            current_input = response

        if improvement_count >= ALERT_ITERATIONS:
            print(f"Alerta: La respuesta ha sido mejorada más de {ALERT_ITERATIONS} veces.")

    return current_input, improvement_count

def process_question(question):

    initial_response = get_model_response(initial_model_id, question)
    initial_score = evaluate_response(initial_response)

    if initial_score >= QUALITY_THRESHOLD:
        return rephrase_response(initial_response), 0
    else:
        final_response, improvement_count = improve_response_in_discussion(initial_response)
        
        return final_response, improvement_count

if __name__ == '__main__':
    question = input("Por favor, introduce tu pregunta: ")
    response, improvement_count = process_question(question)
    print(f"Respuesta final: {response}")
    print(f"Número de revisiones: {improvement_count}")
