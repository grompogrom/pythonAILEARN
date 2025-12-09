from huggingface_hub import InferenceClient
import time

# Твой HuggingFace токен
hf_token = 'hf_********************************'  # Получи на https://huggingface.co/settings/tokens

# Выбери 2 модели из разных частей списка HF
models = [
    "google/gemma-2-2b-it",
    "meta-llama/Meta-Llama-3-8B-Instruct", 
    "Qwen/Qwen2.5-72B-Instruct"
]
# Тестовый промпт
prompt = "Calculate non negative integer n to make expression 1000 + 100 * n + 10 * n * n + n * n * n equal to 3623. Write short answer."
messages = [{"role": "user", "content": prompt}]

# Функция для замера
def test_model(model_name, messages, token):
    client = InferenceClient(model=model_name, token=token)
    
    start_time = time.time()
    response = client.chat_completion(
        messages=messages,
        max_tokens=2000
    )
    end_time = time.time()
    
    # Результаты
    response_time = end_time - start_time
    content = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else "N/A"
    
    return {
        "model": model_name,
        "response_time": f"{response_time:.2f}s",
        "tokens": tokens_used,
        "content": content,
        "cost": "Free (serverless API)"  # Бесплатно в рамках лимитов
    }

# Тестирование
for model in models:
    result = test_model(model, messages, hf_token)
    print(f"\n{'='*60}")
    print(f"Модель: {result['model']}")
    print(f"Время: {result['response_time']}")
    print(f"Токены: {result['tokens']}")
    print(f"Стоимость: {result['cost']}")
    print(f"Ответ: {result['content']}")
