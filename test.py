import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle # Para carregar o tokenizer

# --- Configurações (Devem ser as mesmas usadas no treinamento!) ---
MAX_URL_LENGTH = 200  # Máximo comprimento de URL usado no treinamento
# VOCAB_SIZE não é mais estritamente necessário aqui, pois o tokenizer carregado já tem essa informação
# Mas é bom manter para referência se precisar re-fitar um tokenizer temporário
VOCAB_SIZE = 256 + 1

MODEL_PATH = 'url_phishing_detector_model.keras' # Caminho para o modelo salvo
TOKENIZER_PATH = 'url_tokenizer.pkl'       # Caminho para o tokenizer salvo

# --- 1. Carregar o Modelo e o Tokenizer Salvos ---
def load_trained_assets(model_path, tokenizer_path):
    """Carrega o modelo Keras treinado e o tokenizer do disco."""
    model = None
    tokenizer = None

    # Carregar Modelo
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em '{model_path}'.")
        print("Certifique-se de que o modelo foi salvo no treinamento e o caminho está correto.")
    else:
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Modelo carregado com sucesso de '{model_path}'")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")

    # Carregar Tokenizer
    if not os.path.exists(tokenizer_path):
        print(f"Erro: Tokenizer não encontrado em '{tokenizer_path}'.")
        print("Certifique-se de que o tokenizer foi salvo no treinamento e o caminho está correto.")
    else:
        try:
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            print(f"Tokenizer carregado com sucesso de '{tokenizer_path}'")
        except Exception as e:
            print(f"Erro ao carregar o tokenizer: {e}")
            
    return model, tokenizer

# --- 2. Pré-processamento da URL de Entrada ---
def preprocess_url(url, tokenizer_obj, max_length):
    """
    Pré-processa uma única URL para ser usada pelo modelo.
    Usa o tokenizer carregado e o padding com as mesmas configurações do treinamento.
    """
    # Converter URL em sequência de inteiros usando o tokenizer carregado
    sequence = tokenizer_obj.texts_to_sequences([url])
    # Padding da sequência para ter o mesmo comprimento (MAX_URL_LENGTH)
    # Garante que a entrada para o modelo tenha a dimensão esperada
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return padded_sequence

# --- Loop de Teste Interativo ---
if __name__ == "__main__":
    # Carregar o modelo e o tokenizer
    model, tokenizer = load_trained_assets(MODEL_PATH, TOKENIZER_PATH)
    
    if model is None or tokenizer is None:
        print("Não foi possível carregar todos os recursos necessários. Saindo.")
        exit()

    print("\n--- Detector de Phishing por URL ---")
    print("Digite 'sair' a qualquer momento para sair.")

    while True:
        input_url = input("\nInsira uma URL para testar: ").strip()

        if input_url.lower() == 'sair':
            print("Saindo do detector.")
            break

        if not input_url:
            print("Por favor, insira uma URL válida.")
            continue

        # Pré-processar a URL de entrada
        processed_url = preprocess_url(input_url, tokenizer, MAX_URL_LENGTH)

        # Fazer a previsão
        try:
            # model.predict retorna um array numpy, precisamos acessar o valor escalar
            prediction_proba = model.predict(processed_url)[0][0]
            
            # Interpretar a probabilidade e apresentar o resultado
            # O limiar de 0.5 é o padrão, você pode ajustar se precisar de mais sensibilidade/especificidade
            if prediction_proba >= 0.5:
                result = "PHISHING"
                confidence = f"{prediction_proba * 100:.2f}% de chance de ser PHISHING."
            else:
                result = "LEGÍTIMA"
                # A probabilidade de ser legítima é 1 - probabilidade de ser phishing
                confidence = f"{(1 - prediction_proba) * 100:.2f}% de chance de ser LEGÍTIMA."

            print(f"\nResultado da Análise:")
            print(f"URL: {input_url}")
            print(f"Classificação: {result}")
            print(f"Probabilidade: {confidence}")

        except Exception as e:
            print(f"Erro ao fazer a previsão para esta URL: {e}")
            print("Verifique se as configurações (MAX_URL_LENGTH) são as mesmas do treinamento.")
            print(f"Detalhes do erro: {e}")