from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.models import load_model # Importar para carregar o modelo Keras
from tensorflow.keras.preprocessing.sequence import pad_sequences # Importar para pad_sequences

app = Flask(__name__)

# --- Parâmetros de pré-processamento (devem ser os mesmos usados no treinamento) ---
MAX_URL_LENGTH = 200 # Certifique-se de que este valor seja o mesmo do learnPhishing.py

# Carregar o modelo treinado
MODEL_PATH = 'models/url_phishing_detector_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'

try:
    # Carregar o modelo Keras
    model = load_model(MODEL_PATH)
    print(f"Modelo Keras carregado de '{MODEL_PATH}'")

    # Carregar o tokenizer
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer carregado de '{TOKENIZER_PATH}'")

except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado ao carregar modelo ou tokenizer. Verifique o caminho: {e}")
    model = None
    tokenizer = None
except Exception as e:
    print(f"Erro inesperado ao carregar modelo ou tokenizer: {e}")
    model = None
    tokenizer = None

# Função para pré-processar a URL (usando o tokenizer carregado)
def preprocess_url(url):
    if tokenizer is None:
        raise ValueError("Tokenizer não foi carregado. Não é possível pré-processar URLs.")

    # Converter a URL para sequência de números usando o tokenizer
    sequence = tokenizer.texts_to_sequences([url]) # texts_to_sequences espera uma lista
    
    # Preencher/truncar a sequência para o comprimento máximo
    padded_sequence = pad_sequences(sequence, maxlen=MAX_URL_LENGTH, padding='post', truncating='post')
    
    return padded_sequence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "Modelo ou Tokenizer não carregados. Verifique os logs do servidor."}), 500

    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL não fornecida"}), 400

    try:
        # Pré-processar a URL usando a função dedicada
        processed_url = preprocess_url(url)
        
        # Fazer a previsão
        # O modelo Keras retorna uma array numpy, precisamos pegar o primeiro elemento
        # e aplicar o threshold
        prediction_proba = model.predict(processed_url)[0][0] # [0][0] para obter o escalar da probabilidade
        
        # O threshold de 0.5 é comum para modelos sigmoid
        result = "Legítimo" if prediction_proba < 0.5 else "Phishing"
        
        return jsonify({"url": url, "result": result, "confidence": float(prediction_proba)}) # Opcional: retornar confiança
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        return jsonify({"error": f"Erro na previsão: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)