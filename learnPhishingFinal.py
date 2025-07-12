import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pickle # Para salvar/carregar o tokenizer
import os

# --- Configurações de Arquivos ---
LEGITIMATE_URLS_FILE = 'sites_legitimos_limpos.txt'
PHISHING_URLS_FILE = 'phishing_urls_openphish.txt'

# --- Nomes dos arquivos de saída para o modelo e tokenizer ---
MODEL_SAVE_PATH = 'url_phishing_detector_model.keras' # Novo formato .keras
TOKENIZER_SAVE_PATH = 'url_tokenizer.pkl' # Usaremos .pkl (pickle) para o tokenizer

# --- Parâmetros da Rede Neural ---
MAX_URL_LENGTH = 200  # Máximo comprimento de URL a ser considerado
EMBEDDING_DIM = 50    # Dimensão dos vetores de embedding para cada caractere
VOCAB_SIZE = 256 + 1  # 256 caracteres ASCII (0-255) + 1 para o token de "zero padding"
FILTERS = 128         # Número de filtros na camada Conv1D
KERNEL_SIZE = 5       # Tamanho do kernel na camada Conv1D
DROPOUT_RATE = 0.5    # Taxa de dropout para regularização
BATCH_SIZE = 64       # Tamanho do lote de treinamento
EPOCHS = 10           # Número de épocas de treinamento

# --- 1. Coleta e Preparação dos Dados (URLs) ---
def load_urls(file_path, label):
    """Carrega URLs de um arquivo e atribui um rótulo."""
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:
                    urls.append({'url': url, 'label': label})
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado. Certifique-se de que o caminho está correto.")
    return urls

print("Carregando URLs legítimas...")
legitimate_data_full = load_urls(LEGITIMATE_URLS_FILE, 0) # Carrega todas as URLs legítimas
print(f"Total de URLs legítimas disponíveis: {len(legitimate_data_full)}")

print("Carregando URLs de phishing...")
phishing_data_full = load_urls(PHISHING_URLS_FILE, 1)    # Carrega todas as URLs de phishing
print(f"Total de URLs de phishing disponíveis: {len(phishing_data_full)}")

# --- Balanceamento de Classes: Usar todas as URLs de phishing e igualar as legítimas ---
num_phishing_urls = len(phishing_data_full)
TARGET_LEGITIMATE_COUNT = num_phishing_urls

if len(legitimate_data_full) < TARGET_LEGITIMATE_COUNT:
    print(f"Aviso: O número de URLs legítimas disponíveis ({len(legitimate_data_full)}) é menor do que o número de URLs de phishing ({TARGET_LEGITIMATE_COUNT}).")
    print("Usando todas as URLs legítimas disponíveis para o balanceamento.")
    selected_legitimate_data = legitimate_data_full
else:
    # Seleciona aleatoriamente 'TARGET_LEGITIMATE_COUNT' URLs legítimas
    selected_indices = np.random.choice(
        len(legitimate_data_full),
        TARGET_LEGITIMATE_COUNT,
        replace=False # Garante que as URLs legítimas selecionadas sejam únicas
    )
    selected_legitimate_data = [legitimate_data_full[i] for i in selected_indices]

print(f"\nURLs de phishing usadas no treinamento: {len(phishing_data_full)}")
print(f"URLs legítimas selecionadas para balanceamento: {len(selected_legitimate_data)}")

# Combinar em um DataFrame
data = pd.DataFrame(selected_legitimate_data + phishing_data_full)

if data.empty:
    print("Nenhum dado carregado após balanceamento. Verifique os arquivos de URL e seus caminhos.")
    exit()

# Embaralhar o DataFrame para garantir que as classes não estejam em blocos
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTotal de URLs no dataset final (após balanceamento e embaralhamento): {len(data)}")
print(f"URLs legítimas no dataset final: {data[data['label'] == 0].shape[0]}")
print(f"URLs de phishing no dataset final: {data[data['label'] == 1].shape[0]}")

# Separar URLs (X) e rótulos (y)
urls = data['url'].tolist()
labels = data['label'].values

# Dividir em conjuntos de treinamento e teste
# stratify=labels garante que a proporção de classes seja mantida nos dois conjuntos
X_train_urls, X_test_urls, y_train, y_test = train_test_split(
    urls, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTamanho do conjunto de treinamento (URLs): {len(X_train_urls)}")
print(f"Tamanho do conjunto de teste (URLs): {len(X_test_urls)}")
print(f"Phishing no treino: {np.sum(y_train)} | Legítimo no treino: {len(y_train) - np.sum(y_train)}")
print(f"Phishing no teste: {np.sum(y_test)} | Legítimo no teste: {len(y_test) - np.sum(y_test)}")


# --- 2. Pré-processamento da URL para Redes Neurais ---

print("\nInicializando e fitando o Tokenizer...")
# Usar Tokenizer para mapear caracteres para inteiros
tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, oov_token="<unk>")
# Ajusta o tokenizer APENAS nos dados de treinamento para evitar vazamento de dados (data leakage)
tokenizer.fit_on_texts(X_train_urls)

# Salvar o tokenizer treinado
try:
    with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer salvo com sucesso em '{TOKENIZER_SAVE_PATH}'")
except Exception as e:
    print(f"Erro ao salvar o tokenizer: {e}")


# Converter URLs em sequências de inteiros
X_train_sequences = tokenizer.texts_to_sequences(X_train_urls)
X_test_sequences = tokenizer.texts_to_sequences(X_test_urls)

# Padding das sequências para ter o mesmo comprimento
# 'post' padding adiciona zeros ao final; 'truncating'='post' corta do final se for muito longo
X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_URL_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_URL_LENGTH, padding='post', truncating='post')

print(f"\nShape dos dados de treinamento padded: {X_train_padded.shape}")
print(f"Shape dos dados de teste padded: {X_test_padded.shape}")

# --- 3. Construção do Modelo de Rede Neural ---

print("\nConstruindo o modelo de Rede Neural...")
model = Sequential([
    # Embedding Layer: Converte os inteiros (caracteres) em vetores densos e de baixa dimensão
    # input_dim é o tamanho do vocabulário, output_dim é a dimensão do embedding
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_URL_LENGTH),

    # Camada Convolucional 1D: Aprende padrões locais (como sequências de caracteres ou "n-grams")
    # relu é uma função de ativação comum para camadas ocultas
    Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'),

    # GlobalMaxPooling1D: Reduz a dimensão pegando o valor máximo de cada filtro ao longo da sequência
    # Isso ajuda a capturar as características mais salientes, independentemente da sua posição
    GlobalMaxPooling1D(),

    # Camadas Densas (Fully Connected): Para combinar as características aprendidas e fazer a classificação
    Dense(64, activation='relu'), # Camada oculta com 64 neurônios
    Dropout(DROPOUT_RATE), # Dropout para prevenir overfitting, desligando aleatoriamente neurônios durante o treino

    # Camada de Saída: 1 neurônio com ativação sigmoide para classificação binária (phishing ou legítimo)
    Dense(1, activation='sigmoid')
])

# Compila o modelo
# 'adam' é um otimizador popular e eficiente
# 'binary_crossentropy' é a função de perda padrão para classificação binária
# 'accuracy' é a métrica para monitorar durante o treinamento
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary() # Imprime um resumo da arquitetura do modelo

# --- 4. Treinamento e Avaliação ---

print("\nIniciando o treinamento do modelo...")
history = model.fit(
    X_train_padded, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1, # Usa 10% dos dados de treinamento para validação (bom para monitorar overfitting)
    verbose=1 # Exibe o progresso do treinamento
)
print("Modelo treinado com sucesso!")

# Salvar o modelo treinado no novo formato .keras
try:
    model.save(MODEL_SAVE_PATH)
    print(f"Modelo salvo com sucesso em '{MODEL_SAVE_PATH}'")
except Exception as e:
    print(f"Erro ao salvar o modelo: {e}")

print("\nAvaliando o modelo no conjunto de teste final...")
# Avalia o desempenho do modelo em dados que ele nunca viu
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")

# Fazer previsões para calcular Precisão, Recall e F1-Score
y_pred_proba = model.predict(X_test_padded)
y_pred = (y_pred_proba > 0.5).astype(int) # Converte probabilidades para 0 ou 1 (limiar de 0.5)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nRelatório de Classificação Detalhado:")
# Fornece precisão, recall e f1-score para cada classe (0: legítimo, 1: phishing)
print(classification_report(y_test, y_pred, target_names=['Legítimo', 'Phishing']))