import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# --- Configurações de Arquivos ---
LEGITIMATE_URLS_FILE = 'sites_legitimos.txt'
PHISHING_URLS_FILE = 'phishing_urls_openphish.txt'

# --- Parâmetros da Rede Neural ---
MAX_URL_LENGTH = 200  # Máximo comprimento de URL a ser considerado
EMBEDDING_DIM = 50    # Dimensão dos vetores de embedding
VOCAB_SIZE = 256 + 1  # 256 caracteres ASCII + 1 para o token de "zero padding"
FILTERS = 128         # Número de filtros na camada Conv1D
KERNEL_SIZE = 5       # Tamanho do kernel na camada Conv1D
DROPOUT_RATE = 0.5    # Taxa de dropout para regularização
BATCH_SIZE = 32       # Reduzir o batch size para datasets menores pode ser útil
EPOCHS = 10           # Número de épocas de treinamento

# --- NOVO: Número de amostras para o teste inicial ---
TARGET_SAMPLE_COUNT = 1000 # 1000 URLs de phishing e 1000 URLs legítimas

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
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
    return urls

# Carregar dados completos
legitimate_data_full = load_urls(LEGITIMATE_URLS_FILE, 0)
phishing_data_full = load_urls(PHISHING_URLS_FILE, 1)

# --- Seleção de Subconjunto para Teste ---
print(f"Total de URLs legítimas disponíveis: {len(legitimate_data_full)}")
print(f"Total de URLs de phishing disponíveis: {len(phishing_data_full)}")

# Selecionar TARGET_SAMPLE_COUNT URLs de phishing
if len(phishing_data_full) < TARGET_SAMPLE_COUNT:
    print(f"Aviso: Não há {TARGET_SAMPLE_COUNT} URLs de phishing disponíveis. Usando todas as {len(phishing_data_full)}.")
    selected_phishing_data = phishing_data_full
else:
    selected_phishing_data = np.random.choice(
        phishing_data_full,
        TARGET_SAMPLE_COUNT,
        replace=False
    ).tolist()

# Selecionar TARGET_SAMPLE_COUNT URLs legítimas
if len(legitimate_data_full) < TARGET_SAMPLE_COUNT:
    print(f"Aviso: Não há {TARGET_SAMPLE_COUNT} URLs legítimas disponíveis. Usando todas as {len(legitimate_data_full)}.")
    selected_legitimate_data = legitimate_data_full
else:
    selected_legitimate_data = np.random.choice(
        legitimate_data_full,
        TARGET_SAMPLE_COUNT,
        replace=False
    ).tolist()

print(f"\nURLs de phishing selecionadas para teste: {len(selected_phishing_data)}")
print(f"URLs legítimas selecionadas para teste: {len(selected_legitimate_data)}")

# Combinar em um DataFrame
data = pd.DataFrame(selected_legitimate_data + selected_phishing_data)

if data.empty:
    print("Nenhum dado selecionado para teste. Verifique os arquivos de URL e seus caminhos.")
    exit()

# Embaralhar o DataFrame para garantir que as classes não estejam em blocos
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTotal de URLs no dataset de teste (após seleção): {len(data)}")
print(f"URLs legítimas no dataset de teste: {data[data['label'] == 0].shape[0]}")
print(f"URLs de phishing no dataset de teste: {data[data['label'] == 1].shape[0]}")

# Separar URLs (X) e rótulos (y)
urls = data['url'].tolist()
labels = data['label'].values

# Dividir em conjuntos de treinamento e teste
# O test_size padrão é 0.25 (25%), então 1000 + 1000 = 2000 amostras
# Com test_size=0.2, teremos 1600 para treino e 400 para teste
X_train_urls, X_test_urls, y_train, y_test = train_test_split(
    urls, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTamanho do conjunto de treinamento (URLs): {len(X_train_urls)}")
print(f"Tamanho do conjunto de teste (URLs): {len(X_test_urls)}")
print(f"Phishing no treino: {np.sum(y_train)} Legítimo no treino: {len(y_train) - np.sum(y_train)}")
print(f"Phishing no teste: {np.sum(y_test)} Legítimo no teste: {len(y_test) - np.sum(y_test)}")


# --- 2. Pré-processamento da URL para Redes Neurais ---

tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, oov_token="<unk>")
tokenizer.fit_on_texts(X_train_urls) # Fit apenas nos dados de treino para evitar data leakage

X_train_sequences = tokenizer.texts_to_sequences(X_train_urls)
X_test_sequences = tokenizer.texts_to_sequences(X_test_urls)

X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_URL_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_URL_LENGTH, padding='post', truncating='post')

print(f"\nShape dos dados de treinamento padded: {X_train_padded.shape}")

# --- 3. Construção do Modelo de Rede Neural ---

print("\nConstruindo o modelo de Rede Neural...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_URL_LENGTH),
    Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# --- 4. Treinamento e Avaliação ---

print("\nTreinando o modelo...")
history = model.fit(
    X_train_padded, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)
print("Modelo treinado com sucesso!")

print("\nAvaliando o modelo no conjunto de teste...")
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")

y_pred_proba = model.predict(X_test_padded)
y_pred = (y_pred_proba > 0.5).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Após o treinamento
model.save('url_phishing_detector_model.h5') # Salva o modelo em formato HDF5
print("Modelo salvo como 'url_phishing_detector_model.h5'")