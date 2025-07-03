from selenium import webdriver
from selenium.webdriver.firefox.service import Service # Mudança aqui para Firefox
from selenium.webdriver.firefox.options import Options # Mudança aqui para Firefox
from selenium.common.exceptions import WebDriverException, TimeoutException
import os
import time

# --- Configurações Iniciais ---
# 1. Caminho para o executável do GeckoDriver
#    Substitua 'seu_caminho_para_geckodriver' pelo caminho real onde você salvou o geckodriver.
#    Se o geckodriver estiver na mesma pasta do script, você pode usar apenas 'geckodriver' (ou 'geckodriver.exe' no Windows)
GECKODRIVER_PATH = 'venv/geckodriver' # Exemplo para Linux/macOS
# No Windows, seria algo como: GECKODRIVER_PATH = 'seu_caminho_para_geckodriver/geckodriver.exe'

# 2. Pasta para salvar os screenshots
OUTPUT_DIR = 'screenshots_legitimos'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 3. Arquivo com a lista de URLs
URL_LIST_FILE = 'sites_legitimos.txt' # Certifique-se de que este arquivo existe

# --- Configurações do Navegador (Opcional, mas Recomendado) ---
firefox_options = Options() # Mudança aqui
# Modo headless: o navegador roda em segundo plano, sem interface gráfica.
# Descomente a linha abaixo se quiser rodar em modo headless:
# firefox_options.add_argument("--headless") # Mudança aqui para o argumento do Firefox
firefox_options.add_argument("--width=1920") # Define a largura da janela
firefox_options.add_argument("--height=1080") # Define a altura da janela
# O Firefox geralmente não precisa de --no-sandbox ou --disable-dev-shm-usage como o Chrome

# --- Configuração do Serviço do GeckoDriver ---
service = Service(GECKODRIVER_PATH) # Mudança aqui para Service do Firefox

def take_screenshot(url, driver, output_dir):
    """
    Navega até uma URL e tira um screenshot da página inteira.
    """
    # Cria um nome de arquivo seguro a partir da URL
    file_name = f"{url.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '_').replace(':', '')}.png"
    output_path = os.path.join(output_dir, file_name)

    try:
        print(f"Tentando abrir: {url}")
        driver.set_page_load_timeout(30) # Define um timeout para o carregamento da página
        driver.get(url)
        time.sleep(5) # Dá um tempo para a página carregar conteúdo dinâmico

        # Rola a página para garantir que todo o conteúdo seja capturado
        # A lógica para altura total é diferente para o Firefox e pode não ser tão simples quanto no Chrome
        # Para capturar a página inteira no Firefox (se o driver suportar), use full_page=True
        driver.save_screenshot(output_path) # Por padrão, ele tenta capturar o que está visível.
                                            # Para página inteira, a documentação pode variar com o driver.
                                            # Alguns drivers podem ter um argumento full_page=True aqui.
                                            # Ou você pode precisar usar métodos de rolagem via JS.
        print(f"Screenshot salvo: {output_path}")

    except TimeoutException:
        print(f"Erro: Timeout ao carregar a página {url}")
    except WebDriverException as e:
        print(f"Erro do WebDriver ao processar {url}: {e}")
    except Exception as e:
        print(f"Erro inesperado ao processar {url}: {e}")

# --- Loop Principal ---
if __name__ == "__main__":
    driver = None # Inicializa driver como None
    try:
        # Inicia o WebDriver do Firefox
        driver = webdriver.Firefox(service=service, options=firefox_options) # Mudança aqui para Firefox

        # Lê as URLs do arquivo
        with open(URL_LIST_FILE, 'r') as f:
            urls = [line.strip() for line in f if line.strip()] # Garante que não há linhas vazias

        for url in urls:
            take_screenshot(url, driver, OUTPUT_DIR)

    except FileNotFoundError:
        print(f"Erro: O arquivo '{URL_LIST_FILE}' não foi encontrado. Crie-o com suas URLs.")
    except Exception as e:
        print(f"Um erro crítico ocorreu: {e}")
    finally:
        # Garante que o navegador seja fechado, mesmo se ocorrerem erros
        if driver:
            driver.quit()
            print("Navegador fechado.")