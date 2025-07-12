from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time

# --- Configurações Iniciais ---
GECKODRIVER_PATH = 'venv/geckodriver' # Ou o caminho correto para seu geckodriver

# Pasta para salvar os screenshots de phishing
OUTPUT_DIR = 'screenshots_phishing'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Nome do arquivo de texto com as URLs de phishing
URL_LIST_FILE = 'phishing_urls_openphish.txt' # Agora lendo de um TXT

# --- Configurações do Navegador para MAIOR SEGURANÇA (na VM!) ---
firefox_options = Options()
firefox_options.add_argument("--headless") # Modo headless: recomendado para phishing
firefox_options.add_argument("--width=1920")
firefox_options.add_argument("--height=1080")
firefox_options.add_argument("--disable-notifications")
firefox_options.add_argument("--disable-popup-blocking")
firefox_options.set_preference("browser.download.folderList", 2)
firefox_options.set_preference("browser.download.dir", "/dev/null")
firefox_options.set_preference("browser.download.useDownloadDir", True)
firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream,application/zip,application/x-zip,application/x-zip-compressed,application/x-rar-compressed,application/x-rar,application/pdf")
firefox_options.set_preference("pdfjs.disabled", True)

service = Service(GECKODRIVER_PATH)

def read_urls_from_file(file_path):
    """
    Lê URLs de um arquivo de texto, uma por linha.
    """
    urls = []
    if not os.path.exists(file_path):
        print(f"Erro: O arquivo de URLs '{file_path}' não foi encontrado.")
        print("Por favor, certifique-se de que o arquivo TXT com as URLs foi gerado.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and (url.startswith("http://") or url.startswith("https://")):
                    urls.append(url)
        print(f"Total de URLs carregadas do arquivo: {len(urls)}")
        return urls
    except Exception as e:
        print(f"Erro ao ler URLs do arquivo '{file_path}': {e}")
        return []

def handle_deceptive_site_warning(driver, url):
    """
    Tenta lidar com o aviso 'Deceptive site ahead' no Firefox.
    Retorna True se conseguiu prosseguir, False caso contrário.
    """
    try:
        # Espera um curto período para a página de aviso carregar
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "advancedButton")) # ID do botão "See details"
        )

        print(f"Aviso 'Deceptive site ahead' detectado para: {url}")

        # Clica em "See details"
        see_details_button = driver.find_element(By.ID, "advancedButton")
        see_details_button.click()
        time.sleep(1) # Pequena pausa para a próxima seção aparecer

        # Espera pelo link "Ignore the risk" ou similar
        try:
            ignore_risk_link = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "exceptionDialogButton"))
            )
            ignore_risk_link.click()
            print(f"Clicou em 'Ignore the risk' para: {url}")
            time.sleep(5) # Dá um tempo extra para a página de phishing carregar
            return True
        except TimeoutException:
            print(f"AVISO: Não encontrou o botão 'Ignore the risk' para {url} após clicar em 'See details'.")
            return False
        except NoSuchElementException:
            print(f"AVISO: Elemento 'Ignore the risk' não encontrado para {url}.")
            return False

    except TimeoutException:
        # O aviso não apareceu ou não carregou a tempo
        return False
    except NoSuchElementException:
        # Elementos do aviso não foram encontrados (talvez a página não seja o aviso)
        return False
    except Exception as e:
        print(f"Erro inesperado ao tentar lidar com o aviso para {url}: {e}")
        return False

def take_screenshot_phishing(url, driver, output_dir):
    """
    Navega até uma URL de phishing, lida com o aviso de segurança e tira um screenshot.
    """
    timestamp = int(time.time())
    safe_url_name = url.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '_').replace(':', '')
    file_name = f"{timestamp}_{safe_url_name[:150]}.png"
    output_path = os.path.join(output_dir, file_name)

    try:
        print(f"Tentando abrir URL de phishing: {url}")
        driver.set_page_load_timeout(30)

        try:
            driver.get(url)
            # Verifica se o aviso de site enganoso apareceu
            # O texto da página do aviso geralmente contém "Deceptive site ahead" ou "Security Risk" no título ou no corpo.
            # Também podemos verificar a URL da página de aviso se ela for padronizada.
            # Usando uma verificação mais robusta:
            current_url = driver.current_url
            if "safeBrowse.google.com" in current_url or \
               "Deceptive site ahead" in driver.title or \
               "Warning: Potential Security Risk Ahead" in driver.title or \
               "bolodboss.github.io" in current_url: # Você pode adicionar domínios específicos de aviso
                
                success_handling_warning = handle_deceptive_site_warning(driver, url)
                if not success_handling_warning:
                    print(f"Não foi possível contornar o aviso para {url}. Pulando screenshot.")
                    return # Não tira screenshot se não conseguiu contornar
            else:
                # Se não há aviso, apenas espera a página carregar
                time.sleep(5)

        except TimeoutException:
            print(f"AVISO: Timeout inicial ao carregar {url}. (Pode estar offline ou lento)")
            return

        driver.save_screenshot(output_path)
        print(f"Screenshot de phishing salvo: {output_path}")

    except WebDriverException as e:
        print(f"AVISO: Erro do WebDriver ao processar URL de phishing {url}: {e}")
    except Exception as e:
        print(f"ERRO INESPERADO ao processar URL de phishing {url}: {e}")

# --- Loop Principal ---
if __name__ == "__main__":
    driver = None
    try:
        # 1. Carregar as URLs de phishing do arquivo TXT
        phishing_urls_from_txt = read_urls_from_file(URL_LIST_FILE)

        if not phishing_urls_from_txt:
            print("Nenhuma URL de phishing válida carregada do TXT para processar. Saindo.")
        else:
            # Inicia o WebDriver do Firefox com opções de segurança
            driver = webdriver.Firefox(service=service, options=firefox_options)

            # Para um teste inicial, comece com um número pequeno.
            urls_to_process = phishing_urls_from_txt[:200] # Processa as primeiras 200 URLs

            print(f"Processando {len(urls_to_process)} URLs de phishing...")
            for url in urls_to_process:
                take_screenshot_phishing(url, driver, OUTPUT_DIR)

    except Exception as e:
        print(f"Um erro crítico ocorreu no fluxo principal: {e}")
    finally:
        if driver:
            driver.quit()
            print("Navegador de phishing fechado.")