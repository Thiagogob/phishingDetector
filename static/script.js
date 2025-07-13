document.getElementById('checkButton').addEventListener('click', async () => {
    const urlInput = document.getElementById('urlInput');
    const resultDiv = document.getElementById('result');
    const url = urlInput.value.trim();

    if (!url) {
        resultDiv.textContent = 'Por favor, insira uma URL.';
        resultDiv.style.color = 'orange';
        return;
    }

    resultDiv.textContent = 'Verificando...';
    resultDiv.style.color = 'gray';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: url })
        });

        const data = await response.json();

        if (response.ok) {
            resultDiv.textContent = `A URL é: ${data.result}`;
            resultDiv.style.color = data.result === 'Phishing' ? 'red' : 'green';
        } else {
            resultDiv.textContent = `Erro: ${data.error || 'Algo deu errado.'}`;
            resultDiv.style.color = 'red';
        }
    } catch (error) {
        resultDiv.textContent = `Erro na comunicação: ${error.message}`;
        resultDiv.style.color = 'red';
    }
});