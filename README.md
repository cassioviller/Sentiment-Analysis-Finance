# 📊 Análise de Sentimentos de Notícias Financeiras

## 📌 Visão Geral

Este projeto utiliza **análise de sentimentos** para interpretar notícias financeiras e avaliar o impacto sobre os mercados. Ele combina **dados financeiros, notícias e modelos de NLP (Processamento de Linguagem Natural)** para fornecer insights sobre o sentimento do mercado em relação a empresas e ativos financeiros.

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit** – Interface interativa para visualização de dados.
- **Transformers (Hugging Face)** – Modelo de análise de sentimentos baseado em BERT.
- **Yahoo Finance API** – Coleta de dados de mercado.
- **NewsAPI** – Busca de notícias financeiras.
- **Pandas e Plotly** – Processamento e visualização de dados.

## 🧠 Metodologia

1. O usuário insere o **ticker da empresa** ou seu **nome**.
2. O sistema busca **notícias financeiras** relevantes usando a API do NewsAPI.
3. Um modelo de **Machine Learning** analisa as manchetes e determina se o sentimento é **positivo, neutro ou negativo**.
4. Os resultados são exibidos em um **gráfico interativo** e podem ser baixados.

## 📊 Exemplos de Resultados

Após a execução, o sistema gera gráficos mostrando a evolução dos sentimentos ao longo do tempo.

![Imagem do WhatsApp de 2025-01-22 à(s) 18 46 24_975196d7](https://github.com/user-attachments/assets/9b0a01c1-bb5f-4348-8ccf-11cf0840580f)
