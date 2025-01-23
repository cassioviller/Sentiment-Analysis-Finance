import os
import streamlit as st
import pandas as pd
from transformers import pipeline
import yfinance as yf
import requests
import plotly.express as px
from datetime import datetime
from yahooquery import search

# Suprimir avisos do Hugging Face Hub
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Suprimir avisos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Inicializar o pipeline de análise de sentimentos
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")



def get_ticker(company_name):
    try:
        results = search(company_name)
        if results and not results.empty:
            return results.iloc[0]['symbol']
        else:
            return None
    except Exception as e:
        print(f"Erro ao buscar ticker para {company_name}: {e}")
        return None



# Função para validar se o ticker é válido
def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        return 'symbol' in stock.info and stock.info['symbol'] == ticker
    except:
        return False

# Função para buscar notícias usando a NewsAPI
def fetch_news(query, api_key, from_date, to_date):
    url = ('https://newsapi.org/v2/everything?'
           f'q={query}&'
           f'from={from_date}&'
           f'to={to_date}&'
           'sortBy=publishedAt&'
           'language=pt&'
           f'apiKey={api_key}')
    response = requests.get(url)
    return response.json()

# Título da aplicação
st.title('Análise de Sentimentos de Notícias Financeiras')

# Campos de entrada
ticker_input = st.text_input('Digite o Ticker da Empresa (ex: AAPL)')
company_input = st.text_input('Digite o Nome da Empresa (ex: Apple Inc.)')

# Seleção de datas
st.sidebar.header('Filtros de Data')
from_date = st.sidebar.date_input('Data Inicial', datetime(2025, 1, 1))
to_date = st.sidebar.date_input('Data Final', datetime(2025, 1, 22))

# Garantir que a data inicial não seja posterior à data final
if from_date > to_date:
    st.sidebar.error('A Data Inicial não pode ser posterior à Data Final.')

# Botão de busca
if st.button('Analisar Sentimentos'):
    queries = []
    if ticker_input:
        if validate_ticker(ticker_input):
            queries.append(ticker_input)
        else:
            st.error('Ticker inválido. Por favor, verifique e tente novamente.')
    if company_input:
        mapped_ticker = get_ticker(company_input)
        if mapped_ticker:
            queries.append(mapped_ticker)
        else:
            st.error('Nome da empresa inválido ou não encontrado.')

    if not queries:
        st.error('Por favor, insira pelo menos o Ticker ou o Nome da Empresa válido.')
    else:
        query = ' OR '.join(queries)
        api_key = 'sua chave'  # Substitua pela sua chave da NewsAPI
        # Converter datas para o formato YYYY-MM-DD
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        news = fetch_news(query, api_key, from_date_str, to_date_str)

        if news['status'] == 'ok' and news['totalResults'] > 0:
            sentiments = []
            dates = []
            for article in news['articles']:
                description = article['description'] or article['title']
                result = sentiment_pipeline(description)[0]
                label = result['label']
                sentiments.append(label)
                dates.append(article['publishedAt'][:10])  # Extrair apenas a data

            # Criar um DataFrame com os resultados
            df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Sentiment': sentiments
            })

            # Agregar os sentimentos por data
            sentiment_counts = df.groupby(['Date', 'Sentiment']).size().reset_index(name='Count')
            sentiment_pivot = sentiment_counts.pivot(index='Date', columns='Sentiment', values='Count').fillna(0)

            # Plotar o gráfico usando Plotly
            fig = px.bar(sentiment_pivot, 
                         x=sentiment_pivot.index, 
                         y=sentiment_pivot.columns,
                         title='Distribuição de Sentimentos por Data',
                         labels={'value': 'Número de Notícias', 'Date': 'Data'},
                         barmode='stack',
                         color_discrete_map={
                             'Positivo': 'green',
                             'Negativo': 'red',
                             'Neutro': 'gray'
                         })

            st.plotly_chart(fig)

            # Informações adicionais
            st.write(f"**Total de Notícias Analisadas:** {len(sentiments)}")

            # Opção para baixar os dados
            csv = sentiment_pivot.to_csv().encode('utf-8')
            st.download_button(
                label="Baixar Dados em CSV",
                data=csv,
                file_name='sentiment_by_date.csv',
                mime='text/csv',
            )
        else:
            st.info('Nenhuma notícia encontrada para os critérios fornecidos.')
