import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import requests
from googletrans import Translator
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_company_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        company_name = info.get("longName", "Nombre no disponible")
        description = info.get("longBusinessSummary", "Descripción no disponible para esta empresa.")
        
        # Traducción automática
        translator = Translator()
        description_es = translator.translate(description, src='en', dest='es').text
        
        return company_name, description_es
    except Exception as e:
        return None, f"Error al obtener datos: {str(e)}"

def get_stock_data(symbol, period="5y"):
    stock = yf.Ticker(symbol)
    return stock.history(period=period)

def plot_stock_price(symbol, period="5y", compare_symbol=None):
    df = get_stock_data(symbol, period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=symbol))
    
    # Agregar SMA 50 y SMA 200
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name=f'{symbol} SMA 50', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name=f'{symbol} SMA 200', line=dict(dash='dot')))
    
    # Si se desea comparar con otra empresa
    if compare_symbol:
        df_compare = get_stock_data(compare_symbol, period)
        fig.add_trace(go.Scatter(x=df_compare.index, y=df_compare['Close'], mode='lines', name=compare_symbol, line=dict(dash='solid', color='orange')))
    
    fig.update_layout(title=f'Histórico de {symbol}', xaxis_title='Fecha', yaxis_title='Precio')
    return fig

def get_stock_news_yahoo(symbol):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        news = []
        for item in data.get("news", [])[:5]:  # Tomar las 5 noticias más recientes
            news.append({"title": item["title"], "url": item["link"]})
        return news
    else:
        return [{"title": "No se pudieron obtener noticias", "url": "#"}]

def financial_analysis(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    
    revenue = info.get("totalRevenue", None)
    net_income = info.get("netIncome", None)
    debt = info.get("totalDebt", None)
    market_cap = info.get("marketCap", None)
    
    roi = None
    if revenue and net_income:
        roi = round((net_income / revenue) * 100, 2)
    
    return {
        "Ingresos Totales": revenue,
        "Utilidad Neta": net_income,
        "Deuda Total": debt,
        "Capitalización de Mercado": market_cap,
        "ROI (%)": roi
    }

def predict_stock_price(symbol):
    df = get_stock_data(symbol, period="2y")
    df = df.reset_index()
    df['Days'] = np.arange(len(df))
    
    model = LinearRegression()
    model.fit(df[['Days']], df['Close'])
    
    future_days = np.array([len(df) + 30]).reshape(-1, 1)
    predicted_price = model.predict(future_days)[0]
    
    return round(predicted_price, 2)

# Configuración de la página
st.set_page_config(page_title="📈 Análisis de Acciones", layout="wide")

st.title("🔍 Análisis de Empresas en Yahoo Finance")

symbol = st.text_input("Ingrese el símbolo de la acción (Ej: AAPL, TSLA, MSFT)", "", key="symbol_input")
compare_symbol = st.text_input("Ingrese el símbolo de otra acción para comparar (opcional)", "", key="compare_symbol")

# Seleccionar el período de tiempo para la gráfica
period = st.selectbox("Seleccione el período de tiempo", ["1y", "2y", "3y", "5y"], index=3)

if symbol:
    company_name, description = get_company_info(symbol)
    if company_name:
        st.subheader(company_name)
        st.write("**Descripción de la empresa:**")
        st.write(description)
    else:
        st.error(description)
    
    # Mostrar gráfico de precios
    st.plotly_chart(plot_stock_price(symbol, period, compare_symbol))
    
    st.subheader("📊 Análisis Financiero")
    analysis = financial_analysis(symbol)
    for key, value in analysis.items():
        st.write(f"**{key}:** {value}")
    
    predicted_price = predict_stock_price(symbol)
    st.subheader("📈 Predicción de Precio a 30 días")
    st.write(f"Precio estimado: ${predicted_price}")
    
    st.subheader("📰 Noticias Recientes")
    news_articles = get_stock_news_yahoo(symbol)
    for article in news_articles:
        st.markdown(f"[{article['title']}]({article['url']})")
