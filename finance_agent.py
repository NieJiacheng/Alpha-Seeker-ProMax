
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import praw
import yfinance as yf
import numpy as np
import json
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# ==========================================

st.set_page_config(page_title="AlphaSeeker Pro Max - AI Agent", layout="wide", page_icon="🏦")

# Session State
if 'symbol' not in st.session_state:
    st.session_state.symbol = "NVDA"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_view_symbol' not in st.session_state:
    st.session_state.current_view_symbol = "NVDA"
if 'persona' not in st.session_state:
    st.session_state.persona = "Professional Analyst" 
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = ""  
if 'sector_recommendation' not in st.session_state:
    st.session_state.sector_recommendation = None 

# --- API Keys ---
DEEPSEEK_API_KEY = "YourDeepSeekAPIKey"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat" # deepseek-chat
FINNHUB_API_KEY = "YourFinhubAPIKey"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Reddit Config
REDDIT_CONFIG = {
    "client_id": "YourClientID",
    "client_secret": "YourClientSecret",
    "user_agent": "YourAgentName",
    "username": "YourRedditName",
    "password": "YourRedditPassword"
}

# CSS 
st.markdown("""
<style>
    .market-card {background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); padding: 15px; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .market-card-down {background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);}
    .market-card-up {background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);}
    .metric-value {font-size: 28px; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);}
    .report-box {background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; border-left: 5px solid #4834d4;}
    .rec-card {background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .rec-buy {border-left: 5px solid #00b894;}
    .rec-sell {border-left: 5px solid #ff7675;}
    .rec-hold {border-left: 5px solid #fdcb6e;}
    .stButton button {width: 100%; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# ==========================================

@st.cache_resource
def init_resources():
    res = {}
    try: res['llm'] = ChatOpenAI(model=DEEPSEEK_MODEL, openai_api_key=DEEPSEEK_API_KEY, openai_api_base=DEEPSEEK_BASE_URL, temperature=0.3)
    except: res['llm'] = None
    try: res['reddit'] = praw.Reddit(**REDDIT_CONFIG, request_timeout=10.0)
    except: res['reddit'] = None
    res['vader'] = SentimentIntensityAnalyzer()
    return res

RESOURCES = init_resources()

def generate_mock_data(symbol, days=180):
    dates = pd.date_range(end=datetime.now(), periods=days)
    base_price = 100
    prices = [base_price]
    for _ in range(days-1):
        prices.append(max(prices[-1] + np.random.normal(0, 2), 1))
    df = pd.DataFrame({
        'Date': dates, 'Close': prices,
        'Open': [p+np.random.normal(0,1) for p in prices],
        'High': [p+2 for p in prices], 'Low': [p-2 for p in prices],
        'Volume': np.random.randint(1000,10000, days)
    })
    return df

@st.cache_data(ttl=600)
def get_market_indices_robust():
    indices = {'S&P 500': '^GSPC', 'Nasdaq': '^IXIC', 'VIX': '^VIX', 'Bitcoin': 'BTC-USD'}
    result = {}
    for name, ticker in indices.items():
        try:
            df = yf.download(ticker, period="5d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) >= 2:
                curr, prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
                curr = float(curr.item()) if hasattr(curr, 'item') else curr
                prev = float(prev.item()) if hasattr(prev, 'item') else prev
                result[name] = {'price': curr, 'change': ((curr-prev)/prev)*100}
            else: raise ValueError
        except:
            result[name] = {'price': 0, 'change': 0, 'mock': True}
    return result


# ==========================================
# ==========================================

def generate_full_report(symbol, quote, financials, news_list, sentiment_score, hist_df):
    if not RESOURCES['llm']:
        return "⚠️ LLM service is not connected, report generation failed." #

    news_context = "\n".join([f"- {n['headline']}" for n in news_list[:5]])
    tech_trend = "Bullish" if hist_df['Close'].iloc[-1] > hist_df['SMA_50'].iloc[-1] else "Bearish" #
    
    prompt = f"""
    Please write a professional investment research report for the stock {symbol}.
    
    [Input Data] #
    1. Current Price: {quote.get('c')} (Daily Change: {quote.get('dp')}%) #
    2. Financial Data: PE(TTM)={financials.get('peBasicExclExtraTTM', 'N/A')}, EPS={financials.get('epsExclExtraItemsTTM', 'N/A')}, Beta={financials.get('beta', 'N/A')} #
    3. Recent News Headlines: #
    {news_context}
    4. Social Media Sentiment Score (-1 to 1): {sentiment_score:.2f} #
    5. Technical Trend: Currently {'Above' if tech_trend=='Bullish' else 'Below'} the 50-day moving average. #

    [Report Requirements] #
    Please use Markdown format, including the following sections:
    1. **📊 Executive Summary**: A one-sentence summary of the current investment opportunity. #
    2. **📰 News and Public Opinion**: Analysis of the impact of news and Reddit sentiment on the stock price. #
    3. **🧬 Fundamental Snapshot**: Evaluation of the valuation level and financial health. #
    4. **📈 Technical Outlook**: Provide a short-term forecast based on moving averages and trends. #
    5. **💡 Final Investment Recommendation**: Explicitly give a rating of "Strong Buy", "Buy", "Hold", or "Sell", and explain the reason. #
    
    Maintain a professional and objective tone, with a word count limit of 600 words. #
    """
    
    try:
        response = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
        return response
    except Exception as e:
        return f"Report generation failed: {str(e)}" #

def get_sector_tickers(sector_name):
    predefined = {
        "AI & Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "AVGO"], #
        "Tech Giants (Mag 7)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"], #
        "Electric Vehicles (EV)": ["TSLA", "RIVN", "LCID", "NIO", "XPEV"], #
        "Biopharma": ["LLY", "NVO", "PFE", "MRK", "JNJ"], #
        "Crypto Related": ["COIN", "MSTR", "MARA", "RIOT"] #
    }
    
    if sector_name in predefined:
        return predefined[sector_name]
    
    if RESOURCES['llm']:
        prompt = f"Please list 5 US-listed stock tickers belonging to the '{sector_name}' sector. Return only the tickers, separated by commas, with no other text. For example: AAPL, MSFT" #
        try:
            resp = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
            tickers = [t.strip().upper() for t in resp.split(',') if t.strip().isalpha()]
            return tickers[:5]
        except: return []
    return []

def analyze_sector_recommendation(sector_name, tickers):
    if not tickers: return "No stocks found for this sector." 
    
    data_summary = []
    for t in tickers:
        try:
            q = requests.get(f"{FINNHUB_BASE_URL}/quote", params={'symbol': t, 'token': FINNHUB_API_KEY}).json()
            m = requests.get(f"{FINNHUB_BASE_URL}/stock/metric", params={'symbol': t, 'metric': 'all', 'token': FINNHUB_API_KEY}).json().get('metric', {})
            data_summary.append({
                "symbol": t,
                "price": q.get('c'),
                "change": q.get('dp'),
                "pe": m.get('peBasicExclExtraTTM', 0),
                "high52": m.get('52WeekHigh', 0)
            })
        except: continue
        
    if not RESOURCES['llm']: return "LLM service is not connected."

    # Prompt
    data_str = "\n".join([f"{d['symbol']}: Price ${d['price']}, Change {d['change']}%, PE={d['pe']}" for d in data_summary])
    prompt = f"""
    You are a seasoned fund manager. The user wants to know about investment opportunities in the '{sector_name}' sector. #
    
    Here is the real-time data for several representative stocks in this sector: #
    {data_str}
    
    Please output an investment recommendation table.
    For each stock:
    1. Give a recommendation of "Buy", "Sell", or "Hold". #
    2. Explain the reason in one sentence (combining valuation, momentum, or your knowledge of the company). #
    3. Give a "Recommendation Index" (1-5 stars). #
    
    Return a list in JSON format, with fields: symbol, action, reason, stars (integer).
    Do not output any text other than the JSON. #
    """
    
    try:
        resp = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
        if "```json" in resp: resp = resp.split("```json")[1].split("```")[0]
        elif "```" in resp: resp = resp.split("```")[1].split("```")[0]
        return json.loads(resp)
    except Exception as e:
        return f"Analysis failed: {str(e)}" 

@st.cache_data(ttl=1800)
def get_stock_history_enhanced(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        clean_cols = {}
        for c in df.columns:
            if 'date' in str(c).lower(): clean_cols[c] = 'Date'
            elif 'close' in str(c).lower(): clean_cols[c] = 'Close'
            elif 'open' in str(c).lower(): clean_cols[c] = 'Open'
            elif 'high' in str(c).lower(): clean_cols[c] = 'High'
            elif 'low' in str(c).lower(): clean_cols[c] = 'Low'
            elif 'volume' in str(c).lower(): clean_cols[c] = 'Volume'
        df = df.rename(columns=clean_cols)
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
        if df.empty or 'Close' not in df.columns: raise ValueError
        return df, False
    except:
        return generate_mock_data(symbol), True

@st.cache_data(ttl=3600)
def get_finnhub_news(symbol):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news = requests.get(f"{FINNHUB_BASE_URL}/company-news", 
                          params={'symbol': symbol, 'from': start, 'to': end, 'token': FINNHUB_API_KEY}).json()
        return news[:10] 
    except:
        return []

@st.cache_data(ttl=300)
def get_reddit_sentiment(symbol):
    posts_data = []
    sentiment_score = 0
    count = 0
    
    if not RESOURCES['reddit']:
        return 0, []

    try:
        for submission in RESOURCES['reddit'].subreddit("stocks+wallstreetbets+investing").search(symbol, limit=20, time_filter="week"):
            title = submission.title
            score = RESOURCES['vader'].polarity_scores(title)['compound']
            sentiment_score += score
            count += 1
            posts_data.append({
                "title": title,
                "score": score,
                "url": submission.url,
                "created": datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d')
            })
    except Exception as e:
        print(f"Reddit Error: {e}")
        return 0, []
        
    avg_sentiment = sentiment_score / count if count > 0 else 0
    return avg_sentiment, posts_data

def get_filtered_peers(symbol):
    peers_list = []
    try:
        r = requests.get(f"{FINNHUB_BASE_URL}/stock/peers", params={'symbol': symbol, 'token': FINNHUB_API_KEY})
        tickers = [t for t in r.json() if t != symbol and "." not in t and len(t) <= 5 and t.isalpha()]
        for t in tickers[:4]:
            try:
                prof = requests.get(f"{FINNHUB_BASE_URL}/stock/profile2", params={'symbol': t, 'token': FINNHUB_API_KEY}).json()
                quote = requests.get(f"{FINNHUB_BASE_URL}/quote", params={'symbol': t, 'token': FINNHUB_API_KEY}).json()
                if prof and quote:
                    peers_list.append({
                        "ticker": t, "name": prof.get('name', t), "logo": prof.get('logo', ''),
                        "price": quote.get('c', 0), "change": quote.get('dp', 0)
                    })
            except: continue
    except: pass
    return peers_list

def get_basic_financials(symbol):
    try:
        metric = requests.get(f"{FINNHUB_BASE_URL}/stock/metric", params={'symbol': symbol, 'metric': 'all', 'token': FINNHUB_API_KEY}).json()
        return metric.get('metric', {})
    except: return {}

# ==========================================
# ==========================================

def plot_advanced_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')) 
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='gray', width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(128,128,128,0.2)', name='Bollinger Bands')) 
    fig.update_layout(title=f"{symbol} Trend Analysis", height=450, xaxis_rangeslider_visible=False, template="plotly_white") 
    return fig

def plot_radar_fundamentals(quote, profile, metrics):
    pe = metrics.get('peBasicExclExtraTTM', 20)
    beta = metrics.get('beta', 1)
    
    scores = {
        'Low Valuation': max(0, min(100, 100 - pe if pe else 50)),
        'Growth': 85, 
        'Profitability': max(0, min(100, (metrics.get('netProfitMarginTTM', 0) * 2 + 50))), 
        'Market Momentum': max(0, min(100, 50 + metrics.get('52WeekPriceReturnDaily', 0))), 
        'Safety': max(0, min(100, 100 - (beta * 20))) 
    }
    
    fig = go.Figure(go.Scatterpolar(r=list(scores.values()), theta=list(scores.keys()), fill='toself', line_color='#4834d4'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, margin=dict(t=30, b=20))
    return fig

# ==========================================
# ==========================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("🎮 Control Panel") 
        st.write("Current AI Persona:") 
        persona = st.selectbox("", ["Professional Analyst", "Wall Street Bets (WSB)", "Warren Buffett"], index=0) 
        st.session_state.persona = persona
        st.markdown("---")
        st.info("💡 Tip: Go to the 'Smart Sector Picks' tab to enter a sector and let the AI select stocks for you.") 

    # --- Header ---
    st.markdown("### 🌍 Global Market Overview") 
    indices = get_market_indices_robust()
    cols = st.columns(4)
    for i, (name, data) in enumerate(indices.items()):
        bg = "market-card-up" if data['change'] >= 0 else "market-card-down"
        arrow = "▲" if data['change'] >= 0 else "▼"
        with cols[i]:
            st.markdown(f"""<div class="market-card {bg}" style="color: #333;">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{data['price']:,.2f}</div>
                <div style="font-weight:bold;">{arrow} {data['change']:.2f}%</div></div>""", unsafe_allow_html=True)
    st.markdown("---")

    # --- Search & Trigger ---
    c1, c2 = st.columns([3, 1])
    with c1: st.title("🚀 AlphaSeeker Pro Max")
    with c2:
        symbol_input = st.text_input("🔍 Stock Ticker (US Stocks Only):", value=st.session_state.symbol).upper() 
        start_scan = st.button("🚀 Start Deep Scan & Generate Report", type="primary") 

    if start_scan:
        st.session_state.symbol = symbol_input
        st.session_state.chat_history = [] 
        st.session_state.analysis_report = "" 
        st.rerun()

    symbol = st.session_state.symbol
    
    # --- Data Fetching ---
    with st.spinner(f"Scanning {symbol} data and generating report across the web..."): 
        try:
            quote = requests.get(f"{FINNHUB_BASE_URL}/quote", params={'symbol': symbol, 'token': FINNHUB_API_KEY}).json()
            profile = requests.get(f"{FINNHUB_BASE_URL}/stock/profile2", params={'symbol': symbol, 'token': FINNHUB_API_KEY}).json()
        except: quote, profile = {}, {}
        
        hist_df, is_mock = get_stock_history_enhanced(symbol)
        financials = get_basic_financials(symbol)
        financial_metrics = get_basic_financials(symbol)
        news_list = get_finnhub_news(symbol)
        sentiment_score, reddit_posts = get_reddit_sentiment(symbol)
        peers = get_filtered_peers(symbol)

        if start_scan or not st.session_state.analysis_report:
            report = generate_full_report(symbol, quote, financials, news_list, sentiment_score, hist_df)
            st.session_state.analysis_report = report

    # --- UI Layout ---
    st.markdown(f"""
    <div style="background:#fff; padding:20px; border-radius:10px; border-left:5px solid #0984e3; box-shadow:0 2px 5px rgba(0,0,0,0.05); margin-top: 10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center; gap:15px;">
                <img src="{profile.get('logo','')}" style="width:60px; height:60px; border-radius:50%; object-fit:contain;" onerror="this.style.display='none'">
                <div>
                    <h1 style="margin:0;">{symbol} <span style="font-size:0.5em; color:gray;">{profile.get('name', 'Unknown')}</span></h1>
                    <p style="color:#666; margin:0;">{profile.get('finnhubIndustry', '-')} | {profile.get('exchange', '-')}</p>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:2.5em; font-weight:bold; color:{'#00b894' if quote.get('dp',0)>0 else '#ff7675'}">${quote.get('c', 0)}</div>
                <div>{quote.get('dp', 0)}% (Today)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("###")

    # Tabs
    t_report, t_chart, t2, t3, t4, t_rec, t_chat = st.tabs(["📝 Deep Dive Report", "📈 Market Data",  "🧬 Fundamentals & Financials", "🔥 Sentiment & News", "⚔️ Peer Comparison", "🎯 Smart Sector Picks", "🤖 AI Q&A"]) #
 

    # Tab 1
    with t_report:
        st.subheader(f"📄 {symbol} Investment Analysis Report (AI Generated)") #
        if st.session_state.analysis_report:
            st.markdown(f'<div class="report-box">{st.session_state.analysis_report}</div>', unsafe_allow_html=True)
        else:
            st.info("Please click 'Start Deep Scan' above to generate the report.") #
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔥 Community Sentiment") #
            st.metric("Reddit Sentiment Score", f"{sentiment_score:.2f}", delta="Bullish" if sentiment_score > 0 else "Bearish") #
        with c2:
            st.markdown("#### 📰 Latest News") #
            for n in news_list[:3]:
                st.markdown(f"- [{n['headline']}]({n['url']})")

    # Tab 2
    with t_chart:
        if not hist_df.empty:
            st.plotly_chart(plot_advanced_chart(hist_df, symbol), use_container_width=True)
        
        st.subheader("Core Financials") #
        cols = st.columns(4)
        cols[0].metric("P/E (TTM)", f"{financials.get('peBasicExclExtraTTM', 0):.2f}")
        cols[1].metric("EPS", f"{financials.get('epsExclExtraItemsTTM', 0):.2f}")
        cols[2].metric("Beta", f"{financials.get('beta', 0):.2f}")
        cols[3].metric("52-Week High", f"{financials.get('52WeekHigh', 0):.2f}") #
    with t2:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Capability Radar") #
            st.plotly_chart(plot_radar_fundamentals(quote, profile, financial_metrics), use_container_width=True)
        with c2: 
            st.subheader("📊 Core Financial Metrics") #
            if financial_metrics:
                m_cols = st.columns(3)
                m_cols[0].metric("P/E Ratio (TTM)", f"{financial_metrics.get('peBasicExclExtraTTM', 0):.2f}") #
                m_cols[1].metric("EPS", f"{financial_metrics.get('epsExclExtraItemsTTM', 0):.2f}") #
                m_cols[2].metric("Beta", f"{financial_metrics.get('beta', 0):.2f}")
                
                m_cols2 = st.columns(3)
                m_cols2[0].metric("52-Week High", f"{financial_metrics.get('52WeekHigh', 0):.2f}") #
                m_cols2[1].metric("52-Week Low", f"{financial_metrics.get('52WeekLow', 0):.2f}") #
                m_cols2[2].metric("Dividend Yield", f"{financial_metrics.get('dividendYieldIndicatedAnnual', 0):.2f}%") #
            else:
                st.info("No detailed financial data available.") #

    # Tab 3
    with t3:
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader(f"🗣️ Reddit Retail Sentiment") #
            if sentiment_score > 0.05:
                sent_color, sent_text = "#00b894", "Bullish" #
            elif sentiment_score < -0.05:
                sent_color, sent_text = "#ff7675", "Bearish" #
            else:
                sent_color, sent_text = "#fab1a0", "Neutral" #
            
            st.markdown(f"### Sentiment Index: <span style='color:{sent_color}'>{sent_text} ({sentiment_score:.2f})</span>", unsafe_allow_html=True) #
            
            st.markdown("#### Latest Hot Posts") #
            if reddit_posts:
                for post in reddit_posts[:5]:
                    score_emoji = "🟢" if post['score'] > 0 else "🔴" if post['score'] < 0 else "⚪"
                    st.markdown(f"""
                    <div style="font-size:0.9em; border-bottom:1px solid #eee; padding:5px;">
                        {score_emoji} <a href="{post['url']}" target="_blank" style="text-decoration:none; color:#333;">{post['title']}</a>
                        <div style="color:#888; font-size:0.8em;">{post['created']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Not connected to Reddit API or no relevant discussions.") #

        with c2:
            st.subheader("📰 Latest Financial News") #
            if news_list:
                for news in news_list:
                    st.markdown(f"""
                    <div class="news-card">
                        <div style="font-weight:bold;">{news.get('headline')}</div>
                        <div style="font-size:0.85em; color:#555; margin:5px 0;">{news.get('summary', '')[:100]}...</div>
                        <div style="font-size:0.8em; color:#888;">{news.get('source')} - {datetime.fromtimestamp(news.get('datetime')).strftime('%m-%d %H:%M')}</div>
                        <a href="{news.get('url')}" target="_blank" style="font-size:0.8em;">Read Full Article</a>
                    </div>
                    """, unsafe_allow_html=True) #
            else:
                st.info("No latest news available.") #

    # Tab 4
    with t4:
        st.subheader(f"⚔️ {symbol}'s Main Competitors") #
        if peers:
            cols = st.columns(4)
            for i, p in enumerate(peers):
                with cols[i % 4]:
                    color = "#00b894" if p['change'] > 0 else "#ff7675"
                    st.markdown(f"""
                    <div class="competitor-card">
                        <div style="font-weight:bold; font-size:1.2em;">{p['ticker']}</div>
                        <div style="font-size:0.8em; color:#666; height:20px; overflow:hidden;">{p['name'][:15]}</div>
                        <div style="margin:5px 0; font-weight:bold;">${p['price']} <span style="color:{color}">({p['change']}%)</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Analyze {p['ticker']}", key=f"btn_{p['ticker']}"): #
                        st.session_state.symbol = p['ticker']
                        st.rerun()
        else:
            st.info("No suitable competitors found.") #
    # Tab 3
    with t_rec:
        st.subheader("🎯 Sector Scan and Stock Picks") #
        st.write("Select a sector to let the AI scan its leading stocks and provide trading recommendations.") #
        
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            sector_input = st.selectbox("Select or Enter Sector:", #
                                      ["AI & Semiconductors", "Tech Giants (Mag 7)", "Electric Vehicles (EV)", "Biopharma", "Crypto Related", "Custom..."]) #
            if sector_input == "Custom...": #
                custom_sector = st.text_input("Enter Sector Name (e.g., Quantum Computing, Aerospace):") #
                if custom_sector: sector_input = custom_sector
        
        with col_btn:
            st.write("") # Spacer
            st.write("")
            btn_analyze_sector = st.button("🔍 Analyze Sector") #
            
        if btn_analyze_sector:
            with st.spinner(f"Analyzing {sector_input} sector..."): #
                tickers = get_sector_tickers(sector_input)
                if tickers:
                    st.write(f"Scanning: {', '.join(tickers)}") #
                    rec_result = analyze_sector_recommendation(sector_input, tickers)
                    st.session_state.sector_recommendation = rec_result
                else:
                    st.error("Could not identify the sector or find relevant stocks.") #
        
        # 
        if st.session_state.sector_recommendation and isinstance(st.session_state.sector_recommendation, list):
            st.markdown("### 📋 AI Trading Recommendation Letter") #
            rec_cols = st.columns(3)
            for i, rec in enumerate(st.session_state.sector_recommendation):
                action = rec.get('action', 'Hold')
                style_class = "rec-buy" if "Buy" in action or "买入" in action else "rec-sell" if "Sell" in action or "卖出" in action else "rec-hold" #
                stars = "⭐" * int(rec.get('stars', 3))
                
                with rec_cols[i % 3]:
                    st.markdown(f"""
                    <div class="rec-card {style_class}">
                        <h3>{rec['symbol']}</h3>
                        <div style="font-weight:bold; font-size:1.2em; color:#333;">{action}</div>
                        <div style="color:#f1c40f;">{stars}</div>
                        <p style="color:#666; font-size:0.9em; margin-top:5px;">{rec['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Tab 4
    with t_chat:
        st.subheader(f"🤖 Ask the AI ({st.session_state.persona})") #
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                    
        if user_input := st.chat_input(f"What else would you like to know about {symbol}?"): #
            with st.chat_message("user"): st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            if RESOURCES['llm']:
                
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                sys_prompt = f"""
                You are a {st.session_state.persona}. 
                You are currently analyzing {symbol}. 
                Crucially, **THE CURRENT DATE is {current_date}**. 
                Use this context for any time-sensitive questions and reference data up to this date.
                """
                # -----------------------------------------------
                
                full_context = [SystemMessage(content=sys_prompt), HumanMessage(content=user_input)]
                with st.chat_message("assistant"):
                    response = st.write_stream(RESOURCES['llm'].stream(full_context))
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()