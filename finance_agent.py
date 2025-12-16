# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import requests
# import praw
# import yfinance as yf
# import numpy as np
# import json
# from datetime import datetime, timedelta
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

# # ==========================================
# # 1. 基础配置与 Session State 初始化
# # ==========================================

# st.set_page_config(page_title="AlphaSeeker Pro Max - AI Agent", layout="wide", page_icon="🏦")

# # 初始化 Session State
# if 'symbol' not in st.session_state:
#     st.session_state.symbol = "NVDA"
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'current_view_symbol' not in st.session_state:
#     st.session_state.current_view_symbol = "NVDA"
# if 'persona' not in st.session_state:
#     st.session_state.persona = "专业分析师"
# if 'analysis_report' not in st.session_state:
#     st.session_state.analysis_report = ""  # 存储生成的研报
# if 'sector_recommendation' not in st.session_state:
#     st.session_state.sector_recommendation = None # 存储荐股结果

# # --- API Keys ---
# DEEPSEEK_API_KEY = "sk-cafba043052344568e72f6d9be865c7d"
# DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# DEEPSEEK_MODEL = "deepseek-chat" # 或 deepseek-chat
# FINNHUB_API_KEY = "d46s26hr01qgc9euamk0d46s26hr01qgc9euamkg"
# FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# # Reddit Config
# REDDIT_CONFIG = {
#     "client_id": "kpCP6k3-q3wWb0UGhGgk-w",
#     "client_secret": "3JXSxS2Rs32EzEr09Ywd-XiWCpUCvA",
#     "user_agent": "Financial_AI_Agent_V1",
#     "username": "ParamedicRelative368",
#     "password": "Tsm@928501"
# }

# # CSS 样式
# st.markdown("""
# <style>
#     .market-card {background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); padding: 15px; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
#     .market-card-down {background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);}
#     .market-card-up {background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);}
#     .metric-value {font-size: 28px; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);}
#     .report-box {background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; border-left: 5px solid #4834d4;}
#     .rec-card {background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
#     .rec-buy {border-left: 5px solid #00b894;}
#     .rec-sell {border-left: 5px solid #ff7675;}
#     .rec-hold {border-left: 5px solid #fdcb6e;}
#     .stButton button {width: 100%; border-radius: 5px;}
# </style>
# """, unsafe_allow_html=True)

# # ==========================================
# # 2. 核心资源与数据获取
# # ==========================================

# @st.cache_resource
# def init_resources():
#     res = {}
#     try: res['llm'] = ChatOpenAI(model=DEEPSEEK_MODEL, openai_api_key=DEEPSEEK_API_KEY, openai_api_base=DEEPSEEK_BASE_URL, temperature=0.3)
#     except: res['llm'] = None
#     try: res['reddit'] = praw.Reddit(**REDDIT_CONFIG, request_timeout=10.0)
#     except: res['reddit'] = None
#     res['vader'] = SentimentIntensityAnalyzer()
#     return res

# RESOURCES = init_resources()

# def generate_mock_data(symbol, days=180):
#     dates = pd.date_range(end=datetime.now(), periods=days)
#     base_price = 100
#     prices = [base_price]
#     for _ in range(days-1):
#         prices.append(max(prices[-1] + np.random.normal(0, 2), 1))
#     df = pd.DataFrame({
#         'Date': dates, 'Close': prices,
#         'Open': [p+np.random.normal(0,1) for p in prices],
#         'High': [p+2 for p in prices], 'Low': [p-2 for p in prices],
#         'Volume': np.random.randint(1000,10000, days)
#     })
#     return df

# @st.cache_data(ttl=600)
# def get_market_indices_robust():
#     indices = {'S&P 500': '^GSPC', 'Nasdaq': '^IXIC', 'VIX': '^VIX', 'Bitcoin': 'BTC-USD'}
#     result = {}
#     for name, ticker in indices.items():
#         try:
#             df = yf.download(ticker, period="5d", progress=False)
#             if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
#             if len(df) >= 2:
#                 curr, prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
#                 curr = float(curr.item()) if hasattr(curr, 'item') else curr
#                 prev = float(prev.item()) if hasattr(prev, 'item') else prev
#                 result[name] = {'price': curr, 'change': ((curr-prev)/prev)*100}
#             else: raise ValueError
#         except:
#             result[name] = {'price': 0, 'change': 0, 'mock': True}
#     return result

# @st.cache_data(ttl=1800)
# def get_stock_history_enhanced(symbol):
#     try:
#         df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
#         if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
#         df = df.reset_index()
#         clean_cols = {}
#         for c in df.columns:
#             if 'date' in str(c).lower(): clean_cols[c] = 'Date'
#             elif 'close' in str(c).lower(): clean_cols[c] = 'Close'
#             elif 'open' in str(c).lower(): clean_cols[c] = 'Open'
#             elif 'high' in str(c).lower(): clean_cols[c] = 'High'
#             elif 'low' in str(c).lower(): clean_cols[c] = 'Low'
#             elif 'volume' in str(c).lower(): clean_cols[c] = 'Volume'
#         df = df.rename(columns=clean_cols)
#         if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
#         df['SMA_20'] = df['Close'].rolling(window=20).mean()
#         df['SMA_50'] = df['Close'].rolling(window=50).mean()
#         df['BB_Mid'] = df['Close'].rolling(window=20).mean()
#         df['BB_Std'] = df['Close'].rolling(window=20).std()
#         df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
#         df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
#         if df.empty or 'Close' not in df.columns: raise ValueError
#         return df, False
#     except:
#         return generate_mock_data(symbol), True

# @st.cache_data(ttl=3600)
# def get_finnhub_news(symbol):
#     try:
#         end = datetime.now().strftime('%Y-%m-%d')
#         start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
#         news = requests.get(f"{FINNHUB_BASE_URL}/company-news", 
#                           params={'symbol': symbol, 'from': start, 'to': end, 'token': FINNHUB_API_KEY}).json()
#         return news[:8]
#     except: return []

# @st.cache_data(ttl=300)
# def get_reddit_sentiment(symbol):
#     posts_data = []
#     sentiment_score = 0
#     count = 0
#     if not RESOURCES['reddit']: return 0, []
#     try:
#         for submission in RESOURCES['reddit'].subreddit("stocks+wallstreetbets+investing").search(symbol, limit=15, time_filter="week"):
#             title = submission.title
#             score = RESOURCES['vader'].polarity_scores(title)['compound']
#             sentiment_score += score
#             count += 1
#             posts_data.append({"title": title, "score": score, "url": submission.url})
#     except: return 0, []
#     avg_sentiment = sentiment_score / count if count > 0 else 0
#     return avg_sentiment, posts_data

# def get_basic_financials(symbol):
#     try:
#         metric = requests.get(f"{FINNHUB_BASE_URL}/stock/metric", params={'symbol': symbol, 'metric': 'all', 'token': FINNHUB_API_KEY}).json()
#         return metric.get('metric', {})
#     except: return {}

# # ==========================================
# # 3. 新增业务逻辑：自动研报 & 行业荐股
# # ==========================================

# def generate_full_report(symbol, quote, financials, news_list, sentiment_score, hist_df):
#     """调用 LLM 生成深度研报"""
#     if not RESOURCES['llm']:
#         return "⚠️ LLM 服务未连接，无法生成研报。"

#     # 准备上下文数据
#     news_context = "\n".join([f"- {n['headline']}" for n in news_list[:5]])
#     tech_trend = "看涨" if hist_df['Close'].iloc[-1] > hist_df['SMA_50'].iloc[-1] else "看跌"
    
#     prompt = f"""
#     请为股票 {symbol} 撰写一份专业的投资研究报告。
    
#     【输入数据】
#     1. 当前价格: {quote.get('c')} (日涨跌: {quote.get('dp')}%)
#     2. 财务数据: PE(TTM)={financials.get('peBasicExclExtraTTM', 'N/A')}, EPS={financials.get('epsExclExtraItemsTTM', 'N/A')}, Beta={financials.get('beta', 'N/A')}
#     3. 近期新闻头条:
#     {news_context}
#     4. 社交媒体情绪分(-1到1): {sentiment_score:.2f}
#     5. 技术面趋势: 目前位于50日均线之{'上' if tech_trend=='看涨' else '下'}。

#     【报告要求】
#     请用 Markdown 格式，包含以下章节：
#     1. **📊 核心摘要**: 一句话概括当前投资机会。
#     2. **📰 消息面与舆情**: 分析新闻和Reddit情绪对股价的影响。
#     3. **🧬 基本面快照**: 评价估值水平和财务健康度。
#     4. **📈 技术面展望**: 基于均线和趋势给出短期预测。
#     5. **💡 最终投资建议**: 明确给出“强力买入”、“买入”、“持有”或“卖出”评级，并说明理由。
    
#     保持专业、客观，字数控制在 600 字以内。
#     """
    
#     try:
#         response = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
#         return response
#     except Exception as e:
#         return f"生成报告失败: {str(e)}"

# def get_sector_tickers(sector_name):
#     """根据行业名称获取代表性股票代码"""
#     # 常用行业硬编码，自定义行业通过 LLM 获取
#     predefined = {
#         "AI & 半导体": ["NVDA", "AMD", "INTC", "TSM", "AVGO"],
#         "科技巨头 (Mag 7)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"],
#         "电动汽车 (EV)": ["TSLA", "RIVN", "LCID", "NIO", "XPEV"],
#         "生物医药": ["LLY", "NVO", "PFE", "MRK", "JNJ"],
#         "加密货币相关": ["COIN", "MSTR", "MARA", "RIOT"]
#     }
    
#     if sector_name in predefined:
#         return predefined[sector_name]
    
#     # 如果是用户自定义输入（如“量子计算”），让 LLM 推荐
#     if RESOURCES['llm']:
#         prompt = f"请列出 5 个属于 '{sector_name}' 行业的美国上市公司股票代码。只返回代码，用逗号分隔，不要其他文字。例如: AAPL, MSFT"
#         try:
#             resp = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
#             tickers = [t.strip().upper() for t in resp.split(',') if t.strip().isalpha()]
#             return tickers[:5]
#         except: return []
#     return []

# def analyze_sector_recommendation(sector_name, tickers):
#     """分析行业并给出推荐"""
#     if not tickers: return "未找到该行业的股票。"
    
#     # 批量获取简要数据
#     data_summary = []
#     for t in tickers:
#         try:
#             q = requests.get(f"{FINNHUB_BASE_URL}/quote", params={'symbol': t, 'token': FINNHUB_API_KEY}).json()
#             m = requests.get(f"{FINNHUB_BASE_URL}/stock/metric", params={'symbol': t, 'metric': 'all', 'token': FINNHUB_API_KEY}).json().get('metric', {})
#             data_summary.append({
#                 "symbol": t,
#                 "price": q.get('c'),
#                 "change": q.get('dp'),
#                 "pe": m.get('peBasicExclExtraTTM', 0),
#                 "high52": m.get('52WeekHigh', 0)
#             })
#         except: continue
        
#     if not RESOURCES['llm']: return "LLM 服务未连接。"

#     # 构建 Prompt
#     data_str = "\n".join([f"{d['symbol']}: 价格${d['price']}, 涨跌{d['change']}%, PE={d['pe']}" for d in data_summary])
#     prompt = f"""
#     你是一位资深基金经理。用户想了解 '{sector_name}' 行业的投资机会。
    
#     以下是该行业几只代表性股票的实时数据：
#     {data_str}
    
#     请输出一份投资建议表。
#     对于每一只股票：
#     1. 给出“买入”、“卖出”或“持有”的建议。
#     2. 用一句话解释理由（结合估值、动量或你对该公司的了解）。
#     3. 给出一个“推荐指数”（1-5星）。
    
#     请以 JSON 格式返回列表，字段为: symbol, action, reason, stars(整数)。
#     不要输出 JSON 以外的文字。
#     """
    
#     try:
#         resp = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
#         # 清理可能的 markdown 标记
#         if "```json" in resp: resp = resp.split("```json")[1].split("```")[0]
#         elif "```" in resp: resp = resp.split("```")[1].split("```")[0]
#         return json.loads(resp)
#     except Exception as e:
#         return f"分析失败: {str(e)}"

# @st.cache_data(ttl=1800)
# def get_stock_history_enhanced(symbol):
#     """K线获取 + 技术指标计算"""
#     try:
#         df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
#         if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
#         df = df.reset_index()
#         clean_cols = {}
#         for c in df.columns:
#             if 'date' in str(c).lower(): clean_cols[c] = 'Date'
#             elif 'close' in str(c).lower(): clean_cols[c] = 'Close'
#             elif 'open' in str(c).lower(): clean_cols[c] = 'Open'
#             elif 'high' in str(c).lower(): clean_cols[c] = 'High'
#             elif 'low' in str(c).lower(): clean_cols[c] = 'Low'
#             elif 'volume' in str(c).lower(): clean_cols[c] = 'Volume'
#         df = df.rename(columns=clean_cols)
#         if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
#         # 计算技术指标
#         df['SMA_20'] = df['Close'].rolling(window=20).mean()
#         df['SMA_50'] = df['Close'].rolling(window=50).mean()
#         # Bollinger Bands
#         df['BB_Mid'] = df['Close'].rolling(window=20).mean()
#         df['BB_Std'] = df['Close'].rolling(window=20).std()
#         df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
#         df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
#         if df.empty or 'Close' not in df.columns: raise ValueError
#         return df, False
#     except:
#         return generate_mock_data(symbol), True

# @st.cache_data(ttl=3600)
# def get_finnhub_news(symbol):
#     """获取 Finnhub 公司新闻"""
#     try:
#         end = datetime.now().strftime('%Y-%m-%d')
#         start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
#         news = requests.get(f"{FINNHUB_BASE_URL}/company-news", 
#                           params={'symbol': symbol, 'from': start, 'to': end, 'token': FINNHUB_API_KEY}).json()
#         return news[:10] # 返回最新的10条
#     except:
#         return []

# @st.cache_data(ttl=300)
# def get_reddit_sentiment(symbol):
#     """分析 Reddit 舆情 (新功能)"""
#     posts_data = []
#     sentiment_score = 0
#     count = 0
    
#     if not RESOURCES['reddit']:
#         return 0, []

#     try:
#         # 搜索相关帖子
#         for submission in RESOURCES['reddit'].subreddit("stocks+wallstreetbets+investing").search(symbol, limit=20, time_filter="week"):
#             title = submission.title
#             score = RESOURCES['vader'].polarity_scores(title)['compound']
#             sentiment_score += score
#             count += 1
#             posts_data.append({
#                 "title": title,
#                 "score": score,
#                 "url": submission.url,
#                 "created": datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d')
#             })
#     except Exception as e:
#         print(f"Reddit Error: {e}")
#         return 0, []
        
#     avg_sentiment = sentiment_score / count if count > 0 else 0
#     return avg_sentiment, posts_data

# def get_filtered_peers(symbol):
#     """竞品获取"""
#     peers_list = []
#     try:
#         r = requests.get(f"{FINNHUB_BASE_URL}/stock/peers", params={'symbol': symbol, 'token': FINNHUB_API_KEY})
#         tickers = [t for t in r.json() if t != symbol and "." not in t and len(t) <= 5 and t.isalpha()]
#         for t in tickers[:4]:
#             try:
#                 prof = requests.get(f"{FINNHUB_BASE_URL}/stock/profile2", params={'symbol': t, 'token': FINNHUB_API_KEY}).json()
#                 quote = requests.get(f"{FINNHUB_BASE_URL}/quote", params={'symbol': t, 'token': FINNHUB_API_KEY}).json()
#                 if prof and quote:
#                     peers_list.append({
#                         "ticker": t, "name": prof.get('name', t), "logo": prof.get('logo', ''),
#                         "price": quote.get('c', 0), "change": quote.get('dp', 0)
#                     })
#             except: continue
#     except: pass
#     return peers_list

# def get_basic_financials(symbol):
#     """获取基本财务数据"""
#     try:
#         metric = requests.get(f"{FINNHUB_BASE_URL}/stock/metric", params={'symbol': symbol, 'metric': 'all', 'token': FINNHUB_API_KEY}).json()
#         return metric.get('metric', {})
#     except: return {}

# # ==========================================
# # 4. 绘图函数
# # ==========================================

# def plot_advanced_chart(df, symbol):
#     fig = go.Figure()
#     fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K线'))
#     if 'SMA_20' in df.columns:
#         fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
#     if 'BB_Upper' in df.columns:
#         fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='gray', width=0), showlegend=False))
#         fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(128,128,128,0.2)', name='布林带'))
#     fig.update_layout(title=f"{symbol} 趋势分析", height=450, xaxis_rangeslider_visible=False, template="plotly_white")
#     return fig

# def plot_radar_fundamentals(quote, profile, metrics):
#     pe = metrics.get('peBasicExclExtraTTM', 20)
#     beta = metrics.get('beta', 1)
    
#     # 归一化分数计算 (简化逻辑)
#     scores = {
#         '低估值': max(0, min(100, 100 - pe if pe else 50)),
#         '成长性': 85, # 示例固定值，实际可根据 revenueGrowthTTM 计算
#         '盈利能力': max(0, min(100, (metrics.get('netProfitMarginTTM', 0) * 2 + 50))),
#         '市场动量': max(0, min(100, 50 + metrics.get('52WeekPriceReturnDaily', 0))),
#         '安全性': max(0, min(100, 100 - (beta * 20)))
#     }
    
#     fig = go.Figure(go.Scatterpolar(r=list(scores.values()), theta=list(scores.keys()), fill='toself', line_color='#4834d4'))
#     fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, margin=dict(t=30, b=20))
#     return fig

# # ==========================================
# # 5. 前端主程序
# # ==========================================

# def main():
#     # --- Sidebar ---
#     with st.sidebar:
#         st.header("🎮 控制台")
#         st.write("当前 AI 风格:")
#         persona = st.selectbox("", ["专业分析师", "华尔街赌徒 (WSB)", "巴菲特"], index=0)
#         st.session_state.persona = persona
#         st.markdown("---")
#         st.info("💡 提示: 在 '智能荐股' 标签页输入行业，让 AI 帮你选股。")

#     # --- Header ---
#     st.markdown("### 🌍 全球市场实况")
#     indices = get_market_indices_robust()
#     cols = st.columns(4)
#     for i, (name, data) in enumerate(indices.items()):
#         bg = "market-card-up" if data['change'] >= 0 else "market-card-down"
#         arrow = "▲" if data['change'] >= 0 else "▼"
#         with cols[i]:
#             st.markdown(f"""<div class="market-card {bg}" style="color: #333;">
#                 <div class="metric-label">{name}</div>
#                 <div class="metric-value">{data['price']:,.2f}</div>
#                 <div style="font-weight:bold;">{arrow} {data['change']:.2f}%</div></div>""", unsafe_allow_html=True)
#     st.markdown("---")

#     # --- Search & Trigger ---
#     c1, c2 = st.columns([3, 1])
#     with c1: st.title("🚀 AlphaSeeker Pro Max")
#     with c2:
#         symbol_input = st.text_input("🔍 股票代码:", value=st.session_state.symbol).upper()
#         start_scan = st.button("🚀 启动深度扫描 & 生成研报", type="primary")

#     if start_scan:
#         st.session_state.symbol = symbol_input
#         st.session_state.chat_history = [] # 重置聊天
#         st.session_state.analysis_report = "" # 重置报告
#         st.rerun()

#     symbol = st.session_state.symbol
    
#     # --- Data Fetching ---
#     with st.spinner(f"正在全网扫描 {symbol} 数据并生成研报..."):
#         # 1. 基础数据
#         try:
#             quote = requests.get(f"{FINNHUB_BASE_URL}/quote", params={'symbol': symbol, 'token': FINNHUB_API_KEY}).json()
#             profile = requests.get(f"{FINNHUB_BASE_URL}/stock/profile2", params={'symbol': symbol, 'token': FINNHUB_API_KEY}).json()
#         except: quote, profile = {}, {}
        
#         hist_df, is_mock = get_stock_history_enhanced(symbol)
#         financials = get_basic_financials(symbol)
#         financial_metrics = get_basic_financials(symbol)
#         news_list = get_finnhub_news(symbol)
#         sentiment_score, reddit_posts = get_reddit_sentiment(symbol)
#         peers = get_filtered_peers(symbol)
        
#         # 2. 自动生成研报 (如果是点击了按钮，且报告为空)
#         if start_scan or not st.session_state.analysis_report:
#             report = generate_full_report(symbol, quote, financials, news_list, sentiment_score, hist_df)
#             st.session_state.analysis_report = report

#     # --- UI Layout ---
#     # 头部信息
#     st.markdown(f"""
#     <div style="background:#fff; padding:20px; border-radius:10px; border-left:5px solid #0984e3; box-shadow:0 2px 5px rgba(0,0,0,0.05); margin-top: 10px;">
#         <div style="display:flex; justify-content:space-between; align-items:center;">
#             <div style="display:flex; align-items:center; gap:15px;">
#                 <img src="{profile.get('logo','')}" style="width:60px; height:60px; border-radius:50%; object-fit:contain;" onerror="this.style.display='none'">
#                 <div>
#                     <h1 style="margin:0;">{symbol} <span style="font-size:0.5em; color:gray;">{profile.get('name', 'Unknown')}</span></h1>
#                     <p style="color:#666; margin:0;">{profile.get('finnhubIndustry', '-')} | {profile.get('exchange', '-')}</p>
#                 </div>
#             </div>
#             <div style="text-align:right;">
#                 <div style="font-size:2.5em; font-weight:bold; color:{'#00b894' if quote.get('dp',0)>0 else '#ff7675'}">${quote.get('c', 0)}</div>
#                 <div>{quote.get('dp', 0)}% (Today)</div>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("###")

#     # Tabs
#     t_report, t_chart, t2, t3, t4, t_rec, t_chat = st.tabs(["📝 深度研报", "📈 市场数据",  "🧬 基本面 & 财务", "🔥 舆情 & 新闻", "⚔️ 竞品对比", "🎯 智能荐股", "🤖 AI 问答"])
 

#     # Tab 1: 深度研报 (新增功能)
#     with t_report:
#         st.subheader(f"📄 {symbol} 投资分析报告 (AI Generated)")
#         if st.session_state.analysis_report:
#             st.markdown(f'<div class="report-box">{st.session_state.analysis_report}</div>', unsafe_allow_html=True)
#         else:
#             st.info("请点击上方的 '启动深度扫描' 生成报告。")
        
#         # 附带舆情摘要
#         st.markdown("---")
#         c1, c2 = st.columns(2)
#         with c1:
#             st.markdown("#### 🔥 社区舆情")
#             st.metric("Reddit 情绪分", f"{sentiment_score:.2f}", delta="Bullish" if sentiment_score > 0 else "Bearish")
#         with c2:
#             st.markdown("#### 📰 最新新闻")
#             for n in news_list[:3]:
#                 st.markdown(f"- [{n['headline']}]({n['url']})")

#     # Tab 2: 市场数据
#     with t_chart:
#         if not hist_df.empty:
#             st.plotly_chart(plot_advanced_chart(hist_df, symbol), use_container_width=True)
        
#         st.subheader("核心财务")
#         cols = st.columns(4)
#         cols[0].metric("P/E (TTM)", f"{financials.get('peBasicExclExtraTTM', 0):.2f}")
#         cols[1].metric("EPS", f"{financials.get('epsExclExtraItemsTTM', 0):.2f}")
#         cols[2].metric("Beta", f"{financials.get('beta', 0):.2f}")
#         cols[3].metric("52周最高", f"{financials.get('52WeekHigh', 0):.2f}")
#     with t2:
#         c1, c2 = st.columns([1, 2])
#         with c1:
#             st.subheader("能力雷达图")
#             st.plotly_chart(plot_radar_fundamentals(quote, profile, financial_metrics), use_container_width=True)
#         with c2: 
#             st.subheader("📊 核心财务指标")
#             if financial_metrics:
#                 m_cols = st.columns(3)
#                 m_cols[0].metric("市盈率 (P/E TTM)", f"{financial_metrics.get('peBasicExclExtraTTM', 0):.2f}")
#                 m_cols[1].metric("每股收益 (EPS)", f"{financial_metrics.get('epsExclExtraItemsTTM', 0):.2f}")
#                 m_cols[2].metric("Beta 系数", f"{financial_metrics.get('beta', 0):.2f}")
                
#                 m_cols2 = st.columns(3)
#                 m_cols2[0].metric("52周最高", f"{financial_metrics.get('52WeekHigh', 0):.2f}")
#                 m_cols2[1].metric("52周最低", f"{financial_metrics.get('52WeekLow', 0):.2f}")
#                 m_cols2[2].metric("股息率", f"{financial_metrics.get('dividendYieldIndicatedAnnual', 0):.2f}%")
#             else:
#                 st.info("暂无详细财务数据")

#     # Tab 3: 舆情与新闻 (全新功能)
#     with t3:
#         c1, c2 = st.columns([1, 1])
        
#         with c1:
#             st.subheader(f"🗣️ Reddit 散户情绪")
#             if sentiment_score > 0.05:
#                 sent_color, sent_text = "#00b894", "看涨 (Bullish)"
#             elif sentiment_score < -0.05:
#                 sent_color, sent_text = "#ff7675", "看跌 (Bearish)"
#             else:
#                 sent_color, sent_text = "#fab1a0", "中性 (Neutral)"
            
#             st.markdown(f"### 情绪指数: <span style='color:{sent_color}'>{sent_text} ({sentiment_score:.2f})</span>", unsafe_allow_html=True)
            
#             st.markdown("#### 最新热帖")
#             if reddit_posts:
#                 for post in reddit_posts[:5]:
#                     score_emoji = "🟢" if post['score'] > 0 else "🔴" if post['score'] < 0 else "⚪"
#                     st.markdown(f"""
#                     <div style="font-size:0.9em; border-bottom:1px solid #eee; padding:5px;">
#                         {score_emoji} <a href="{post['url']}" target="_blank" style="text-decoration:none; color:#333;">{post['title']}</a>
#                         <div style="color:#888; font-size:0.8em;">{post['created']}</div>
#                     </div>
#                     """, unsafe_allow_html=True)
#             else:
#                 st.info("未连接到 Reddit API 或无相关讨论。")

#         with c2:
#             st.subheader("📰 最新财经新闻")
#             if news_list:
#                 for news in news_list:
#                     st.markdown(f"""
#                     <div class="news-card">
#                         <div style="font-weight:bold;">{news.get('headline')}</div>
#                         <div style="font-size:0.85em; color:#555; margin:5px 0;">{news.get('summary', '')[:100]}...</div>
#                         <div style="font-size:0.8em; color:#888;">{news.get('source')} - {datetime.fromtimestamp(news.get('datetime')).strftime('%m-%d %H:%M')}</div>
#                         <a href="{news.get('url')}" target="_blank" style="font-size:0.8em;">阅读全文</a>
#                     </div>
#                     """, unsafe_allow_html=True)
#             else:
#                 st.info("暂无最新新闻。")

#     # Tab 4: 竞品
#     with t4:
#         st.subheader(f"⚔️ {symbol} 的主要竞争对手")
#         if peers:
#             cols = st.columns(4)
#             for i, p in enumerate(peers):
#                 with cols[i % 4]:
#                     color = "#00b894" if p['change'] > 0 else "#ff7675"
#                     st.markdown(f"""
#                     <div class="competitor-card">
#                         <div style="font-weight:bold; font-size:1.2em;">{p['ticker']}</div>
#                         <div style="font-size:0.8em; color:#666; height:20px; overflow:hidden;">{p['name'][:15]}</div>
#                         <div style="margin:5px 0; font-weight:bold;">${p['price']} <span style="color:{color}">({p['change']}%)</span></div>
#                     </div>
#                     """, unsafe_allow_html=True)
#                     if st.button(f"分析 {p['ticker']}", key=f"btn_{p['ticker']}"):
#                         st.session_state.symbol = p['ticker']
#                         st.rerun()
#         else:
#             st.info("未找到符合条件的竞品。")
#     # Tab 3: 智能荐股 (新增功能)
#     with t_rec:
#         st.subheader("🎯 行业扫描与荐股")
#         st.write("选择一个行业，让 AI 扫描该行业的龙头股并给出操作建议。")
        
#         col_input, col_btn = st.columns([3, 1])
#         with col_input:
#             sector_input = st.selectbox("选择或输入行业:", 
#                                       ["AI & 半导体", "科技巨头 (Mag 7)", "电动汽车 (EV)", "生物医药", "加密货币相关", "自定义..."])
#             if sector_input == "自定义...":
#                 custom_sector = st.text_input("请输入行业名称 (如: 量子计算, 航空航天):")
#                 if custom_sector: sector_input = custom_sector
        
#         with col_btn:
#             st.write("") # Spacer
#             st.write("")
#             btn_analyze_sector = st.button("🔍 分析该行业")
            
#         if btn_analyze_sector:
#             with st.spinner(f"正在分析 {sector_input} 行业..."):
#                 tickers = get_sector_tickers(sector_input)
#                 if tickers:
#                     st.write(f"正在扫描: {', '.join(tickers)}")
#                     rec_result = analyze_sector_recommendation(sector_input, tickers)
#                     st.session_state.sector_recommendation = rec_result
#                 else:
#                     st.error("未能识别该行业或找到相关股票。")
        
#         # 展示荐股结果
#         if st.session_state.sector_recommendation and isinstance(st.session_state.sector_recommendation, list):
#             st.markdown("### 📋 AI 交易建议书")
#             rec_cols = st.columns(3)
#             for i, rec in enumerate(st.session_state.sector_recommendation):
#                 action = rec.get('action', 'Hold')
#                 style_class = "rec-buy" if "Buy" in action or "买入" in action else "rec-sell" if "Sell" in action or "卖出" in action else "rec-hold"
#                 stars = "⭐" * int(rec.get('stars', 3))
                
#                 with rec_cols[i % 3]:
#                     st.markdown(f"""
#                     <div class="rec-card {style_class}">
#                         <h3>{rec['symbol']}</h3>
#                         <div style="font-weight:bold; font-size:1.2em; color:#333;">{action}</div>
#                         <div style="color:#f1c40f;">{stars}</div>
#                         <p style="color:#666; font-size:0.9em; margin-top:5px;">{rec['reason']}</p>
#                     </div>
#                     """, unsafe_allow_html=True)

#     # Tab 4: AI 对话
#     with t_chat:
#         st.subheader(f"🤖 咨询 ({st.session_state.persona})")
#         for msg in st.session_state.chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])
                
#         if user_input := st.chat_input(f"关于 {symbol} 还有什么想问的?"):
#             with st.chat_message("user"): st.markdown(user_input)
#             st.session_state.chat_history.append({"role": "user", "content": user_input})
            
#             if RESOURCES['llm']:
#                 sys_prompt = f"你是一个{st.session_state.persona}。当前正在分析 {symbol}。"
#                 full_context = [SystemMessage(content=sys_prompt), HumanMessage(content=user_input)]
#                 with st.chat_message("assistant"):
#                     response = st.write_stream(RESOURCES['llm'].stream(full_context))
#                 st.session_state.chat_history.append({"role": "assistant", "content": response})

# if __name__ == "__main__":
#     main()


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
# 1. 基础配置与 Session State 初始化
# ==========================================

st.set_page_config(page_title="AlphaSeeker Pro Max - AI Agent", layout="wide", page_icon="🏦")

# 初始化 Session State
if 'symbol' not in st.session_state:
    st.session_state.symbol = "NVDA"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_view_symbol' not in st.session_state:
    st.session_state.current_view_symbol = "NVDA"
if 'persona' not in st.session_state:
    st.session_state.persona = "Professional Analyst" # 英文修改
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = ""  # 存储生成的研报
if 'sector_recommendation' not in st.session_state:
    st.session_state.sector_recommendation = None # 存储荐股结果

# --- API Keys ---
DEEPSEEK_API_KEY = "sk-cafba043052344568e72f6d9be865c7d"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat" # 或 deepseek-chat
FINNHUB_API_KEY = "d46s26hr01qgc9euamk0d46s26hr01qgc9euamkg"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Reddit Config
REDDIT_CONFIG = {
    "client_id": "kpCP6k3-q3wWb0UGhGgk-w",
    "client_secret": "3JXSxS2Rs32EzEr09Ywd-XiWCpUCvA",
    "user_agent": "Financial_AI_Agent_V1",
    "username": "ParamedicRelative368",
    "password": "Tsm@928501"
}

# CSS 样式
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
# 2. 核心资源与数据获取
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
        return news[:8]
    except: return []

@st.cache_data(ttl=300)
def get_reddit_sentiment(symbol):
    posts_data = []
    sentiment_score = 0
    count = 0
    if not RESOURCES['reddit']: return 0, []
    try:
        for submission in RESOURCES['reddit'].subreddit("stocks+wallstreetbets+investing").search(symbol, limit=15, time_filter="week"):
            title = submission.title
            score = RESOURCES['vader'].polarity_scores(title)['compound']
            sentiment_score += score
            count += 1
            posts_data.append({"title": title, "score": score, "url": submission.url})
    except: return 0, []
    avg_sentiment = sentiment_score / count if count > 0 else 0
    return avg_sentiment, posts_data

def get_basic_financials(symbol):
    try:
        metric = requests.get(f"{FINNHUB_BASE_URL}/stock/metric", params={'symbol': symbol, 'metric': 'all', 'token': FINNHUB_API_KEY}).json()
        return metric.get('metric', {})
    except: return {}

# ==========================================
# 3. 新增业务逻辑：自动研报 & 行业荐股
# ==========================================

def generate_full_report(symbol, quote, financials, news_list, sentiment_score, hist_df):
    """调用 LLM 生成深度研报""" # 英文修改
    if not RESOURCES['llm']:
        return "⚠️ LLM service is not connected, report generation failed." # 英文修改

    # 准备上下文数据 # 英文修改
    news_context = "\n".join([f"- {n['headline']}" for n in news_list[:5]])
    tech_trend = "Bullish" if hist_df['Close'].iloc[-1] > hist_df['SMA_50'].iloc[-1] else "Bearish" # 英文修改
    
    prompt = f"""
    Please write a professional investment research report for the stock {symbol}.
    
    [Input Data] # 英文修改
    1. Current Price: {quote.get('c')} (Daily Change: {quote.get('dp')}%) # 英文修改
    2. Financial Data: PE(TTM)={financials.get('peBasicExclExtraTTM', 'N/A')}, EPS={financials.get('epsExclExtraItemsTTM', 'N/A')}, Beta={financials.get('beta', 'N/A')} # 英文修改
    3. Recent News Headlines: # 英文修改
    {news_context}
    4. Social Media Sentiment Score (-1 to 1): {sentiment_score:.2f} # 英文修改
    5. Technical Trend: Currently {'Above' if tech_trend=='Bullish' else 'Below'} the 50-day moving average. # 英文修改

    [Report Requirements] # 英文修改
    Please use Markdown format, including the following sections:
    1. **📊 Executive Summary**: A one-sentence summary of the current investment opportunity. # 英文修改
    2. **📰 News and Public Opinion**: Analysis of the impact of news and Reddit sentiment on the stock price. # 英文修改
    3. **🧬 Fundamental Snapshot**: Evaluation of the valuation level and financial health. # 英文修改
    4. **📈 Technical Outlook**: Provide a short-term forecast based on moving averages and trends. # 英文修改
    5. **💡 Final Investment Recommendation**: Explicitly give a rating of "Strong Buy", "Buy", "Hold", or "Sell", and explain the reason. # 英文修改
    
    Maintain a professional and objective tone, with a word count limit of 600 words. # 英文修改
    """
    
    try:
        response = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
        return response
    except Exception as e:
        return f"Report generation failed: {str(e)}" # 英文修改

def get_sector_tickers(sector_name):
    """根据行业名称获取代表性股票代码""" # 英文修改
    # 常用行业硬编码，自定义行业通过 LLM 获取 # 英文修改
    predefined = {
        "AI & Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "AVGO"], # 英文修改
        "Tech Giants (Mag 7)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"], # 英文修改
        "Electric Vehicles (EV)": ["TSLA", "RIVN", "LCID", "NIO", "XPEV"], # 英文修改
        "Biopharma": ["LLY", "NVO", "PFE", "MRK", "JNJ"], # 英文修改
        "Crypto Related": ["COIN", "MSTR", "MARA", "RIOT"] # 英文修改
    }
    
    if sector_name in predefined:
        return predefined[sector_name]
    
    # 如果是用户自定义输入（如“量子计算”），让 LLM 推荐 # 英文修改
    if RESOURCES['llm']:
        prompt = f"Please list 5 US-listed stock tickers belonging to the '{sector_name}' sector. Return only the tickers, separated by commas, with no other text. For example: AAPL, MSFT" # 英文修改
        try:
            resp = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
            tickers = [t.strip().upper() for t in resp.split(',') if t.strip().isalpha()]
            return tickers[:5]
        except: return []
    return []

def analyze_sector_recommendation(sector_name, tickers):
    """分析行业并给出推荐""" # 英文修改
    if not tickers: return "No stocks found for this sector." # 英文修改
    
    # 批量获取简要数据 # 英文修改
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
        
    if not RESOURCES['llm']: return "LLM service is not connected." # 英文修改

    # 构建 Prompt # 英文修改
    data_str = "\n".join([f"{d['symbol']}: Price ${d['price']}, Change {d['change']}%, PE={d['pe']}" for d in data_summary]) # 英文修改
    prompt = f"""
    You are a seasoned fund manager. The user wants to know about investment opportunities in the '{sector_name}' sector. # 英文修改
    
    Here is the real-time data for several representative stocks in this sector: # 英文修改
    {data_str}
    
    Please output an investment recommendation table.
    For each stock:
    1. Give a recommendation of "Buy", "Sell", or "Hold". # 英文修改
    2. Explain the reason in one sentence (combining valuation, momentum, or your knowledge of the company). # 英文修改
    3. Give a "Recommendation Index" (1-5 stars). # 英文修改
    
    Return a list in JSON format, with fields: symbol, action, reason, stars (integer).
    Do not output any text other than the JSON. # 英文修改
    """
    
    try:
        resp = RESOURCES['llm'].invoke([HumanMessage(content=prompt)]).content
        # 清理可能的 markdown 标记 # 英文修改
        if "```json" in resp: resp = resp.split("```json")[1].split("```")[0]
        elif "```" in resp: resp = resp.split("```")[1].split("```")[0]
        return json.loads(resp)
    except Exception as e:
        return f"Analysis failed: {str(e)}" # 英文修改

@st.cache_data(ttl=1800)
def get_stock_history_enhanced(symbol):
    """K线获取 + 技术指标计算""" # 英文修改
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
        
        # 计算技术指标 # 英文修改
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
    """获取 Finnhub 公司新闻""" # 英文修改
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news = requests.get(f"{FINNHUB_BASE_URL}/company-news", 
                          params={'symbol': symbol, 'from': start, 'to': end, 'token': FINNHUB_API_KEY}).json()
        return news[:10] # 返回最新的10条
    except:
        return []

@st.cache_data(ttl=300)
def get_reddit_sentiment(symbol):
    """分析 Reddit 舆情 (新功能)""" # 英文修改
    posts_data = []
    sentiment_score = 0
    count = 0
    
    if not RESOURCES['reddit']:
        return 0, []

    try:
        # 搜索相关帖子 # 英文修改
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
    """竞品获取""" # 英文修改
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
    """获取基本财务数据""" # 英文修改
    try:
        metric = requests.get(f"{FINNHUB_BASE_URL}/stock/metric", params={'symbol': symbol, 'metric': 'all', 'token': FINNHUB_API_KEY}).json()
        return metric.get('metric', {})
    except: return {}

# ==========================================
# 4. 绘图函数
# ==========================================

def plot_advanced_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')) # 英文修改
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='gray', width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(128,128,128,0.2)', name='Bollinger Bands')) # 英文修改
    fig.update_layout(title=f"{symbol} Trend Analysis", height=450, xaxis_rangeslider_visible=False, template="plotly_white") # 英文修改
    return fig

def plot_radar_fundamentals(quote, profile, metrics):
    pe = metrics.get('peBasicExclExtraTTM', 20)
    beta = metrics.get('beta', 1)
    
    # 归一化分数计算 (简化逻辑) # 英文修改
    scores = {
        'Low Valuation': max(0, min(100, 100 - pe if pe else 50)), # 英文修改
        'Growth': 85, # 英文修改
        'Profitability': max(0, min(100, (metrics.get('netProfitMarginTTM', 0) * 2 + 50))), # 英文修改
        'Market Momentum': max(0, min(100, 50 + metrics.get('52WeekPriceReturnDaily', 0))), # 英文修改
        'Safety': max(0, min(100, 100 - (beta * 20))) # 英文修改
    }
    
    fig = go.Figure(go.Scatterpolar(r=list(scores.values()), theta=list(scores.keys()), fill='toself', line_color='#4834d4'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, margin=dict(t=30, b=20))
    return fig

# ==========================================
# 5. 前端主程序
# ==========================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("🎮 Control Panel") # 英文修改
        st.write("Current AI Persona:") # 英文修改
        persona = st.selectbox("", ["Professional Analyst", "Wall Street Bets (WSB)", "Warren Buffett"], index=0) # 英文修改
        st.session_state.persona = persona
        st.markdown("---")
        st.info("💡 Tip: Go to the 'Smart Sector Picks' tab to enter a sector and let the AI select stocks for you.") # 英文修改

    # --- Header ---
    st.markdown("### 🌍 Global Market Overview") # 英文修改
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
        symbol_input = st.text_input("🔍 Stock Ticker:", value=st.session_state.symbol).upper() # 英文修改
        start_scan = st.button("🚀 Start Deep Scan & Generate Report", type="primary") # 英文修改

    if start_scan:
        st.session_state.symbol = symbol_input
        st.session_state.chat_history = [] # 重置聊天 # 英文修改
        st.session_state.analysis_report = "" # 重置报告 # 英文修改
        st.rerun()

    symbol = st.session_state.symbol
    
    # --- Data Fetching ---
    with st.spinner(f"Scanning {symbol} data and generating report across the web..."): # 英文修改
        # 1. 基础数据 # 英文修改
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
        
        # 2. 自动生成研报 (如果是点击了按钮，且报告为空) # 英文修改
        if start_scan or not st.session_state.analysis_report:
            report = generate_full_report(symbol, quote, financials, news_list, sentiment_score, hist_df)
            st.session_state.analysis_report = report

    # --- UI Layout ---
    # 头部信息 # 英文修改
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
    t_report, t_chart, t2, t3, t4, t_rec, t_chat = st.tabs(["📝 Deep Dive Report", "📈 Market Data",  "🧬 Fundamentals & Financials", "🔥 Sentiment & News", "⚔️ Peer Comparison", "🎯 Smart Sector Picks", "🤖 AI Q&A"]) # 英文修改
 

    # Tab 1: 深度研报 (新增功能) # 英文修改
    with t_report:
        st.subheader(f"📄 {symbol} Investment Analysis Report (AI Generated)") # 英文修改
        if st.session_state.analysis_report:
            st.markdown(f'<div class="report-box">{st.session_state.analysis_report}</div>', unsafe_allow_html=True)
        else:
            st.info("Please click 'Start Deep Scan' above to generate the report.") # 英文修改
        
        # 附带舆情摘要 # 英文修改
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔥 Community Sentiment") # 英文修改
            st.metric("Reddit Sentiment Score", f"{sentiment_score:.2f}", delta="Bullish" if sentiment_score > 0 else "Bearish") # 英文修改
        with c2:
            st.markdown("#### 📰 Latest News") # 英文修改
            for n in news_list[:3]:
                st.markdown(f"- [{n['headline']}]({n['url']})")

    # Tab 2: 市场数据 # 英文修改
    with t_chart:
        if not hist_df.empty:
            st.plotly_chart(plot_advanced_chart(hist_df, symbol), use_container_width=True)
        
        st.subheader("Core Financials") # 英文修改
        cols = st.columns(4)
        cols[0].metric("P/E (TTM)", f"{financials.get('peBasicExclExtraTTM', 0):.2f}")
        cols[1].metric("EPS", f"{financials.get('epsExclExtraItemsTTM', 0):.2f}")
        cols[2].metric("Beta", f"{financials.get('beta', 0):.2f}")
        cols[3].metric("52-Week High", f"{financials.get('52WeekHigh', 0):.2f}") # 英文修改
    with t2:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Capability Radar") # 英文修改
            st.plotly_chart(plot_radar_fundamentals(quote, profile, financial_metrics), use_container_width=True)
        with c2: 
            st.subheader("📊 Core Financial Metrics") # 英文修改
            if financial_metrics:
                m_cols = st.columns(3)
                m_cols[0].metric("P/E Ratio (TTM)", f"{financial_metrics.get('peBasicExclExtraTTM', 0):.2f}") # 英文修改
                m_cols[1].metric("EPS", f"{financial_metrics.get('epsExclExtraItemsTTM', 0):.2f}") # 英文修改
                m_cols[2].metric("Beta", f"{financial_metrics.get('beta', 0):.2f}")
                
                m_cols2 = st.columns(3)
                m_cols2[0].metric("52-Week High", f"{financial_metrics.get('52WeekHigh', 0):.2f}") # 英文修改
                m_cols2[1].metric("52-Week Low", f"{financial_metrics.get('52WeekLow', 0):.2f}") # 英文修改
                m_cols2[2].metric("Dividend Yield", f"{financial_metrics.get('dividendYieldIndicatedAnnual', 0):.2f}%") # 英文修改
            else:
                st.info("No detailed financial data available.") # 英文修改

    # Tab 3: 舆情与新闻 (全新功能) # 英文修改
    with t3:
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader(f"🗣️ Reddit Retail Sentiment") # 英文修改
            if sentiment_score > 0.05:
                sent_color, sent_text = "#00b894", "Bullish" # 英文修改
            elif sentiment_score < -0.05:
                sent_color, sent_text = "#ff7675", "Bearish" # 英文修改
            else:
                sent_color, sent_text = "#fab1a0", "Neutral" # 英文修改
            
            st.markdown(f"### Sentiment Index: <span style='color:{sent_color}'>{sent_text} ({sentiment_score:.2f})</span>", unsafe_allow_html=True) # 英文修改
            
            st.markdown("#### Latest Hot Posts") # 英文修改
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
                st.info("Not connected to Reddit API or no relevant discussions.") # 英文修改

        with c2:
            st.subheader("📰 Latest Financial News") # 英文修改
            if news_list:
                for news in news_list:
                    st.markdown(f"""
                    <div class="news-card">
                        <div style="font-weight:bold;">{news.get('headline')}</div>
                        <div style="font-size:0.85em; color:#555; margin:5px 0;">{news.get('summary', '')[:100]}...</div>
                        <div style="font-size:0.8em; color:#888;">{news.get('source')} - {datetime.fromtimestamp(news.get('datetime')).strftime('%m-%d %H:%M')}</div>
                        <a href="{news.get('url')}" target="_blank" style="font-size:0.8em;">Read Full Article</a>
                    </div>
                    """, unsafe_allow_html=True) # 英文修改
            else:
                st.info("No latest news available.") # 英文修改

    # Tab 4: 竞品 # 英文修改
    with t4:
        st.subheader(f"⚔️ {symbol}'s Main Competitors") # 英文修改
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
                    if st.button(f"Analyze {p['ticker']}", key=f"btn_{p['ticker']}"): # 英文修改
                        st.session_state.symbol = p['ticker']
                        st.rerun()
        else:
            st.info("No suitable competitors found.") # 英文修改
    # Tab 3: 智能荐股 (新增功能) # 英文修改
    with t_rec:
        st.subheader("🎯 Sector Scan and Stock Picks") # 英文修改
        st.write("Select a sector to let the AI scan its leading stocks and provide trading recommendations.") # 英文修改
        
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            sector_input = st.selectbox("Select or Enter Sector:", # 英文修改
                                      ["AI & Semiconductors", "Tech Giants (Mag 7)", "Electric Vehicles (EV)", "Biopharma", "Crypto Related", "Custom..."]) # 英文修改
            if sector_input == "Custom...": # 英文修改
                custom_sector = st.text_input("Enter Sector Name (e.g., Quantum Computing, Aerospace):") # 英文修改
                if custom_sector: sector_input = custom_sector
        
        with col_btn:
            st.write("") # Spacer
            st.write("")
            btn_analyze_sector = st.button("🔍 Analyze Sector") # 英文修改
            
        if btn_analyze_sector:
            with st.spinner(f"Analyzing {sector_input} sector..."): # 英文修改
                tickers = get_sector_tickers(sector_input)
                if tickers:
                    st.write(f"Scanning: {', '.join(tickers)}") # 英文修改
                    rec_result = analyze_sector_recommendation(sector_input, tickers)
                    st.session_state.sector_recommendation = rec_result
                else:
                    st.error("Could not identify the sector or find relevant stocks.") # 英文修改
        
        # 展示荐股结果 # 英文修改
        if st.session_state.sector_recommendation and isinstance(st.session_state.sector_recommendation, list):
            st.markdown("### 📋 AI Trading Recommendation Letter") # 英文修改
            rec_cols = st.columns(3)
            for i, rec in enumerate(st.session_state.sector_recommendation):
                action = rec.get('action', 'Hold')
                style_class = "rec-buy" if "Buy" in action or "买入" in action else "rec-sell" if "Sell" in action or "卖出" in action else "rec-hold" # 保留中文判断以防LLM返回中文
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

    # Tab 4: AI 对话 # 英文修改
    with t_chat:
        st.subheader(f"🤖 Ask the AI ({st.session_state.persona})") # 英文修改
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if user_input := st.chat_input(f"What else would you like to know about {symbol}?"): # 英文修改
            with st.chat_message("user"): st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            if RESOURCES['llm']:
                sys_prompt = f"You are a {st.session_state.persona}. You are currently analyzing {symbol}." # 英文修改
                full_context = [SystemMessage(content=sys_prompt), HumanMessage(content=user_input)]
                with st.chat_message("assistant"):
                    response = st.write_stream(RESOURCES['llm'].stream(full_context))
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()