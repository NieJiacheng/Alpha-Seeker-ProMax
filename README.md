# FinAgent

# 🏦 AlphaSeeker Pro Max: AI-Powered Financial Investment Research Agent

AlphaSeeker is a high-performance, full-stack investment research agent designed to bridge the gap between institutional-grade analysis and retail market sentiment. By fusing real-time market data, fundamental metrics, and social media pulses (Reddit), AlphaSeeker provides a 24/7 "Cognitive Engine" for modern financial decision-making.

---

## 🌟 Project Motivation
In the current financial landscape, investors face **Information Overload** and **Data Silos**. AlphaSeeker addresses these by:
* **Synthesizing Heterogeneous Data**: Merging hard numbers (Earnings, P/E) with soft signals (Reddit Hype).
* **Democratizing Analysis**: Providing professional-level insights using advanced LLM reasoning.
* **Reducing Latency**: Converting hours of manual research into seconds of automated intelligence.



---

## 🏗️ System Architecture (The Four-Layer Model)

AlphaSeeker is built on a disciplined four-layer architecture, as detailed in our technical presentation:

### 1. Data Adapter Layer (`finance_agent.py` -> `DataFetcher`)
* **Market Data**: Uses `yFinance` to extract real-time OHLCV data and historical trends.
* **Social Pulse**: Connects to the Reddit API via `PRAW` to scrape subreddits like `r/wallstreetbets` and `r/stocks`.
* **Graceful Degradation**: Implements a "Survival Instinct"—if APIs fail, the system automatically triggers `generate_mock_data` to ensure continuous UI stability.

### 2. Feature Engineering Layer
* **Technical Engine**: Computes indicators like SMA_20, SMA_50, and Bollinger Bands using `pandas` and `numpy`.
* **Quantified Sentiment**: Employs `vaderSentiment` to transform raw text into a normalized Sentiment Score $[-1, 1]$.

### 3. Cognitive Decision Layer (The Brain)
* **Orchestration**: Built with `LangChain` to manage LLM interactions (DeepSeek/OpenAI).
* **Persona Engine**: Supports dynamic switching between **Professional Analyst**, **Wall Street Bettor**, and **Value Investor**.
* **Context Injection**: Explicitly injects the `current_date` into the system prompt to eliminate temporal hallucinations.

### 4. Interaction Layer
* **Streamlit Dashboard**: A highly interactive Web UI.
* **Visualization**: Uses `Plotly` for financial charts and **Radar Analysis** across 5 dimensions (Valuation, Growth, Profitability, Momentum, Safety).



---

## 🛠️ Key Technical Implementations

### Progressive Loading & UX
To handle the latency of LLM generation, the system utilizes **Streamlit's Progressive Rendering**. It populates the technical charts and fundamental tables first, while the AI report streams in the background.

### Robustness & Determinism
* **Temperature Control**: Set to `0.3` to ensure consistency in financial recommendations.
* **Constraint-Based Prompting**: The agent is strictly instructed: *"If data is missing, respond 'I don't have enough information'."* to prevent data fabrication.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* API Keys for:
    * **OpenAI/DeepSeek**: For the reasoning core.
    * **Reddit (PRAW)**: Client ID and Secret for sentiment scraping.

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/alphaseeker.git](https://github.com/your-username/alphaseeker.git)
    cd alphaseeker
    ```

2.  **Install Dependencies**:
    ```bash
    pip install streamlit pandas plotly yfinance vaderSentiment langchain_openai praw numpy
    ```

3.  **Run the Agent**:
    ```bash
    streamlit run finance_agent.py
    ```

---

## 📈 Usage Scenarios

1.  **Deep Dive Analysis**: Enter a ticker (e.g., `NVDA`) to get a holistic view of technicals vs. Reddit sentiment.
2.  **Sector Discovery**: Use the "Sector Recommendation" feature to find the top-performing stocks in "Semiconductors" or "AI".
3.  **Stress Testing**: Change personas to see how a "Value Investor" perspective differs from a "Wall Street Bettor" on the same stock.

---

## 🔮 Future Roadmap

* **Multi-Agent Swarm**: Deploying a "Macro-Analyst Agent" and a "Quant-Specialist Agent" to debate before issuing a final rating.
* **RAG (Retrieval-Augmented Generation)**: Integrating a Vector Database to allow the agent to read 10-K filings and analyst PDFs.
* **Backtesting Module**: A "Reality Check" tool to simulate how the AI's past recommendations would have performed in the market.

---

**Tech Stack**: `Python`, `LangChain`, `Streamlit`, `DeepSeek LLM`, `yFinance`, `PRAW`.

**Disclaimer**: *AlphaSeeker is for educational and research purposes only. It does not constitute financial advice.*