import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib import style
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class StreamlitStockChatBot:
    def __init__(self):
        self.company_name = None
        self.previous_company_name = None
        self.selected_period = None
        self.buttons_shown = False
        
    def run(self):
        st.set_page_config(page_title="Stock Price Chatbot", layout="wide")
        st.title("Stock Price Chatbot")

        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.company_name = None
            st.session_state.previous_company_name = None
            st.session_state.selected_period = None
            st.session_state.buttons_shown = False

        # Main layout
        col1, col2 = st.columns([1, 1])

        with col1:
            self.chat_interface()

        with col2:
            self.graph_interface()

    def chat_interface(self):
        st.subheader("Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            self.process_input(prompt)

        # Show initial message and buttons if not already shown
        if not st.session_state.buttons_shown:
            self.start_chat()

    def graph_interface(self):
        st.subheader("Graph")
        if st.session_state.company_name:
            self.show_details()
        else:
            st.write("Please select a company in the chat to view the graph.")

    def start_chat(self):
        self.append_message("assistant", "Choose a stock ticker:")
        self.show_ticker_buttons()
        st.session_state.buttons_shown = True

    def show_ticker_buttons(self):
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        cols = st.columns(len(tickers))
        for i, ticker in enumerate(tickers):
            if cols[i].button(ticker, key=f"ticker_{ticker}"):
                self.handle_ticker(ticker)

    def handle_ticker(self, ticker):
        st.session_state.previous_company_name = st.session_state.company_name
        st.session_state.company_name = ticker
        self.append_message("user", f"I chose {st.session_state.company_name}")
        self.append_message("assistant", f"What do you want to do with {st.session_state.company_name}?")
        self.show_choice_buttons()

    def show_choice_buttons(self):
        choices = ["Period", "Current", "Predict"]
        cols = st.columns(len(choices))
        for i, choice in enumerate(choices):
            if cols[i].button(choice, key=f"choice_{choice}"):
                if choice == "Period":
                    self.choose_period()
                elif choice == "Current":
                    self.show_current_price()
                elif choice == "Predict":
                    self.show_predict_price()

    def choose_period(self):
        self.append_message("assistant", "Choose a period:")
        periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        cols = st.columns(len(periods))
        for i, period in enumerate(periods):
            if cols[i].button(period, key=f"period_{period}"):
                self.handle_period(period)

    def handle_period(self, period):
        st.session_state.selected_period = period
        self.append_message("user", f"I chose {period}")
        self.show_period_price(period)

    def show_current_price(self):
        try:
            current_data = yf.download(st.session_state.company_name, start=datetime.now().date(), end=datetime.now().date() + pd.DateOffset(days=1), interval='1m')
            if current_data.empty:
                self.append_message("assistant", "No data available for today.")
            else:
                self.append_message("assistant", "Today's closing prices:")
                self.append_message("assistant", str(current_data["Close"].tail()))
                self.display_plot(current_data, f"{st.session_state.company_name} Prices Today")
        except Exception as e:
            self.append_message("assistant", f"Error: {e}")

    def show_period_price(self, period):
        try:
            data = yf.download(st.session_state.company_name, period=period, interval="1d")
            if data.empty:
                self.append_message("assistant", f"No data available for the period: {period}.")
            else:
                self.append_message("assistant", f"{st.session_state.company_name} Prices for {period}:")
                self.append_table(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
                self.display_plot(data, f"{st.session_state.company_name} Prices for {period}")
        except Exception as e:
            self.append_message("assistant", f"Error: {e}")

    def show_predict_price(self):
        try:
            end_date = datetime.now() - pd.DateOffset(days=1)
            start_date = end_date - pd.DateOffset(days=365)
            data = yf.download(st.session_state.company_name, start=start_date, end=end_date + pd.DateOffset(days=1), interval="1d")

            if data.empty:
                self.append_message("assistant", "No historical data available.")
                return

            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)
            X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = data['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            latest_data = pd.DataFrame([X.iloc[-1]], columns=X.columns)
            predicted_price = model.predict(latest_data)[0]

            accuracy = model.score(X_test, y_test) * 100

            self.append_message("assistant", f"Next price prediction: {predicted_price:.2f}")
            self.append_message("assistant", f"Accuracy of prediction: {accuracy:.2f}%")

            self.display_prediction_plot(data, predicted_price)

        except Exception as e:
            self.append_message("assistant", f"Error: {e}")

    def display_plot(self, data, title):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        data = data.reset_index()
        data['Date'] = mdates.date2num(pd.to_datetime(data['Date']))
        ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']].values

        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.grid(True)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.set_title(title)
        fig.autofmt_xdate()

        for i, row in data.iterrows():
            ax1.annotate(f'{row["Close"]:.2f}', (row["Date"], row["Close"]),
                        textcoords="offset points", xytext=(0,5), ha='center',
                        fontsize=8, color='black', weight='bold')

        st.pyplot(fig)

    def display_prediction_plot(self, data, predicted_price):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(data.index, data['Close'], label='Historical Prices', color='blue')
        ax1.axhline(y=predicted_price, color='r', linestyle='--', label=f'Predicted Price: {predicted_price:.2f}')

        ax1.set_xlim(data.index[0], data.index[-1])
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.grid(True)

        ax1.set_title(f"{st.session_state.company_name} Price Prediction")
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()

        ax1.text(0.5, 0.05, 'Note: Predictions are based on historical data and may vary.', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax1.transAxes, fontsize=12, color='black', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        fig.autofmt_xdate()

        st.pyplot(fig)

    def show_details(self):
        try:
            data = yf.download(st.session_state.company_name, period="1y", interval="1d")
            data = data.reset_index()
            data['Date'] = mdates.date2num(pd.to_datetime(data['Date']))
            ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']].values

            fig, ax1 = plt.subplots(figsize=(12, 8))

            candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
            ax1.grid(True)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')

            ax1.set_title(f"{st.session_state.company_name} Detailed Analysis")
            fig.autofmt_xdate()

            selected_points = [0, len(data)//2, len(data)-1]
            for i in selected_points:
                row = data.iloc[i]
                ax1.annotate(f'{row["Close"]:.2f}', (row["Date"], row["Close"]),
                            textcoords="offset points", xytext=(0,5), ha='center',
                            fontsize=8, color='black', weight='bold')

            st.pyplot(fig)

            # Display additional stock information
            stock = yf.Ticker(st.session_state.company_name)
            info = stock.info
            st.subheader("Company Information")
            st.write(f"Company Name: {info.get('longName', 'N/A')}")
            st.write(f"Sector: {info.get('sector', 'N/A')}")
            st.write(f"Industry: {info.get('industry', 'N/A')}")
            st.write(f"Website: {info.get('website', 'N/A')}")
            st.write(f"Market Cap: ${info.get('marketCap', 'N/A'):,}")
            st.write(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
            st.write(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")

        except Exception as e:
            st.error(f"Error: {e}")

    def append_message(self, role, content):
        st.session_state.messages.append({"role": role, "content": content})

    def append_table(self, data):
        st.table(data.style.format("{:.2f}").applymap(self.color_negative_red))

    def color_negative_red(self, val):
        color = 'red' if val < 0 else 'black'
        return f'color: {color}'

    def process_input(self, input_text):
        input_text = input_text.strip().upper()
        if input_text in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
            self.handle_ticker(input_text)
        elif input_text in ["PERIOD", "CURRENT", "PREDICT"]:
            if input_text == "PERIOD":
                self.choose_period()
            elif input_text == "CURRENT":
                self.show_current_price()
            elif input_text == "PREDICT":
                self.show_predict_price()
        elif input_text in ['1D', '5D', '1MO', '3MO', '6MO', '1Y', '2Y', '5Y', '10Y', 'YTD', 'MAX']:
            self.handle_period(input_text.lower())
        else:
            self.append_message("assistant", "I'm sorry, I didn't understand that command. Please choose from the available options.")

if __name__ == "__main__":
    bot = StreamlitStockChatBot()
    bot.run()