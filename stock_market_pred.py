from PyQt5.QtWidgets import (QStackedWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, 
                             QWidget, QApplication, QScrollArea,QMainWindow)
from PyQt5.QtGui import QIcon, QTextCursor
import sys
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib import style
import numpy as np

class StockChatBot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Price Chatbot")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #ECF0F1; color: #2C3E50;")
        self.company_name = None
        self.previous_company_name = None
        self.buttons_shown = False
        self.canvas = None
        self.selected_period = None  # Store the selected period

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.stack = QStackedWidget(self)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.stack)

        # Chat Window
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.stack.addWidget(self.chat_widget)

        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
        background-color: #F7F9F9;
        color: #2C3E50;
        border: 1px solid #D5DBDB;
        padding: 10px;
        font-family: 'Arial';  
        font-size: 18px;  /* Increased font size */
    """)

        self.chat_layout.addWidget(self.chat_display)

        self.input_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setStyleSheet("""
            background-color: #FFFFFF;
            color: #2C3E50;
            border-radius: 10px;
            padding: 10px;
            font-family: 'Roboto';
            font-size: 18px;  /* Increased font size */
            border: 1px solid #D5DBDB;
        """)
        self.input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton(self)
        self.send_button.setIcon(QIcon("/Users/pradyumnadeepakaher/Downloads/Stock market python project 2/message.png"))  # Replace with your icon path
        self.send_button.setStyleSheet("""
            background-color: #1ABC9C;
            color: #FFFFFF;
            border-radius: 10px;
            padding: 10px;
            margin-left: 10px;
        """)
        self.send_button.clicked.connect(self.process_input)
        self.input_field.returnPressed.connect(self.process_input)
        self.input_layout.addWidget(self.send_button)
        
        self.chat_layout.addLayout(self.input_layout)

        # Graph Window
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_widget)
        self.stack.addWidget(self.graph_widget)

        # Add navigation buttons
        self.nav_layout = QHBoxLayout()
        self.switch_to_chat_button = QPushButton("Chat")
        self.switch_to_chat_button.setStyleSheet("""
            background-color: #3498DB;
            color: #ECF0F1;
            border-radius: 10px;
            padding: 10px;
            font-family: 'Roboto';
            font-size: 16px;
        """)
        self.switch_to_chat_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.chat_widget))
        self.nav_layout.addWidget(self.switch_to_chat_button)

        self.switch_to_graph_button = QPushButton("Graph")
        self.switch_to_graph_button.setStyleSheet("""
            background-color: #3498DB;
            color: #ECF0F1;
            border-radius: 10px;
            padding: 10px;
            font-family: 'Roboto';
            font-size: 16px;
        """)
        self.switch_to_graph_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.graph_widget))
        self.nav_layout.addWidget(self.switch_to_graph_button)

        self.layout.addLayout(self.nav_layout)
        self.layout.addWidget(self.stack)

        self.start_chat()

    def start_chat(self):
        self.append_chat("Bot", "Choose a stock ticker:")
        self.show_ticker_buttons()

    def process_input(self):
        ticker = self.input_field.text().strip().upper()
        if ticker:
            self.handle_ticker(ticker)
            self.show_details()  # Show detailed graph immediately after handling the ticker
        else:
            self.append_chat("Bot", "Enter a valid ticker symbol.")
        self.input_field.clear()

    def append_chat(self, sender, message):
        if sender == "Bot":
            self.chat_display.append(f'''
                <div style="margin: 10px;">
                    <table width="100%">
                        <tr>
                            <td align="left" style="background-color: #E8F8F5; border-radius: 10px; padding: 10px; color: #2C3E50;">
                                <b>{sender}:</b> {message}
                            </td>
                            <td width="30%"></td>
                        </tr>
                    </table>
                </div>
            ''')
        else:
            self.chat_display.append(f'''
                <div style="margin: 10px;">
                    <table width="100%">
                        <tr>
                            <td width="30%"></td>
                            <td align="right" style="background-color: #D6DBDF; border-radius: 10px; padding: 10px; color: #2C3E50;">
                                <b>{sender}:</b> {message}
                            </td>
                        </tr>
                    </table>
                </div>
            ''')
        self.chat_display.moveCursor(QTextCursor.End)




    def show_ticker_buttons(self):
        self.ticker_layout = QHBoxLayout()
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        for ticker in tickers:
            button = QPushButton(ticker)
            button.setStyleSheet("""
                background-color: #2980B9;
                color: #ECF0F1;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Roboto';
                font-size: 14px;
            """)
            button.setToolTip(f"Choose {ticker}")
            button.clicked.connect(lambda _, t=ticker: self.handle_ticker(t))
            self.ticker_layout.addWidget(button)
        self.chat_layout.addLayout(self.ticker_layout)


    def handle_ticker(self, ticker):
        self.previous_company_name = self.company_name
        self.company_name = ticker
        if self.company_name != self.previous_company_name:
            self.clear_plot()  

        self.append_chat("You", f"I chose {self.company_name}")
        self.clear_layout(self.ticker_layout)
        self.append_chat("Bot", f"What do you want to do with {self.company_name}?")
        if not self.buttons_shown:
            self.show_choice_buttons()
            self.buttons_shown = True

        # Immediately show detailed graph after choosing ticker
        self.show_details()


    def show_choice_buttons(self):
        self.choice_layout = QHBoxLayout()
        choices = {
            "Period": self.choose_period,
            "Current": self.show_current_price,
            "Predict": self.show_predict_price,
            "Exit": self.close
        }
        for choice, command in choices.items():
            button = QPushButton(choice)
            button.setStyleSheet("""
                background-color: #E67E22;
                color: #ECF0F1;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Roboto';
                font-size: 14px;
            """)
            button.setToolTip(f"{choice} analysis")
            button.clicked.connect(command)
            self.choice_layout.addWidget(button)
        self.chat_layout.addLayout(self.choice_layout)


    def choose_period(self):
        self.clear_layout(self.choice_layout)
        self.append_chat("Bot", "Choose a period:")
        self.show_period_buttons()

    def show_period_buttons(self):
        self.period_layout = QHBoxLayout()
        periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        for period in periods:
            button = QPushButton(period)
            button.setStyleSheet("""
                background-color: #9B59B6;
                color: #ECF0F1;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Roboto';
                font-size: 14px;
            """)
            button.setToolTip(f"Show data for {period}")
            button.clicked.connect(lambda _, p=period: self.handle_period(p))
            self.period_layout.addWidget(button)
        self.chat_layout.addLayout(self.period_layout)


    def handle_period(self, period):
        self.selected_period = period  # Store the selected period
        self.append_chat("You", f"I chose {period}")
        self.clear_layout(self.period_layout)
        self.show_period_price(period)

    def show_current_price(self):
        self.clear_layout(self.choice_layout)
        self.clear_plot()  
        try:
            current_data = yf.download(self.company_name, start=datetime.now().date(), end=datetime.now().date() + pd.DateOffset(days=1), interval='1m')
            if current_data.empty:
                self.append_chat("Bot", "No data available for today.")
            else:
                self.append_chat("Bot", "Today's closing prices:")
                self.append_chat("Bot", str(current_data["Close"].tail()))

                self.display_plot(current_data, f"{self.company_name} Prices Today")
                # Switch to the graph view
                self.stack.setCurrentWidget(self.graph_widget)
        except Exception as e:
            self.append_chat("Bot", f"Error: {e}")

        self.restart_process()

    def show_period_price(self, period):
        try:
            # Download data for the selected period
            data = yf.download(self.company_name, period=period, interval="1d")
            if data.empty:
                self.append_chat("Bot", f"No data available for the period: {period}.")
            else:
                # Append data as a table to chat
                self.append_chat("Bot", f"{self.company_name} Prices for {period}:")
                self.append_table(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))  # Show last 10 rows
                
                # Display plot
                self.display_plot(data, f"{self.company_name} Prices for {period}")
                # Switch to the graph view
                self.stack.setCurrentWidget(self.graph_widget)
        except Exception as e:
            self.append_chat("Bot", f"Error: {e}")

        self.restart_process()
    
    # Function to apply color styling to positive and negative values
    
    # Function to apply color styling to positive and negative values
    def append_table(self, data):
    # Remove empty columns if they exist
        data = data.loc[:, data.notna().any(axis=0)]
        
        # Convert the data to HTML with additional styling
        table_html = data.to_html(classes='table table-striped', index=True, escape=False)
        
        # Define custom styles for the table
        styled_table = f"""
        <style>
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 16px;
                color: #333;
            }}
            .table thead {{
                background-color: #343A40;
                color: #FFFFFF;
            }}
            .table thead th {{
                padding: 12px;
                text-align: left;
                color: #000000; /* Set header text color to black */
            }}
            .table tbody tr:nth-of-type(odd) {{
                background-color: #F9F9F9;
            }}
            .table tbody tr:nth-of-type(even) {{
                background-color: #FFFFFF;
            }}
            .table tbody tr:hover {{
                background-color: #E8F8F5;
            }}
            .table td {{
                padding: 12px;
                border-bottom: 1px solid #E0E0E0;
            }}
            .positive {{
                color: #28A745;  /* Green color for positive values */
            }}
            .negative {{
                color: #DC3545;  /* Red color for negative values */
            }}
            .date {{
                color: #007BFF;  /* Blue color for dates */
            }}
            .table th, .table td {{
                border: none;
                white-space: nowrap; /* Prevent text from wrapping */
            }}
            .table-container {{
                overflow-x: auto; /* Enable horizontal scrolling */
                border: 1px solid #D5DBDB;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
        </style>
        <div class="table-container">
            {table_html}
        </div>
        """
        
        # Append the styled table to the chat display
        self.chat_display.setHtml(self.chat_display.toHtml() + styled_table)
        self.chat_display.moveCursor(QTextCursor.End)






  
    
    
    def show_predict_price(self):
        self.clear_layout(self.choice_layout)
        self.clear_plot()
        
        try:
            # Define the end date as the day before the current date
            end_date = datetime.now() - pd.DateOffset(days=1)

            # Define the start date as one year before the end date
            start_date = end_date - pd.DateOffset(days=365)

            # Download historical data
            data = yf.download(self.company_name, start=start_date, end=end_date + pd.DateOffset(days=1), interval="1d")

            # Print the first few rows of the data for debugging
            print(data.head())

            if data.empty:
                self.append_chat("Bot", "No historical data available.")
                return

            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)
            X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = data['Target']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict next value
            latest_data = pd.DataFrame([X.iloc[-1]], columns=X.columns)
            predicted_price = model.predict(latest_data)[0]

            # Calculate accuracy
            accuracy = model.score(X_test, y_test) * 100

            # Append results to chat
            self.append_chat("Bot", f"Next price prediction: {predicted_price:.2f}")
            self.append_chat("Bot", f"Accuracy of prediction: {accuracy:.2f}%")

            # Display prediction plot
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot historical data
            ax1.plot(data.index, data['Close'], label='Historical Prices', color='blue')

            # Plot predicted price as a horizontal line
            ax1.axhline(y=predicted_price, color='r', linestyle='--', label=f'Predicted Price: {predicted_price:.2f}')

            # Set the x-axis limits to the date range
            ax1.set_xlim(start_date, end_date)

            # Format the x-axis to show dates properly
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.grid(True)

            # Set labels and title
            ax1.set_title(f"{self.company_name} Price Prediction")
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()

            # Add message about prediction uncertainty
            ax1.text(0.5, 0.05, 'Note: Predictions are based on historical data and may vary.', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax1.transAxes, fontsize=12, color='black', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

            fig.autofmt_xdate()

            self.canvas = FigureCanvas(fig)

            # Create a scroll area
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setStyleSheet("border: none;")
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            scroll_layout.addWidget(self.canvas)
            scroll_widget.setLayout(scroll_layout)
            scroll_area.setWidget(scroll_widget)

            # Clear previous content before adding new scroll area
            self.clear_layout(self.graph_layout)
            self.graph_layout.addWidget(scroll_area)

        except Exception as e:
            self.append_chat("Bot", f"Error: {e}")

        self.restart_process()







    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    def fetch_and_plot_data(self, period):
        try:
            stock_data = yf.download(self.company_name, period=period)
            if stock_data.empty:
                raise ValueError("No data retrieved")

            self.plot_stock_data(stock_data)
            self.append_chat("Bot", "Data plotted successfully.")
        except Exception as e:
            self.append_chat("Bot", f"Error: {str(e)}")



    def clear_plot(self):
        if self.canvas:
            self.canvas.deleteLater()
            self.canvas = None

    def display_plot(self, data, title):
        if self.canvas:
            self.canvas.deleteLater()
            self.canvas = None

        style.use('ggplot')  

        data = data.reset_index()
        data['Date'] = mdates.date2num(pd.to_datetime(data['Date']))
        ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']].values

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot candlestick chart
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.grid(True)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')

        ax1.set_title(title)
        fig.autofmt_xdate()

        # Annotate data points
        for i, row in data.iterrows():
            ax1.annotate(f'{row["Close"]:.2f}', (row["Date"], row["Close"]),
                        textcoords="offset points", xytext=(0,5), ha='center',
                        fontsize=8, color='black', weight='bold')

        self.canvas = FigureCanvas(fig)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none;")
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.addWidget(self.canvas)
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)

        # Clear previous content before adding new scroll area
        self.clear_layout(self.graph_layout)
        self.graph_layout.addWidget(scroll_area)

        # Add button for detailed information if not already present
        if not hasattr(self, 'info_button'):
            self.info_button = QPushButton("Show Details")
            self.info_button.setStyleSheet("""
                background-color: #3498DB;
                color: #ECF0F1;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Roboto';
                font-size: 16px;
                margin-top: 10px;
            """)
            self.info_button.clicked.connect(self.show_details)
            self.graph_layout.addWidget(self.info_button)


    def show_details(self):
        # Clear previous plot
        self.clear_plot()
        
        # Create a detailed plot
        self.display_detailed_plot()

        # Add button to go back to normal graph
        if not hasattr(self, 'back_button'):
            self.back_button = QPushButton("Back to Normal")
            self.back_button.setStyleSheet("""
                background-color: #E74C3C;
                color: #ECF0F1;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Roboto';
                font-size: 16px;
                margin-top: 10px;
            """)
            self.back_button.clicked.connect(self.go_back_to_normal)
            self.graph_layout.addWidget(self.back_button)

    def display_detailed_plot(self):
        try:
            # Fetch the data again (you may want to adjust this as per your requirements)
            data = yf.download(self.company_name, period="1y", interval="1d")
            data = data.reset_index()
            data['Date'] = mdates.date2num(pd.to_datetime(data['Date']))
            ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']].values

            fig, ax1 = plt.subplots(figsize=(12, 8))

            # Plot candlestick chart
            candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
            ax1.grid(True)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')

            ax1.set_title(f"{self.company_name} Detailed Analysis")
            fig.autofmt_xdate()

            # Annotate specific data points (e.g., first, middle, and last)
            selected_points = [0, len(data)//2, len(data)-1]
            for i in selected_points:
                row = data.iloc[i]
                ax1.annotate(f'{row["Close"]:.2f}', (row["Date"], row["Close"]),
                            textcoords="offset points", xytext=(0,5), ha='center',
                            fontsize=8, color='black', weight='bold')

            self.canvas = FigureCanvas(fig)

            # Create a scroll area
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setStyleSheet("border: none;")
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            scroll_layout.addWidget(self.canvas)
            scroll_widget.setLayout(scroll_layout)
            scroll_area.setWidget(scroll_widget)

            # Clear previous content before adding new scroll area
            self.clear_layout(self.graph_layout)
            self.graph_layout.addWidget(scroll_area)

        except Exception as e:
            self.append_chat("Bot", f"Error: {e}")



    def go_back_to_normal(self):
        # Clear detailed plot and button
        self.clear_plot()
        if hasattr(self, 'back_button'):
            self.back_button.deleteLater()
            del self.back_button  # Ensure that the reference to back_button is removed

        # Return to normal plot based on the selected period
        if self.selected_period:
            self.display_plot(yf.download(self.company_name, period=self.selected_period, interval="1d"), f"{self.company_name} Prices for {self.selected_period}")

        # Add back the "Show Details" button if not already present
        if not hasattr(self, 'info_button'):
            self.info_button = QPushButton("Show Details")
            self.info_button.setStyleSheet("""
                background-color: #3498DB;
                color: #ECF0F1;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Roboto';
                font-size: 16px;
                margin-top: 10px;
            """)
            self.info_button.clicked.connect(self.show_details)
            self.graph_layout.addWidget(self.info_button)

    def restart_process(self):
        self.show_choice_buttons()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockChatBot()
    window.show()
    sys.exit(app.exec_())
