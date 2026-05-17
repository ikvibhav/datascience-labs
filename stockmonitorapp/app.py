import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib import style
from models.moving_average import moving_average
from models.percentage_changes import yearly_percentage_change
from utils.base_stock_reader import Snp500Reader

style.use("ggplot")


@st.cache_data
@st.cache_resource
def fetch_stock_data_period(stock_symbol, period):
    stock_object = Snp500Reader()
    return stock_object.get_web_stock_data_period(stock_symbol, period)


def main():
    st.title("Stock Monitor")
    st.write("Welcome to the Stock Monitor App!")
    st.write(
        "This app will allow you to track the stock prices of your favorite companies."
    )
    st.write("Please select the stock you would like to track from the sidebar.")

    stock_list = ["^GSPC", "AMZN", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"]
    st.sidebar.title("1. Time Series Analysis")
    interactive_chart = st.sidebar.checkbox("Interactive Chart", value=False)
    selected_stock = st.sidebar.selectbox("Select a stock", stock_list)
    period = st.sidebar.radio(
        "Select the period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    )
    data = fetch_stock_data_period(selected_stock, period)

    # Add a checkbox for Correlation Analysis
    st.sidebar.title("2. Correlation Analysis")
    correlanalysis = st.sidebar.checkbox("Correlation Analysis", value=False)

    # Add a checkbox for Yearly Percentage Changes
    st.sidebar.title("3. Yearly Percentage Changes")
    yearlypct = st.sidebar.checkbox("Yearly Percentage Changes", value=False)

    if data is not None and not data.empty:

        st.subheader(f"Stock Data for {selected_stock} for last 5 days")
        st.write(data.tail())

        # Calculate moving averages
        data["50ma"] = moving_average(data["Close"], 50)
        data["200ma"] = moving_average(data["Close"], 200)

        # Calculate moving averages
        data["50ma"] = data["Close"].rolling(window=50).mean()
        data["200ma"] = data["Close"].rolling(window=200).mean()

        # Plot 50-day and 200-day moving averages as well as the closing price
        st.subheader(f"Time Series Plot for {selected_stock}")
        st.write(
            "Golden Cross - 50 day moving average (short term) crosses above the 200 day moving average (long term). This is a bullish signal."
        )
        st.write(
            "Death Cross - 50 day moving average (short term) crosses below the 200 day moving average (long term)"
        )

        if interactive_chart:
            # Create a DataFrame with the columns to plot
            plot_data = data[["Close", "50ma", "200ma"]]
            st.line_chart(plot_data)
        else:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [5, 1]},
            )

            ax1.plot(data.index, data["Close"], label="Close Price")
            ax1.plot(data.index, data["50ma"], label="50 day moving average")
            ax1.plot(data.index, data["200ma"], label="200 day moving average")
            ax1.legend()
            ax1.set_ylabel("Price")

            ax2.bar(data.index, data["Volume"], label="Volume")
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Date")

            st.pyplot(fig)
    else:
        st.write("No data available for the selected stock and date range.")

    if correlanalysis:
        period_corr = st.sidebar.radio(
            "Select the period", ["1mo", "3mo", "6mo", "1y", "2y"]
        )
        st.subheader("Correlation Analysis")
        st.write(f"Stocks - {stock_list}, Period - {period_corr}")
        stock_object = Snp500Reader()
        df_corr = stock_object.correlation_analysis(stock_list, period_corr)
        fig, ax = plt.subplots()
        sns.heatmap(df_corr, cmap="coolwarm", annot=True)
        # Add a title to the heatmap
        ax.set_title(f"Correlation Matrix ({period_corr})")
        st.pyplot(fig)

    if yearlypct:
        period_corr = st.sidebar.radio("Select the period (years)", ["5", "10", "20"])
        st.subheader("Yearly Percentage Changes")
        st.write(f"Selected Period - {period_corr} years")
        df_pct = yearly_percentage_change(stock_list)
        st.write(df_pct.tail(int(period_corr)))


if __name__ == "__main__":
    main()
