import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "ULKER.IS")
selected_Stock = st.selectbox("Select data set for prediction", stocks)
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)  # this returns data as pandas dataframe
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_Stock)
data_load_state.text("Loading data...DONE!")


st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


#forecasting with facebookprophet

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close": "y"})


m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

