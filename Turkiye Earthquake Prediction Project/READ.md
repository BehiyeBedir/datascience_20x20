# 🇹🇷 Turkey Earthquake Analysis & Forecasting (1915–2021)

This project analyzes the historical earthquake data of Turkey from **1915 to 2021** and performs a time series forecast using ARIMA to predict possible earthquake trends in the future.

📂 **Dataset Source**: 🔗 https://www.kaggle.com/code/behiyebedir/turkiye-earthquake-prediction-project  


---

## 🧠 Project Goals

- Explore earthquake patterns over time
- Visualize trends in magnitude and depth
- Investigate relationships between earthquake characteristics
- Use ARIMA to forecast future earthquake frequency

---

## 🛠️ Technologies Used

- **Python**: `pandas`, `matplotlib`, `seaborn`, `statsmodels`
- **Time Series Analysis**: ADF Test, Differencing, ACF/PACF, ARIMA
- **Data Source**: CSV data with earthquake records in Turkey (1915–2021)

---

## 📊 Exploratory Data Analysis

✅ **Data Cleaning**
- Converted date strings to datetime format
- Extracted `year` and `month` from timestamps
- Removed or handled missing values

✅ **Visualizations**
- 📈 Earthquakes per year  
- 📉 Average magnitude and depth per year  
- 🔁 Magnitude vs. depth scatter plot  
- 🔥 Monthly heatmap of earthquake counts  

---

## 📈 Time Series Forecasting (ARIMA)

- Conducted **ADF test** for stationarity
- Used **first-order differencing** to achieve stationarity
- Analyzed **ACF** and **PACF** plots to select ARIMA(p,d,q) parameters
- Fitted **ARIMA(2,1,2)** model
- Forecasted earthquake counts for the next **10 years**
- Visualized forecast with **confidence intervals**

-
