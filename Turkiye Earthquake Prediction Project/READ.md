# 🇹🇷 Turkey Earthquake Analysis & Forecasting (1915–2021)

This project analyzes the historical earthquake data of Turkey from **1915 to 2021** and performs a time series forecast using ARIMA to predict possible earthquake trends in the future.

🔗 **[View the project on Kaggle →](https://www.kaggle.com/code/behiyebedir/turkiye-earthquake-prediction-project)**  
📂 **Dataset Source**: [`turkey_earthquakes(1915-2021).csv`](https://www.kaggle.com/datasets/dhruvildave/turkey-earthquakes1915-2021)

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

---

## 📌 Sample Graphs

- ![Earthquakes per Year](your-graph-link-here)
- ![Magnitude vs Depth](your-graph-link-here)
- ![ARIMA Forecast](your-graph-link-here)

(*Replace placeholders with actual images or delete this section if not using it on GitHub.*)

---

## 📎 Dataset Details

- `Olus tarihi`: Earthquake date  
- `Mw`: Magnitude on the Moment Magnitude Scale  
- `Derinlik`: Depth in kilometers  
- Other columns include location and additional metadata

---

## 📍 Conclusion

This project provides a comprehensive look at the seismic activity in Turkey. Using both statistical insights and time series modeling, it gives valuable perspectives on how earthquake behavior has evolved — and what might lie ahead.
