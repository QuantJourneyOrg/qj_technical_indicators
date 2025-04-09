# Technical Indicators Library 🚀

Welcome to the **Technical Indicators Library**, a high-performance, open-source collection of financial technical indicators. Originally developed as part of the [QuantJourney project](https://quantjourney.substack.com) by Jakub Polec, it’s now freely available for everyone to use, fork, and improve under the MIT License.

## 🌟 Features
- **Comprehensive Indicators**: Includes SMA, EMA, RSI, MACD, and many more.
- **Optimized Performance**: Powered by [Numba](https://numba.pydata.org/) for fast calculations.
- **No TA-Lib Required**: Built from the ground up with NumPy, Pandas, and yfinance.
- **Execution Timing**: Features a decorator to measure and log computation times.
- **Open Source**: Licensed under MIT for unrestricted use in any project.

## 🎯 Getting Started

### Prerequisites
- Python 3.8 or higher
- Required packages: `numpy`, `pandas`, `yfinance`, `numba`, `matplotlib` (optional for plotting)

Install dependencies with:
```bash
pip install numpy pandas yfinance numba matplotlib
```

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/QuantJourneyOrg/qj_technical_indicators.git
   cd qj_technical_indicators
   ```
2. Run the example script:
   ```bash
   python try_technical_indicators.py
   ```
   This fetches PL data from yfinance and computes indicators, logging execution times.

## 📋 Example Output
```
2025-04-09 10:00:00 - INFO - Fetching PL data from yfinance...
2025-04-09 10:00:01 - INFO - Calculating SMA...
2025-04-09 10:00:01 - INFO - Finished SMA in 0.0023 seconds
2025-04-09 10:00:01 - INFO - SMA sample:
Date
2024-12-27    150.23
2024-12-30    150.45
...
```

## 🤝 Contributing
We’d love your help! Here’s how:
- **Fork** the repo
- **Create** a feature branch (`git checkout -b feature/awesome-indicator`)
- **Commit** your changes (`git commit -m "Add Awesome Indicator"`)
- **Push** to the branch (`git push origin feature/awesome-indicator`)
- **Open** a Pull Request

Have questions? Reach out to Jakub at [jakub@quantjourney.pro](mailto:jakub@quantjourney.pro).

## 📜 License
This project is licensed under the MIT License. Feel free to use it in your personal or commercial projects!

## Acknowledgements
- Inspired by the QuantJourney community
- Built by Jakub Polec
- Thanks to the open-source community for tools like NumPy and Numba!

Happy coding, and let’s make financial analysis faster and better together!