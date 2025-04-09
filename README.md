# Technical Indicators Library 🚀

Welcome to the **Technical Indicators Library**! This is a high-performance, open-source implementation of financial technical indicators, originally crafted as part of the [QuantJourney project](https://quantjourney.substack.com) by Jakub. Now, it’s free for everyone to use, fork, and enhance under the MIT License! 🎉

## 🌟 Features
- **Wide Range of Indicators**: From SMA and EMA to RSI, MACD, and beyond! 📈
- **High Performance**: Optimized with [Numba](https://numba.pydata.org/) for blazing-fast calculations. ⚡
- **No TA-Lib Dependency**: Built from scratch using NumPy, Pandas, and yfinance. 🛠️
- **Timing Insights**: Includes a decorator to measure execution time. ⏱️
- **Open Source**: MIT License means you can use it anywhere, anytime! 🌍

## 🎯 Getting Started

### Prerequisites
- Python 3.8+
- Required packages: `numpy`, `pandas`, `yfinance`, `numba`

Install them with:
```bash
pip install numpy pandas yfinance numba
```

### Usage
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/technical-indicators.git
   cd technical-indicators
   ```
2. Run the example script:
   ```bash
   python try_technical_indicators.py
   ```
   Watch it fetch PL data and compute indicators with timing logs! 📊

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
- **Fork** the repo 🍴
- **Create** a feature branch (`git checkout -b feature/awesome-indicator`)
- **Commit** your changes (`git commit -m "Add Awesome Indicator"`)
- **Push** to the branch (`git push origin feature/awesome-indicator`)
- **Open** a Pull Request 🎁

Have questions? Reach out to Jakub at [jakub@quantjourney.pro](mailto:jakub@quantjourney.pro).

## 📜 License
This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) for details. Feel free to use it in your personal or commercial projects! 💼

## 🌈 Acknowledgements
- Inspired by the QuantJourney community
- Built with ❤️ by Jakub Polec
- Thanks to the open-source community for tools like NumPy and Numba!

Happy coding, and let’s make financial analysis faster and better together! 🚀