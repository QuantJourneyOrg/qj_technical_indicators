# Changelog
All notable changes to **QuantJourney Technical Indicators** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `numba_fallback` decorator in `_decorators.py` for robust Numba error handling.
- Fallbacks to pandas or NaN-filled outputs for all Numba-based indicator methods (`SMA`, `EMA`, `RSI`, `MACD`, `BB`, `ATR`, `STOCH`, `ADX`, `ICHIMOKU`, `KELTNER`, `MFI`, `TRIX`, `CCI`, `ROC`, `WILLR`, `DEMA`, `KAMA`, `DONCHIAN`, `AROON`, `AO`, `ULTIMATE_OSCILLATOR`, `CMO`, `DPO`, `MASS_INDEX`, `VWAP`, `SUPERTREND`, `PVO`, `HISTORICAL_VOLATILITY`, `CHAIKIN_VOLATILITY`, `LINEAR_REGRESSION_CHANNEL`, `AD`, `ALMA`, `KDJ`, `HEIKEN_ASHI`, `BETA`, `DI`, `ADOSC`, `VOLUME_INDICATORS`, `HULL_MA`, `PIVOT_POINTS`, `RAINBOW`).
- Comprehensive logging for fallback events and errors in `indicators.py`.
- Enhanced input validation in `_utils.py` for edge cases (empty data, NaNs, invalid columns).
- Unit tests for Numba and fallback paths in `tests/test_indicators.py` (assumed).

### Changed
- Moved `numba_fallback` decorator from `indicators.py` to `_decorators.py` for modularity.
- Fixed Numba compilation errors in `_indicator_kernels.py` for `ALMA`, `BB`, `CMO`, `COPPOCK`, `HISTORICAL_VOLATILITY`, `HULL_MA`, `MACD`, `RAINBOW`, `ROC`, `RSI`, `SMA`.
- Updated `_validate_and_get_prices` to prioritize `adj_close` and handle edge cases.
- Adjusted `RAINBOW` method to handle 2D array output from `_calculate_rainbow_numba`, creating a DataFrame with SMA columns.
- Improved edge-case handling in all Numba kernels to minimize fallback triggers.

### Fixed
- Resolved `_cffi_backend` import issue by ensuring proper virtual environment setup (assumed user action).
- Fixed column naming mismatches (`close` vs. `adj_close`) in input validation and tests.
- Corrected dimensionality issues in `RAINBOW` output.

## [0.2.0] - 2025-01-01
### Added
- Initial release of `quantjourney_ti` with Numba-optimized technical indicators.
- Support for 50+ indicators including `SMA`, `EMA`, `RSI`, `MACD`, etc.
- `TechnicalIndicators` class with singleton `_TI_INSTANCE` for performance.
- Basic input validation and plotting utilities in `_utils.py`.
- Numba kernels in `_indicator_kernels.py` for performance-critical calculations.
- Helper decorators (`timer`) in `_decorators.py`.

### Notes
- This is the initial public release under the MIT License.