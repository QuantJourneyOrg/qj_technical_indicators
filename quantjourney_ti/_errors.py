"""
QuantJourney Technical-Indicators - Errors
=========================================
Custom exceptions that provide rich context when indicator calculations or input validations fail.

Author: Jakub Polec <jakub@quantjourney.pro>
License: MIT
"""
from typing import Dict, Any, Optional
import json
from datetime import datetime


def _serialize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert non-serializable objects in context to strings for JSON serialization.

    Args:
        context: Dictionary containing context data.

    Returns:
        Dictionary with all values converted to JSON-serializable types.
    """
    serialized = {}
    for key, value in context.items():
        try:
            json.dumps(value)
            serialized[key] = value
        except TypeError:
            serialized[key] = str(value)
    return serialized


class IndicatorCalculationError(Exception):
    """
    Custom exception for errors during indicator calculations.

    Attributes:
        indicator: Name of the indicator that failed.
        message: Error message describing the failure.
        context: Additional context (e.g., symbol, parameters).
    """
    def __init__(self, indicator: str, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.

        Args:
            indicator: Name of the indicator (e.g., 'SMA', 'RSI').
            message: Description of the error.
            context: Optional dictionary with additional context (e.g., {'symbol': 'AAPL', 'period': 20}).
        """
        self.indicator = indicator
        self.message = message
        self.context = context or {}
        super().__init__(f"Indicator '{indicator}' failed: {message}")

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the error to a JSON-serializable dictionary.

        Returns:
            Dictionary with indicator, message, context, and timestamp.
        """
        return {
            "type": "IndicatorCalculationError",
            "indicator": self.indicator,
            "message": self.message,
            "context": _serialize_context(self.context),
            "timestamp": datetime.now().isoformat()
        }


class InvalidInputError(Exception):
    """
    Custom exception for invalid input data during indicator calculations or validations.

    Attributes:
        message: Error message describing the invalid input.
        context: Additional context (e.g., column names, data shape).
    """
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.

        Args:
            message: Description of the invalid input.
            context: Optional dictionary with additional context (e.g., {'missing_columns': ['high', 'low']}).
        """
        self.message = message
        self.context = context or {}
        super().__init__(message)

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the error to a JSON-serializable dictionary.

        Returns:
            Dictionary with error type, message, context, and timestamp.
        """
        return {
            "type": "InvalidInputError",
            "message": self.message,
            "context": _serialize_context(self.context),
            "timestamp": datetime.now().isoformat()
        }