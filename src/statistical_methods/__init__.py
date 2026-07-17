"""
Statistical Methods Package
"""

from .parametric import ParametricMethods
from .categorical import CategoricalMethods
from .correlation import CorrelationMethods
from .non_parametric import NonParametricMethods
from .regression import RegressionMethods

__all__ = [
    "ParametricMethods",
    "CategoricalMethods",
    "CorrelationMethods",
    "NonParametricMethods",
    "RegressionMethods",
]