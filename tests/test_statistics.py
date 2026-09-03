"""
Unit tests for StatVerify AI Statistical Methods.
"""

import unittest
import pandas as pd
from src.models import StatisticalDataset
from src.statistical_methods.parametric import ParametricMethods
from src.statistical_methods.categorical import CategoricalMethods
from src.statistical_methods.correlation import CorrelationMethods
from src.statistical_methods.non_parametric import NonParametricMethods
from src.statistical_methods.regression import RegressionMethods


class TestStatisticalMethods(unittest.TestCase):

    def test_independent_t_test(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'g': ['A']*15 + ['B']*15, 'v': list(range(15)) + list(range(10, 25))}),
            groups={'A': list(range(15)), 'B': list(range(10, 25))},
            sample_size=30,
        )
        res = ParametricMethods.independent_t_test(ds)
        self.assertEqual(res.test_name, "Independent t-test")
        self.assertIsNotNone(res.statistic)
        self.assertIsNotNone(res.p_value)
        self.assertIsNotNone(res.effect_size)
        self.assertIsNotNone(res.confidence_interval)

    def test_one_way_anova(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'g': ['A']*10 + ['B']*10 + ['C']*10, 'v': list(range(10)) + list(range(5, 15)) + list(range(15, 25))}),
            groups={'A': list(range(10)), 'B': list(range(5, 15)), 'C': list(range(15, 25))},
            sample_size=30,
        )
        res = ParametricMethods.one_way_anova(ds)
        self.assertEqual(res.test_name, "One-Way ANOVA")
        self.assertIsNotNone(res.statistic)
        self.assertIsNotNone(res.p_value)
        self.assertIsNotNone(res.effect_size)

    def test_paired_t_test(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'x': list(range(15)), 'y': [i + 2 for i in range(15)]}),
            x=list(range(15)),
            y=[i + 2 for i in range(15)],
            sample_size=15,
        )
        res = ParametricMethods.paired_t_test(ds)
        self.assertEqual(res.test_name, "Paired t-test")
        self.assertIsNotNone(res.statistic)

    def test_pearson_correlation(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'x': list(range(20)), 'y': [i * 2 + 1 for i in range(20)]}),
            x=list(range(20)),
            y=[i * 2 + 1 for i in range(20)],
            sample_size=20,
        )
        res = CorrelationMethods.pearson(ds)
        self.assertEqual(res.test_name, "Pearson Correlation")
        self.assertAlmostEqual(res.statistic, 1.0, places=3)
        self.assertIsNotNone(res.confidence_interval)

    def test_spearman_correlation(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'x': list(range(20)), 'y': [i * 2 + 1 for i in range(20)]}),
            x=list(range(20)),
            y=[i * 2 + 1 for i in range(20)],
            sample_size=20,
        )
        res = CorrelationMethods.spearman(ds)
        self.assertEqual(res.test_name, "Spearman Correlation")
        self.assertAlmostEqual(res.statistic, 1.0, places=3)

    def test_mann_whitney(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'g': ['A']*15 + ['B']*15, 'v': list(range(15)) + list(range(10, 25))}),
            groups={'A': list(range(15)), 'B': list(range(10, 25))},
            sample_size=30,
        )
        res = NonParametricMethods.mann_whitney(ds)
        self.assertEqual(res.test_name, "Mann-Whitney U")
        self.assertIsNotNone(res.statistic)
        self.assertIsNotNone(res.effect_size)

    def test_kruskal_wallis(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'g': ['A']*10 + ['B']*10 + ['C']*10, 'v': list(range(10)) + list(range(5, 15)) + list(range(15, 25))}),
            groups={'A': list(range(10)), 'B': list(range(5, 15)), 'C': list(range(15, 25))},
            sample_size=30,
        )
        res = NonParametricMethods.kruskal(ds)
        self.assertEqual(res.test_name, "Kruskal-Wallis")
        self.assertIsNotNone(res.statistic)

    def test_linear_regression(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'x': list(range(20)), 'y': [i * 2.5 + 1.0 for i in range(20)]}),
            x=list(range(20)),
            y=[i * 2.5 + 1.0 for i in range(20)],
            sample_size=20,
        )
        res = RegressionMethods.linear_regression(ds)
        self.assertEqual(res.test_name, "Linear Regression")
        self.assertIsNotNone(res.statistic)
        self.assertEqual(res.additional_metrics.get('slope'), 2.5)

    def test_logistic_regression(self):
        ds = StatisticalDataset(
            dataframe=pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'y': [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]}),
            x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            y=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
            sample_size=10,
        )
        res = RegressionMethods.logistic_regression(ds)
        self.assertEqual(res.test_name, "Logistic Regression")
        self.assertIsNotNone(res.statistic)
        self.assertIsNotNone(res.effect_size)


if __name__ == "__main__":
    unittest.main()
