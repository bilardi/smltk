"""The class for managing the data of the main repositories

A collection of methods to simplify your code.
"""

import klib
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import mutual_info_score


class DataAnalysis:
    """
    The class DataAnalysis contains methods to explore data analysis.

    Here's an example:

        >>> from smltk.data_analysis import DataAnalysis
        >>> df = pd.read_csv(filename)
        >>> da = DataAnalysis()
        >>> features = da.get_eda(df)
        >>> print(features.keys())
        ["data_amount", "hue_order", "categorical_features", "numerical_features", "data_missing", "correlations"]
    """

    sklearn = None

    def __init__(self):
        self.sklearn = load_iris()

    def pearson_corr(self, x: pd.Series, y: pd.Series):
        """Pearson (linear, continuous-continuous)"""
        return stats.pearsonr(x, y)[0]

    def spearman_corr(self, x: pd.Series, y: pd.Series):
        """Spearman (monotonic, continuous-ordinal)"""
        return stats.spearmanr(x, y)[0]

    def kendall_corr(self, x: pd.Series, y: pd.Series):
        """Kendall (monotone, ordinal)"""
        return stats.kendalltau(x, y)[0]

    def pointbiserial_corr(self, x: pd.Series, y: pd.Series):
        """point-biserial (binary vs continuous)"""
        return stats.pointbiserialr(x, y)[0]

    def phi_coeff(self, x: pd.Series, y: pd.Series):
        """Phi coefficient (2 binary variables)"""
        table = pd.crosstab(x, y)
        return stats.chi2_contingency(table)[1]

    def cramers_v(self, x: pd.Series, y: pd.Series):
        """Cramer's V (nominal categories)"""
        confusion_matrix = pd.crosstab(x, y).to_numpy()
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    def mutual_info(self, x: pd.Series, y: pd.Series):
        """mutual information (generic, discrete vs discrete)"""
        return mutual_info_score(x, y)

    def biserial_corr(self, x: pd.Series, y: pd.Series):
        """biserial (dichotomous, continuous-continuous)"""
        return pg.biserial(x, y)["r"].values[0]

    def choose_correlations(self, x: pd.Series, y: pd.Series) -> dict:
        """
        Choose correlations based on data type
        """
        # binary check
        # x_clean = self._to_numeric_binary(x)
        # y_clean = self._to_numeric_binary(y)
        # is_x_binary = x.nunique() == 2
        # is_y_binary = y.nunique() == 2

        # continuous-continuous
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(
            y.dtype, np.number
        ):
            return {
                "pearson": self.pearson_corr(x, y),
                "spearman": self.spearman_corr(x, y),
                "kendall": self.kendall_corr(x, y),
            }

        # # binary-continuous
        # if is_x_binary and np.issubdtype(y_clean.dtype, np.number):
        #     return {"pointbiserial": self.pointbiserial_corr(x_clean, y)}
        # if is_y_binary and np.issubdtype(x_clean.dtype, np.number):
        #     return {"pointbiserial": self.pointbiserial_corr(y_clean, x)}

        # # binary-binary
        # if is_x_binary and is_y_binary:
        #     return {
        #         "phi": self.phi_coeff(x, y),
        #         "cramers_v": self.cramers_v(x, y),
        #     }

        # nominal categories
        if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(
            y.dtype, np.number
        ):
            return {
                "cramers_v": self.cramers_v(x, y),
                "mutual_info": self.mutual_info(x, y),
            }

        return {"info": "Type not managed"}

    def get_correlations(
        self, data: pd.DataFrame, columns_to_use: list = []
    ) -> dict:
        """
        Calculate correlation among features

        Arguments:
            :data (Pandas DataFrame): features to analyze
            :columns_to_use (list[str]): list of features names to use
        Returns:
            dictionary of Pandas Dataframe of the correlations
        """
        if len(columns_to_use) == 0:
            columns_to_use = data.keys()
        correlation_matrixes = {}
        for index in columns_to_use:
            inline_correlation_matrix = {}
            for feature in columns_to_use:
                df = data.copy()
                if data[feature].isna().any():
                    df = data.dropna(subset=[feature])
                if df[index].isna().any():
                    df = df.dropna(subset=[index])
                correlations = self.choose_correlations(df[feature], df[index])
                for correlation in correlations:
                    if correlation == "info":
                        raise ValueError(
                            f"Type of {feature} and {index} not managed"
                        )
                    if correlation not in inline_correlation_matrix.keys():
                        inline_correlation_matrix[correlation] = []
                    inline_correlation_matrix[correlation].append(
                        correlations[correlation]
                    )
            for correlation in correlations:
                if correlation not in correlation_matrixes.keys():
                    correlation_matrixes[correlation] = pd.DataFrame()
                if (
                    len(inline_correlation_matrix[correlation])
                    == columns_to_use
                ):
                    correlation_matrixes[correlation] = pd.concat(
                        [
                            correlation_matrixes[correlation],
                            pd.DataFrame(
                                {
                                    index: inline_correlation_matrix[
                                        correlation
                                    ]
                                },
                                index=columns_to_use,
                            ),
                        ]
                    )
        return correlation_matrixes

    def get_features_info(
        self, target: str, data: pd.DataFrame, params: dict = {}
    ) -> dict:
        """
        Calculate features types and data missing percentage

        Arguments:
            :target (string): name of feature target
            :data (Pandas DataFrame): features to analyze
            :params (dict) with the keys below
            :columns_to_filter (list[str]): list of features names to filter

        Returns:
            dict with categorical features list and numerical features list
        """
        if "columns_to_filter" not in params.keys():
            params["columns_to_filter"] = []

        hue_order = sorted(data[target].unique().tolist())
        cat_features = []
        num_features = []
        for feature in data.columns:
            if feature in params["columns_to_filter"]:
                continue
            elif data[feature].dtype == type(object):
                cat_features.append(feature)
            else:
                num_features.append(feature)

        return {
            "data_amount": int(data[target].count()),
            "hue_order": hue_order,
            "categorical_features": cat_features,
            "numerical_features": num_features,
            "data_missing": (data.isnull().mean() * 100).to_dict(),
        }

    def get_eda(
        self, target: str, data: pd.DataFrame, params: dict = {}
    ) -> dict:
        """
        Plot images for exploratory data analysis

        Arguments:
            :target (string): name of feature target
            :data (Pandas DataFrame): features to analyze
            :params (dict): with the keys below
            :columns_to_filter (list[str]): list of features names to filter
            :color_palette (string): matplotlib colormap name, by default Set2
            :sample.frac (string): fraction of axis items to return
            :corr_plot.cmap (string): the mapping from data values to color space, matplotlib colormap name or object, or list of colors, by default viridis
            :missingval_plot.cmap (string): matplotlib colormap name or object, by default Set2

        Returns:
            plots and features slitted in categorical and numerical features
        """
        if "color_palette" not in params.keys():
            params["color_palette"] = "Set2"
        if "sample.frac" not in params.keys():
            params["sample.frac"] = 0.01
        if "corr_plot.cmap" not in params.keys():
            params["corr_plot.cmap"] = "viridis"
        if "missingval_plot.cmap" not in params.keys():
            params["missingval_plot.cmap"] = "Set2"

        sns.set_palette(sns.color_palette(params["color_palette"]))
        features = self.get_features_info(target, data, params)

        ## categorical features - bar charts matrix
        cat_features = features["categorical_features"]
        plt.figure(figsize=(10, len(cat_features) * 2))
        for i, col in enumerate(cat_features):
            plt.subplot(len(cat_features) // 2 + 1, 2, i + 1)
            sns.set_style("white")
            sns.countplot(
                x=col,
                hue=target,
                hue_order=features["hue_order"],
                data=data.sample(frac=params["sample.frac"]),
            )
            plt.title(f"{col} vs {target}")
            if len(features["hue_order"]) > 25:
                legend = plt.gca().get_legend()
                if legend is not None:
                    legend.remove()
            plt.tight_layout()
        plt.show()

        ## numerical features - pair plots matrix
        num_features = features["numerical_features"]
        if len(num_features) > 0:
            pairplot_features = (
                num_features
                if target in num_features
                else num_features + [target]
            )
            sns.pairplot(
                data[pairplot_features].sample(frac=params["sample.frac"]),
                hue=target,
                hue_order=features["hue_order"],
                corner=True,
            )

        ## filtered features - violin plots matrix
        filtered_features = cat_features + num_features
        fig, axes = plt.subplots(
            len(filtered_features) // 2,
            2,
            figsize=(10, len(filtered_features) * 2),
        )
        for i, ax in enumerate(axes.flatten()):
            sns.violinplot(
                x=target,
                y=filtered_features[i],
                data=data.sample(frac=params["sample.frac"]),
                ax=ax,
                hue=target,
                hue_order=features["hue_order"],
            )
            ax.set_title(f"{filtered_features[i]} vs {target}")
        plt.tight_layout()
        plt.show()

        ## filtered features - bar charts
        custom_palette = sns.color_palette(
            params["color_palette"], len(data[target].unique())
        )
        for feature in filtered_features:
            contingency_table = pd.crosstab(
                data[feature], data[target], normalize="index"
            )
            sns.set_style("white")
            contingency_table.plot(
                kind="bar", stacked=True, color=custom_palette, figsize=(20, 4)
            )
            plt.title(f"Percentage Distribution of Target across {feature}")
            plt.xlabel(feature)
            plt.ylabel("Percentage")
            plt.legend(title="Target Class")
            if len(features["hue_order"]) > 25:
                legend = plt.gca().get_legend()
                if legend is not None:
                    legend.remove()
            plt.show()

        ## all features
        columns_to_use = features["numerical_features"] + [target]
        correlations = self.get_correlations(data, columns_to_use)
        data_missing = pd.DataFrame(
            features["data_missing"].values(),
            columns=["data_missing"],
            index=features["data_missing"].keys(),
        )
        for correlation in correlations:
            if len(data_missing.columns) == len(correlations[correlation]):
                fig, (ax1, ax2) = plt.subplots(
                    1,
                    2,
                    figsize=(18, 8),
                    gridspec_kw={"width_ratios": [0.5, 15]},
                )
                fig.suptitle(correlation)
                correlation_data = pd.concat(
                    [correlations[correlation], data_missing], axis=1
                )
                sns.heatmap(
                    correlation_data[["data_missing"]],
                    cmap=params["corr_plot.cmap"],
                    annot=True,
                    cbar=False,
                    ax=ax1,
                    yticklabels=True,
                    xticklabels=True,
                )
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
                correlation_no_dm = correlation_data.drop(
                    columns="data_missing"
                )
                triangular_mask = np.triu(
                    np.ones_like((correlation_no_dm + correlation_no_dm.T) / 2)
                )
                sns.heatmap(
                    correlation_no_dm,
                    mask=triangular_mask,
                    cmap=params["corr_plot.cmap"],
                    annot=True,
                    cbar=True,
                    ax=ax2,
                    yticklabels=False,
                )
                plt.subplots_adjust(wspace=0.05)
        features["correlations"] = correlations

        if len(num_features) > 0:
            klib.corr_plot(
                data.sample(frac=params["sample.frac"]),
                cmap=params["corr_plot.cmap"],
            )
        klib.missingval_plot(
            data.sample(frac=params["sample.frac"]),
            cmap=params["missingval_plot.cmap"],
        )
        klib.cat_plot(data.sample(frac=params["sample.frac"]))

        return features
