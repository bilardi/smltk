# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.1.0] - 2026-04-29

### Added
- in DataAnalysis.get_eda, the params analyses.skip and plots.skip to filter heavy EDA blocks (skip block entirely or skip its plots only)
- DataAnalysis.VALID_BLOCK_NAMES class constant exposing the 7 semantic block names: cat_countplot, num_pairplot, feat_violinplots, feat_barplots, relations_heatmaps, missingval_plot, cat_plots
- pingouin as runtime dependency, used by DataAnalysis.biserial_corr

### Changed
- renamed DataAnalysis.choose_correlations/get_correlations to choose_relations/get_relations and the return key features["correlations"] to features["relations"], since mutual info is a similarity score and not a correlation
- DataProcessing.transform_categories detects categorical features more robustly (is_string_dtype/is_object_dtype plus element-wise string check)
- DataAnalysis.choose_relations and get_features_info use pd.api.types.is_numeric_dtype/is_string_dtype/is_object_dtype for compatibility with newer pandas (StringDtype)
- aligned Makefile build targets: removed twine from localbuild, added black/twine upgrade to buildtest and build

### Fixed
- DataAnalysis.get_relations inner length check (was comparing int to list, the inner block was never executed)
- DataProcessing.transform_categories now preserves np.nan/None/pd.NA as NaN; previously the -1 codes from pd.Categorical(...).codes leaked into downstream calculations

### Updated
- tests: added test_transformation_categories, plus 9 tests for the EDA skip filter (test_valid_block_names_constant, test_is_skipped_*, test_get_eda_unknown_*, test_get_eda_skip_*); updated test_get_eda for the rename and added PLOTS_SKIP env var helper
- documentation: updated DataAnalysis.get_eda docstring with the new params and fixed slitted -> split typo

## [3.0.0] - 2025-08-28

### Added
- EDA methods move on DataAnalysis
- split loading as needed
  - the class Ntk needs nltk and wordcloud, pip install "smltk[ntk]"
  - the class ObjectDetection needs torch, pip install "smltk[object_detection]"
  - the basic package not loads these packages, pip install smltk

### Changed
- import Ntk from data_processing
  - the class Ntk extends DataProcessing
  - removed inherited methods
  - renamed method named word_tokenize in method named tokenize
- import Indicator from feature_engineering
- renamed class Metrics in Modeling
  - import Modeling from modeling
- renamed class DataVisualization in ObjectDetection
  - import ObjectDetection from modeling
- package management from setup.py to pyprojet.toml

### Updated
- tests and documentation

## [2.2.11] - 2025-07-25

### Added
- new method for eda

### Changed
- in the usage documentation, how to synthesize timeseries with trend

## [2.2.10] - 2024-11-01

### Added
- new method for points where each directional change ends and updated notebook

## [2.2.9] - 2024-11-01

### Added
- new method for points where each directional change starts and updated plot and notebook

## [2.2.8] - 2024-10-19

### Added
- new methods for directional change in the class Indicator and DataVisualization

## [2.2.7] - 2023-03-25

### Fixed
- documentation

## [2.2.6] - 2023-03-25

### Added
- API documentation

## [2.2.5] - 2023-03-25

### Fixed
- documentation
- tests

## [2.2.4] - 2023-03-24

### Fixed
- documentation
- requirements

## [2.2.3] - 2023-03-23

### Added
- in the class DataVisualization, the methods bboxes_cxcywh_to_xyxy, rescale_bboxes, get_inference_objects, get_inference_objects_df, plot_inference_objects

## [2.2.2] - 2023-03-22

### Added
- the class DataVisualization for data management about public datasets

## [2.2.1] - 2022-01-22

### Added
- scoring and modeling methods to Metrics

## [2.2.0] - 2021-12-30

### Added
- metrics ROC_AUC and Support to Metrics.get_classification_metrics

## [2.1.0] - 2021-12-30

### Added
- metrics MCC to Metrics.get_classification_metrics

### Fixed
- requirement for nltk

## [2.0.0] - 2021-12-05

### Removed
- data transposition on sns.heatmap

## [1.2.1] - 2021-11-16

### Fixed
- parameter of self.get_ngrams_features on Ntk.get_features

## [1.2.0] - 2021-11-16

### Added
- parameter is_lemma on Ntk.get_ngrams
- some methods on Ntk to manage creation of ngrams feature from docs and tuples

### Changed
- in Ntk.get_features the ngrams creation directly by Ntk.get_ngrams_features

## [1.1.1] - 2021-11-15

### Added
- parameter on all methods that use Ntk.get_features to manage ngrams

## [1.1.0] - 2021-11-15

### Added
- method Ntk.get_ngrams
- parameter on Ntk.get_features to manage ngrams

## [1.0.1] - 2021-11-13

### Fixed
- packages for readthedocs deployment
- prediction when the X_test is empty or it starts with a non-zero index
## [1.0.0] - 2021-11-06

### Added
- in the class Ntk, the methods get_words_top, get_vocabs_cleaned, get_features_from_docs, create_features_from_docs, create_words_map, create_words_cloud

### Changed
- rename method from Ntk.find_features to Ntk.get_features and its return
- rename method from Ntk.create_features to Ntk.create_features_from_tuples and its return
- defaults of Ntk.vectorize_docs
- defaults of Metrics.create_confusion_matrix
- notebook named usage

### Fixed
- documentation of some methods

## [0.1.0] - 2021-10-26

### Added
- API documentation

### Changed
- documentation of all methods

### Fixed
- params type of Metrics.get_classification_metrics()

## [0.0.3] - 2021-10-26

### Added
- the class Metrics for testing models and reporting results

## [0.0.2] - 2021-10-26

### Added
- the class Ntk for data preprocessing with Natural Language Processing

## [0.0.1] - 2021-10-25

### Added
- the outline files
- the init files of package and tests
- the documentation by sphinx

[Unreleased]: https://github.com/bilardi/smltk/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/bilardi/smltk/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/bilardi/smltk/compare/v2.2.11...v3.0.0
[2.2.11]: https://github.com/bilardi/smltk/compare/v2.2.10...v2.2.11
[2.2.10]: https://github.com/bilardi/smltk/compare/v2.2.9...v2.2.10
[2.2.9]: https://github.com/bilardi/smltk/compare/v2.2.8...v2.2.9
[2.2.8]: https://github.com/bilardi/smltk/compare/v2.2.7...v2.2.8
[2.2.7]: https://github.com/bilardi/smltk/compare/v2.2.6...v2.2.7
[2.2.6]: https://github.com/bilardi/smltk/compare/v2.2.5...v2.2.6
[2.2.5]: https://github.com/bilardi/smltk/compare/v2.2.4...v2.2.5
[2.2.4]: https://github.com/bilardi/smltk/compare/v2.2.3...v2.2.4
[2.2.3]: https://github.com/bilardi/smltk/compare/v2.2.2...v2.2.3
[2.2.2]: https://github.com/bilardi/smltk/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/bilardi/smltk/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/bilardi/smltk/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/bilardi/smltk/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/bilardi/smltk/compare/v1.2.1...v2.0.0
[1.2.1]: https://github.com/bilardi/smltk/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/bilardi/smltk/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/bilardi/smltk/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/bilardi/smltk/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/bilardi/smltk/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/bilardi/smltk/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/bilardi/smltk/compare/v0.0.3...v0.1.0
[0.0.3]: https://github.com/bilardi/smltk/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/bilardi/smltk/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/bilardi/smltk/compare/v0.0.1
