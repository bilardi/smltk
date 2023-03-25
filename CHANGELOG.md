# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/bilardi/smltk/compare/v2.2.4...HEAD
[2.2.4]: https://github.com/bilardi/smltk/releases/tag/v2.2.3...v2.2.4
[2.2.3]: https://github.com/bilardi/smltk/releases/tag/v2.2.2...v2.2.3
[2.2.2]: https://github.com/bilardi/smltk/releases/tag/v2.2.1...v2.2.2
[2.2.1]: https://github.com/bilardi/smltk/releases/tag/v2.2.0...v2.2.1
[2.2.0]: https://github.com/bilardi/smltk/releases/tag/v2.1.0...v2.2.0
[2.1.0]: https://github.com/bilardi/smltk/releases/tag/v2.0.0...v2.1.0
[2.0.0]: https://github.com/bilardi/smltk/releases/tag/v1.2.1...v2.0.0
[1.2.1]: https://github.com/bilardi/smltk/releases/tag/v1.2.0...v1.2.1
[1.2.0]: https://github.com/bilardi/smltk/releases/tag/v1.1.1...v1.2.0
[1.1.1]: https://github.com/bilardi/smltk/releases/tag/v1.1.0...v1.1.1
[1.1.0]: https://github.com/bilardi/smltk/releases/tag/v1.0.1...v1.1.0
[1.0.1]: https://github.com/bilardi/smltk/releases/tag/v1.0.0...v1.0.1
[1.0.0]: https://github.com/bilardi/smltk/releases/tag/v0.1.0...v1.0.0
[0.1.0]: https://github.com/bilardi/smltk/releases/tag/v0.0.3...v0.1.0
[0.0.3]: https://github.com/bilardi/smltk/releases/tag/v0.0.2...v0.0.3
[0.0.2]: https://github.com/bilardi/smltk/releases/tag/v0.0.1...v0.0.2
[0.0.1]: https://github.com/bilardi/smltk/releases/tag/v0.0.1
