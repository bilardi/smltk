# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/bilardi/smltk/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/bilardi/smltk/releases/tag/v1.1.1...v1.2.0
[1.1.1]: https://github.com/bilardi/smltk/releases/tag/v1.1.0...v1.1.1
[1.1.0]: https://github.com/bilardi/smltk/releases/tag/v1.0.1...v1.1.0
[1.0.1]: https://github.com/bilardi/smltk/releases/tag/v1.0.0...v1.0.1
[1.0.0]: https://github.com/bilardi/smltk/releases/tag/v0.1.0...v1.0.0
[0.1.0]: https://github.com/bilardi/smltk/releases/tag/v0.0.3...v0.1.0
[0.0.3]: https://github.com/bilardi/smltk/releases/tag/v0.0.2...v0.0.3
[0.0.2]: https://github.com/bilardi/smltk/releases/tag/v0.0.1...v0.0.2
[0.0.1]: https://github.com/bilardi/smltk/releases/tag/v0.0.1
