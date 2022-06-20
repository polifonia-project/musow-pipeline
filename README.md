---
component-id: musow-pipeline
name: musoW discovery pipeline
description: A pipeline to discover music resources on the Web and recommend candidates to be included in musoW
type: Application
release-date: 2022-06-13
release-number: latest
work-package: WP1
keywords:
  - recommendation
  - twitter
  - registry
licence: ISC
release link: https://github.com/polifonia-project/musow-pipeline/releases/latest
credits: Laurent Fintoni (UNIBO), Marilena Daquino (UNIBO)
related-components:
  - musoW
  - clef
---

# musoW discovery pipeline

[![DOI](https://zenodo.org/badge/471056103.svg)](https://zenodo.org/badge/latestdoi/471056103)

A pipeline to discover and classify music collections and archives available on the web and recommend them to the musoW catalogue.

The pipeline for populating musoW includes (1) a Twitter search engine for posts addressing music resources, (2) a classifier of tweets, (3) a web scraper of music web resources referenced in tweets, (4) a text summariser of descriptions included in scraped websites, (5) and classifier of descriptions, and (6) a recommending system of resources to be manually reviewed and included in musoW. 

Currently, the pipeline is stored as a [Jupyter notebook](https://github.com/polifonia-project/musow-pipeline/blob/master/NOTEBOOKS/twitter_pipeline_v2.ipynb) for reproducibility purposes. The script is run on a regular basis to collect new music resources. It is based on Logistic Regression and TF-IDF as the main feature to classify texts. We used cross-validation (10 folds) to avoid overfitting. Currently, we do not perform any particular hyperparameters tuning other than tokenizing bigrams and limiting the number of tokens to 1000.

The pipeline is part of [Polifonia](https://polifonia-project.eu/) H2020 project (described in Deliverable 1.9).

Cite this repository as follows:

```
Laurent Fintoni, and Marilena Daquino. (2022). musoW discovery pipeline (v0.1). DOI: 10.5281/zenodo.6637336
```

