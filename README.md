# Thesis
## Title
**Analyzing code bugs based on method call graphs**

## Abstract
The increasing size and complexity of modern software projects often leads to the appearance of runtime errors (crashes), for instance due to coding inaccuracies or unforeseen use cases. Since errors affect software usability, quickly dealing with them has become an important maintenance task tied to the success of software projects. At the same time, processes for parsing user feedback, for example by dedicated teams, to understand errors or other bags and initiate maintenance operations can prove time-consuming. To mitigate associated costs, an emerging trend is to automate (parts of) error understanding with machine learning systems, for example that perform automatic tagging.
\
\
In this thesis, we focus on understanding errors through extracted latent representations; these can be inputted in machine learning systems to predict error qualities, such as recommending which tags errors should obtain. To achieve this, existing approaches in the broader scope of automated bug understanding make use of natural language processing techniques, such as word embeddings, to understand feedback texts. However, in the case of errors, we propose that available stack traces leading up to crashing code segments also capture useful coding semantics in the form of paths within function call graphs. Thus, we investigate whether graph embeddings---extracted from error stack traces---can be used to obtain a better understanding of errors.
\
\
To test our hypothesis, we developed a system that extracts latent error representations of software projects that combine textual and stack trace embeddings. To verify that these improve error understanding compared to using textual features only, we experimented on three popular software GitHub projects, where we extracted error represenations and used them to predict error tags (e.g. high priority) with neural network predictors. We found that, given a robust selection of predictor and enough example errors to train from, our approach improves text-based tagging by a significant margin across popular recommendation system measures.

## Technologies
Project is created with:
* Python version: 3.8
* Tensorflow version: 1.15

## Dependencies

```
pip install numpy
pip install pandas
pip install pickle
pip install -U nltk
pip install beautifulsoup4
pip install -U scikit-learn
pip install --upgrade tensorflow

```
## Licence
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
