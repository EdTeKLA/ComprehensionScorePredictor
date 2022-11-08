# Predicting reading comprehension scores
**Note:** This is an implementation of the neural network model presented in the paper, [Predicting Reading
Comprehension Scores of Elementary Students](https://educationaldatamining.org/edm2022/proceedings/2022.EDM-long-papers.14/index.html).

# File & Script definitions
* `generate_data.ipynb` - the master script to pre-compute all text-based features and combine them with skill-based
features for each student.
* `model.ipynb` - Script that defines all the models used. This notebook also includes model training, testing, error
analysis, feature importance, pair-wise model comparison by McNemar test.
* `childrens_book_dataset` - Children's Book Test [dataset](https://research.facebook.com/downloads/babi/) by Meta research group
* `awl.txt` - academic word list dataset (experimental, not used in this project)

# Usage
To generate model results, the two notebooks has to be run in the order of first `generate_data.ipynb`, then `model.ipynb`.
Manual index switching is needed to run different combination of "Base + single text feature" models. This is done by changing which tensor index are used by the neural network and the other tensors will simply be skipped by the model.

