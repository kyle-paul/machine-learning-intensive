# Machine Learning Intensive

Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance without explicit programming. It has become an essential tool for solving complex problems in various domains, such as health care, finance, security, and entertainment. However, learning machine learning can be challenging, as it requires a solid foundation of mathematics, statistics, and programming skills. This book is designed for those whohave prior knowledge about machine learning (basic level) and want to pursue this field in an intensive way. It covers the most important concepts, techniques, and applications of machine learning, from supervised and unsupervised learning to deep learning and reinforcement learning. It also provides practical examples and exercises to help you apply what you learn to real-world scenarios. By reading this book, you will gain a comprehensive understanding of machine learning and its potential. You will also learn how to use popular machine learning libraries and frameworks, such as TensorFlow, PyTorch, Scikit-learn, and Keras. Whether you are a student, a researcher, or a professional, this book will help you master machine learning and become a confident machine learning practitioner.

## Usage

### Building the book

If you'd like to develop and/or build the Machine Learning Intensive book, you should:

1. Clone this repository
2. Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
3. (Optional) Edit the books source files located in the `machine-learning-intensive/` directory
4. Run `jupyter-book clean machine-learning-intensive/` to remove any existing builds
5. Run `jupyter-book build machine-learning-intensive/`

A fully-rendered HTML version of the book will be built in `machine-learning-intensive/_build/html/`.

### Hosting the book

Please see the [Jupyter Book documentation](https://jupyterbook.org/publish/web.html) to discover options for deploying a book online using services such as GitHub, GitLab, or Netlify.

For GitHub and GitLab deployment specifically, the [cookiecutter-jupyter-book](https://github.com/executablebooks/cookiecutter-jupyter-book) includes templates for, and information about, optional continuous integration (CI) workflow files to help easily and automatically deploy books online with GitHub or GitLab. For example, if you chose `github` for the `include_ci` cookiecutter option, your book template was created with a GitHub actions workflow file that, once pushed to GitHub, automatically renders and pushes your book to the `gh-pages` branch of your repo and hosts it on GitHub Pages when a push or pull request is made to the main branch.

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/kyle-paul/machine-learning-intensive/graphs/contributors).

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).
