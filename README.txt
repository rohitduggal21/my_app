my_app (Sentiment Analysis App)
======

Getting Started
---------------

- Change directory into your newly created project if not already there. Your
  current directory should be the same as this README.txt file and setup.py.

    cd my_app

- Create a Python virtual environment.

    python3 -m venv env

- Upgrade packaging tools.

    env/bin/pip install --upgrade pip setuptools

- Install the project in editable mode with its testing requirements.

    env/bin/pip install -e ".[testing]"

- Run your project's tests.

    env/bin/pytest

- Run your project.

    env/bin/pserve development.ini

Added in v1.0
---------------
- data

    folder for training data files

-  Screenshots

    screenshots/demo demonstrating the working of the app

-  model.rd

    trained model dumped by trained_routine.py

- model_def.py

    module containing the class which handles the training part

- train_routine.py

    module for training a model, intructions for training the model:
    1) get inside my_app
    2) execute python3 train_routine.py <training_data_file_name> <text_column> <target_column>

- NOTE:

    python version 3.6 may throw an exception => AttributeError: module 'enum' has no attribute 'IntFlag'
    Uninstall enum34: pip3 uninstall -y enum34
    More on this error: https://stackoverflow.com/questions/43124775/why-python-3-6-1-throws-attributeerror-module-enum-has-no-attribute-intflag

