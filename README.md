## Setup

Prerequisites:

-   Install pyenv

Then:

```
pyenv install anaconda3-2020.07
pyenv shell anaconda3-2020.07
conda create -n camera-package-notifier python=3.7
conda activate camera-package-notifier

pip install -r pip-requirements.txt
conda install --yes --file conda-requirements.txt
```

## Running

```
conda activate camera-package-notifier
python src/ring/get_images.py
python src/extract_images.py /tmp/camera-package-notifier
```