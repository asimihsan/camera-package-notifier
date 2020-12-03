## Setup

Prerequisites:

-   Install pyenv

Then:

```
pyenv install anaconda3-2020.07
pyenv shell anaconda3-2020.07
conda update -n base -c defaults conda
conda create -n camera-package-notifier python=3.7
conda activate camera-package-notifier

pip install --upgrade pip
pip install -r pip-requirements.txt
conda install --yes --file conda-requirements.txt
```

## Running

```
pyenv shell anaconda3-2020.07
conda activate camera-package-notifier

VIDEOS_PATH=/Users/asimi/Downloads/front-door-cam-videos
mkdir "${VIDEOS_PATH}"
python src/ring/get_videos.py "${VIDEOS_PATH}"

pyenv shell anaconda3-2020.07
conda activate camera-package-notifier
VIDEOS_PATH=/Users/asimi/Downloads/front-door-cam-videos
python src/extract_images.py "${VIDEOS_PATH}"
```