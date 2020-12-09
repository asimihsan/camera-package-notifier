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

VIDEOS_PATH=/Users/asimi/Programming/front-door-cam-videos
mkdir -p "${VIDEOS_PATH}"


cd /Users/asimi/Dropbox/Private/Programming/camera-package-notifier

# 1) download videos
python src/ring/get_videos.py "${VIDEOS_PATH}"

# 2) extract key images from videos
python src/extract_images.py "${VIDEOS_PATH}"

# 3) annotate events, e.g. is there package or not
python src/annotate_events.py "${VIDEOS_PATH}"

# 4) generate more augmented images
python src/generate_augmented_images.py "${VIDEOS_PATH}"

# 5) train model
python src/train_model.py "${VIDEOS_PATH}"
```