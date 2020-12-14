## TODO random notes

-   You need to manually enable SSM once in the console in the correct region.
-   Get instance ID

```
AWS_REGION=us-west-2
INSTANCE_ID=$(aws --region $AWS_REGION ec2 describe-instances \
    --query "Reservations[].Instances[] | [?Tags[?Key=='Name' && Value=='camera-package-notifier-TrainingStack/AmiPreparerAsg']].InstanceId" \
    --output text)
```

-   Connect to SSM shell

```
aws --region $AWS_REGION ssm start-session --target $INSTANCE_ID
```

Copy model to S3

```
DATA_BUCKET_NAME=$(aws cloudformation --region us-west-2 describe-stacks --stack-name camera-package-notifier-TrainingStack \
    --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" --output text)
aws s3 cp data/data_set.pickle.zst s3://"${DATA_BUCKET_NAME}"/
```

Copy code to S3 TODO

```
# on mac
fd . --no-ignore --exclude '__pycache__' src > /tmp/rsync_files.txt
rsync -avz --files-from /tmp/rsync_files.txt . ec2-user@54.202.177.141:~/camera-package-notifier

# on host
/home/ec2-user/anaconda3/envs/tensorflow2_latest_p37/bin/python src/train_model.py /home/ec2-user/data_set.pickle.zst /home/ec2-user/camera_model.h5
```

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

## Infra

### Docker for Deep Learning AMI setup

```
pip install --upgrade pip
pip install awscli
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3.1-cpu-py37-ubuntu18.04-v1.12

cd /Users/asimi/Dropbox/Private/Programming/camera-package-notifier
docker run -v $(pwd):/mnt/source -it 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3.1-cpu-py37-ubuntu18.04-v1.12
```


### How to setup http-long-job-runner

Install Ruby

```
brew install rbenv
rbenv init

# Add the following to zshrc
export RUBY_CONFIGURE_OPTS="--with-openssl-dir=$(brew --prefix openssl@1.1)"
eval "$(rbenv init -)"

# Restart shell, run this to check rbenv is working
curl -fsSL https://github.com/rbenv/rbenv-installer/raw/master/bin/rbenv-doctor | bash

# Install a Ruby
rbenv install 2.7.2
rbenv global 2.7.2

# Restart shell after setting global
```

Install fpm:

```
gem install --no-document fpm
```

References:

-   https://github.com/aws/deep-learning-containers/blob/master/available_images.md
-   https://github.com/rbenv/rbenv
-   https://fpm.readthedocs.io/en/latest/source/pleaserun.html