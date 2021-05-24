# How to get Tensorflow docker up and running

## Start with a clean ubuntu VM
``` bash
sudo apt upgrade
sudo apt update
sudo apt dist-upgrade
sudo shutdown -r now <or> sudo reboot
```

# Getting the NVidia-Drivers
### You will need this [reference] (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions):
### 1. Verify the system has a CUDA-capable GPU
### 2. Verify the system is running a supported version of Linux
### 3. Verify the system has build tools such as make, gcc installed
### 4. Verify the system has correct Linux kernel headers

### 1. Find the GPU you have on your machine
``` bash
lspci | grep -i nvidia
```
> 0001:00:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)

### 2. Check if you have compatible linux installed (Ubuntu 18.04 is compatible)
#### Refer https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-you-have-supported-version-of-linux
``` bash
uname -m && cat /etc/*release
```

### 3. Install build tools like make, gcc etc.
#### Refer https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/
``` bash
sudo apt update
sudo apt install build-essential
sudo apt-get install manpages-dev
gcc --version
```

### 4. Correct Linux kernel headers
#### Refer https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-kernel-packages
``` bash
uname -r
sudo apt-get install linux-headers-$(uname -r)
```

### Download the driver's .run file for version {450.119.03} 
``` bash
mkdir ~/install-files
cd ~/install-files
BASE_URL=https://us.download.nvidia.com/tesla
DRIVER_VERSION=450.119.03
curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
```
#### Follow on screen prompts (ignore the 32bit installation prompt and hit okay)
``` bash
sudo sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
```
#### check
``` bash
nvidia-smi
```

# Installing CUDA
### Refer https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-nvidia-driver-and-cuda-software
### Select Linux >> x86_64 >> Ubuntu >> 20.04 >> deb (local)
### This installation will be time taking
``` bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

### Check CUDA version
sudo apt install nvidia-cuda-toolkit
nvcc --version

## Install & Configure anaconda

cd ~
mkdir InstallFiles
cd InstallFiles/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

cd ~/.conda
sudo mkdir envs
sudo mkdir pkgs
sudo chown 1000:1000 /home/deepak_sadulla/.conda/envs
sudo chown 1000:1000 /home/deepak_sadulla/.conda/pkgs

## Configure your git and create your ssh keys

Copy you ssh keys into .ssh/ and copy your public key over to github's SSH Key page under Settings

The .ssh directory permissions should be 700 (drwx------).  
The public key (.pub file) should be 644 (-rw-r--r--). 
The private key (id_rsa) on the client host, and the authorized_keys file on the server, should be 600 (-rw-------).
```bash
chmod 700 .ssh
chmod 644 .ssh/id_rsa.pub 
chmod 600 .ssh/id_rsa
chmod 600 .ssh/authorized_keys 
```

### git clone the repo
git clone git@github.com:dsadulla1/yolov4-app.git

### add the yolov4 repo into your current repo
git subtree add --prefix yolo_model https://github.com/Tianxiaomo/pytorch-YOLOv4.git master --squash

### to pull latest changes in the external repo
git subtree pull --prefix yolo_model https://github.com/Tianxiaomo/pytorch-YOLOv4.git master --squash

### Create conda environment
conda create -n Dev python=3.8
conda activate Dev

### get all packages needed
pip install -r requirements.txt
pip install -r app-requirements.txt

### Installing Redis to cache requests
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make
sudo make install

#### Start redis server
redis-server

#### Test redis server is working from another terminal
redis-cli ping