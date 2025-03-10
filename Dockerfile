FROM pytorch/pytorch

RUN apt-get update -y && apt-get install -y git nano python3-pip binutils libproj-dev gdal-bin ffmpeg libgdal-dev libboost-dev build-essential cmake
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.10 1
ADD ./requirements.txt /home/bmahsa/requirements.txt
WORKDIR /home/bmahsa
RUN pip3.10 install cython
RUN mkdir -p /home/bmahsa/.cython/inline
RUN pip3.10 install -r requirements.txt

WORKDIR /home/bmahsa
RUN git clone https://github.com/funkelab/funlib.learn.torch.git
WORKDIR /home/bmahsa/funlib.learn.torch
RUN pip3.10 install . --target=/usr/local/lib/python3.10/dist-packages

WORKDIR /home/bmahsa

RUN chmod 777 /usr/local/lib/python3.10/dist-packages/waterz
RUN chmod 777 /home/bmahsa/.cython/inline