FROM ubuntu:20.04
ARG USER=initial
ARG GROUP=initial
ARG UID=1000
ARG GID=${UID}
ARG SHELL=/bin/bash
ARG workspace=/
#FROM $base_image
RUN echo base image: ${base_image}

#######################################################################
##                            Speeding up                            ##
#######################################################################
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

#######################################################################
##                      install common packages                      ##
#######################################################################
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
   pkg-config \
   apt-utils \
   wget \
   curl \
   git \
   build-essential \ 
   net-tools \
   gedit \
   terminator \
   nautilus \
   software-properties-common \
   apt-transport-https \
#    libopencv-dev \
   ffmpeg \
   x264 \
   libx264-dev \
   zip \
   unzip \
   usbutils \
   sudo \
#    python-pip \
   libusb-1.0-0-dev \
   dbus-x11

#######################################################################
##                    install python environment                     ##
#######################################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
   pip \
   libgl1-mesa-glx \
   libegl1-mesa \
   libxrandr2 \
   libxss1 \
   libxcursor1 \ 
   libxcomposite1 \ 
   libasound2 \
   libxi6 \
   libxtst6 \
   python3-dev \
   python3-pip \
   python3-setuptools \
   qt5-default
   # qt5-default: OpenCVとQtの競合解消のために必須

RUN pip install \
   numpy \
   matplotlib \
   opencv-python

ARG workspace
WORKDIR ${workspace}






#######################################################################
##             install python environment with Anaconda              ##
#######################################################################
# RUN apt-get update && apt-get install -y --no-install-recommends \

# RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh && \
#     bash Anaconda3-2022.05-Linux-x86_64.sh -b && \
#     rm Anaconda3-2022.05-Linux-x86_64.sh

# ENV PATH $PATH:/root/anaconda3/bin

# WORKDIR /root
# ADD requirements.txt /root
# RUN pip install -r requirements.txt