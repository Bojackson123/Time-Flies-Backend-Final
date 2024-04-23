FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y



RUN apt install -y caffe-cpu

RUN apt install -y libgl1-mesa-glx

RUN apt install -y python3-pip

RUN pip3 install opencv-python matplotlib scikit-learn

RUN apt install -y libglib2.0-0

RUN apt install -y xvfb

WORKDIR /Time-Flies-Backend-Main

COPY . /Time-Flies-Backend-Main

RUN pip3 install -r requirements.txt

