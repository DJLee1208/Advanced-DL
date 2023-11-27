FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER author@example.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
