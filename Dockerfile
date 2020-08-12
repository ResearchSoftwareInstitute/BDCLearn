# Specifying the ubuntu base image
FROM ubuntu:latest

# Name and email of the person who maintains the file
MAINTAINER Matt Satusky satusky@renci.org

# Set working directory as "/"
WORKDIR /

# Updating ubuntu and installing other necessary software
RUN apt-get update --yes \
&& apt-get install python python3-pip git vim --yes 

# Clone BDCLearn repository, pip install requirements from the file,
# 
RUN git clone https://github.com/satusky/BDCLearn \
&& cd BDCLearn \
&& pip3 install -r requirements.txt 

# Set command to bash
CMD ["/bin/bash"]
