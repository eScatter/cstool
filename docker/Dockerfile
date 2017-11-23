FROM debian:9

RUN apt-get update && apt-get install --no-install-recommends -y \
	python3 \
	python3-pip \
	python3-dev \
	python3-setuptools \
	python3-wheel \
	build-essential \
	git \
	gfortran

# Get & compile elsepa
WORKDIR /opt

RUN git clone https://github.com/eScatter/elsepa.git && \
	cd /opt/elsepa && \
	make

# Setup python environment
RUN pip3 install --upgrade pip

# Get libraries
RUN git clone http://github.com/eScatter/cslib.git && \
	cd /opt/cslib && \
	pip3 install .
RUN git clone -b no-docker http://github.com/eScatter/pyelsepa.git && \
	cd /opt/pyelsepa && \
	pip3 install .
RUN git clone http://github.com/eScatter/cstool.git && \
	cd /opt/cstool && \
	pip3 install .

# Get endf data
RUN python3 -c 'from cstool.ionization import obtain_endf_files; obtain_endf_files();'

# Mount point for scripts, input and output
RUN mkdir /data
WORKDIR /data
VOLUME /data

ENV LC_CTYPE C.UTF-8
