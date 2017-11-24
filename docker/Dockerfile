FROM debian:9

WORKDIR /tmp

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
RUN git clone https://github.com/eScatter/elsepa.git && \
	cd elsepa && \
	make install && \
	cd .. && \
	rm -r elsepa
ENV ELSEPA_DATA /usr/local/share/elsepa/data

# Setup python environment
RUN pip3 install --upgrade pip

# Get libraries
RUN pip3 install git+https://github.com/eScatter/cslib.git
RUN pip3 install git+https://github.com/eScatter/pyelsepa.git@no-docker
RUN pip3 install git+https://github.com/eScatter/cstool.git

# Get endf data
RUN python3 -c 'from cstool.ionization import obtain_endf_files; obtain_endf_files();'

# Mount point for scripts, input and output
WORKDIR /data
VOLUME /data

ENV LC_CTYPE C.UTF-8
