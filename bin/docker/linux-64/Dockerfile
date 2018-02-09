ARG CONDA_VERSION=2
ARG GIT_TAG=latest
FROM statiskit/linux-64:${GIT_TAG}-py${CONDA_VERSION}k

COPY README /home/main/README
RUN rm /home/main/.condarc
COPY .condarc /home/main/.condarc
RUN echo "source activate fpd18" >> .bashrc
RUN echo "cat /home/main/README" >> .bashrc

RUN conda create -n fpd18 fpd18
RUN git clone --recursive https://github.com/StatisKit/FPD18
WORKDIR /home/main/FPD18/share/jupyter