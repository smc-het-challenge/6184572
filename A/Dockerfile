FROM debian:7.4

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget https://repo.continuum.io/archive/Anaconda3-2.5.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda3-2.5.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda3-2.5.0-Linux-x86_64.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV LANG C.UTF-8

ENV PATH /opt/conda/bin:$PATH

RUN conda update conda && \
    conda update -y anaconda && \
    conda update -y --all

RUN pip install "logbook==0.12.5"

ADD matplotlib-1.5.1-no-user.patch /

RUN cd /opt/conda/lib/python3.5/site-packages/matplotlib && \
    patch < /matplotlib-1.5.1-no-user.patch || true

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
