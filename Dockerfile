FROM nvcr.io/nvidia/dgl:23.11-py3

# Setting up working directory
RUN mkdir /src
WORKDIR /src

# Minimize image size
RUN (apt-get autoremove -y; apt-get autoclean -y)

ENV QT_X11_NO_MITSHM=1
CMD ["bash"]