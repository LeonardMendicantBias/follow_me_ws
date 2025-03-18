# Base image: Ubuntu 22.04 with CUDA 11.7
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Add ROS 2 Repository and Key
RUN apt update && apt install -y curl gnupg2 lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
    && echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

# Update and install required system dependencies
RUN apt update && apt install -y \
    software-properties-common \
    curl wget locales \
    git python3.10 python3.10-venv python3-pip \
    python3-rosdep python3-colcon-common-extensions \
    ros-humble-desktop \
    && locale-gen en_US.UTF-8 \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Install CuDNN 8.5.0.96
RUN apt update && apt install -y libcudnn8=8.5.0.96-1+cuda11.7 \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.0 with CUDA 11.7
RUN pip install --no-cache-dir torch==2.0 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu117
    
# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set ROS2 environment variables
ENV ROS_DOMAIN_ID=0
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3
ENV ROS_DISTRO=humble

# Source ROS2 on startup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.bashrc

WORKDIR /workspace
# RUN git clone https://github.com/open-mmlab/OpenPCDet.git && \
#     cd OpenPCDet && python3 -m pip install tensorflow && \
#     python3 -m pip --no-cache-dir install spconv-cu117

# CMD ["/bin/bash"]

