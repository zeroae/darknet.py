FROM nvidia/cuda:10.2-devel-ubuntu18.04 as mambaforge-cuda

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV CONDA_OVERRIDE_CUDA=10.2

COPY --from=condaforge/mambaforge /opt/conda /opt/conda
COPY --from=condaforge/mambaforge /usr/local/bin/tini /usr/local/bin/tini
COPY --from=condaforge/mambaforge /etc/skel/.bashrc /etc/skel/.bashrc
COPY --from=condaforge/mambaforge /root/.bashrc /root/.bashrc
RUN mamba install pynvml

RUN apt-get update && \
    apt-get install --no-install-recommends --yes wget bzip2 ca-certificates git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


FROM mambaforge-cuda

LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENV TZ="UTC"
ENV DEBIAN_FRONTEND=noninteractive

ENTRYPOINT ["/docker-entrypoint.sh"]
WORKDIR /home/model-server/

ADD https://raw.githubusercontent.com/seatgeek/bash-aptfile/master/bin/aptfile /usr/local/bin/aptfile
RUN chmod +x /usr/local/bin/aptfile

# Ref: https://github.com/seatgeek/bash-aptfile
COPY aptfile /home/model-server
RUN aptfile

# Darknet Library is *either* compiled for CPU or GPU, so we need to conda environments
COPY environment-cpu.yml /home/model-server
COPY environment-gpu.yml /home/model-server
RUN mamba env create -n cpu -f /home/model-server/environment-cpu.yml && \
    mamba env create -n gpu -f /home/model-server/environment-gpu.yml && \
    conda clean --tipsy

COPY entrypoint.sh /docker-entrypoint.sh
CMD ["python", "-m", "darknet.sagemaker"]
