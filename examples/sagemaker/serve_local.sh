#!/usr/bin/env bash

image=${1:-zeroae/sagemaker-darknet-inference }

docker run -v $(pwd)/ml:/opt/ml -p 8080:8080 --rm ${image} serve
