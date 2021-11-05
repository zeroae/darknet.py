#!/usr/bin/env python

import json
import boto3
from sagemaker.local import LocalSession
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

boto3_session = boto3.Session(region_name="us-east-1")
session = LocalSession(boto3_session)
session.config = {"local": {"local_code": True}}

role = DUMMY_IAM_ROLE
model_dir = "file://ml/model"

model = Model(
    predictor_cls=Predictor,
    image_uri="zeroae/sagemaker-darknet-inference",
    model_data=model_dir,
    role=DUMMY_IAM_ROLE,
    env={
        "SAGEMAKER_MODEL_SERVER_WORKERS": 2
    },
    sagemaker_session=session,
)

predictor = model.deploy(
    name="darknet",
    instance_type="local_gpu",
    initial_instance_count=1,
    serializer=IdentitySerializer("image/jpeg"),
    deserializer=JSONDeserializer(),
)

