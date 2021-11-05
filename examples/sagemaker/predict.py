#!/usr/bin/env python

import boto3
import json
from sagemaker.local import LocalSession
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

boto3_session = boto3.Session(region_name="us-east-1")
session = LocalSession(boto3_session)
session.config = {"local": {"local_code": True}}

predictor = Predictor(
    sagemaker_session=session,
    endpoint_name="darknet",
    serializer=IdentitySerializer("image/jpeg"),
    deserializer=JSONDeserializer(),
)

with open("dog.jpg", "rb") as f:
    predictions = predictor.predict(
        f.read()
    )
print(json.dumps(predictions, indent=2))
