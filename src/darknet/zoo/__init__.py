import intake
import os

here = os.path.abspath(os.path.dirname(__file__))
zoo = intake.open_catalog(os.path.join(here, "darknet.yaml"))
imagenet = intake.open_catalog(os.path.join(here, "imagenet.yaml"))

del here
