from retrying import retry
from subprocess import CalledProcessError
from sagemaker_inference import model_server

# TODO: from .classifier import handler_service as classifier_service
from .detector import handler_service as detector_service


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


@retry(stop_max_delay=1000 * 50, retry_on_exception=_retry_if_error)
def _start_mms():
    # by default the number of workers per model is 1, but we can configure it through the
    # environment variable below if desired.
    # os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = '2'
    # TODO: Start Classifier *or* Detector Service
    model_server.start_model_server(handler_service=detector_service.__name__)


def main():
    _start_mms()


main()
