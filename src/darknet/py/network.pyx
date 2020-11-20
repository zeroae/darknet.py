# network.pyx
from copy import deepcopy

import numpy as np
cimport numpy as np
cimport libdarknet as dn

from libc.stdlib cimport free
from .util import fsspec_cache_open

np.import_array()

cdef convert_detections_to_tuples(dn.detection* detections, int num_dets, float nms_threshold, str nms_type):
    if nms_threshold > 0 and num_dets > 0:
        if nms_type == "obj":
            dn.do_nms_obj(detections, num_dets, detections[0].classes, nms_threshold)
        elif nms_type == "sort":
            dn.do_nms_sort(detections, num_dets, detections[0].classes, nms_threshold)
        else:
            raise ValueError(f"non-maximum-suppression type {nms_type} is not one of {['obj', 'sort']}")
    rv = [
        (j,
         detections[i].prob[j],
         (detections[i].bbox.x, detections[i].bbox.y, detections[i].bbox.w, detections[i].bbox.h)
        )
        for i in range(num_dets)
        for j in range(detections[i].classes)
        if detections[i].prob[j] > 0
    ]
    return sorted(rv, key=lambda x: x[1], reverse=True)


cdef class Metadata:
    classes = []  # typing: List[AnyStr]

    def __cinit__(self, config_file):
        cdnet_mdata = dn.get_metadata(config_file.encode())
        for i in range(cdnet_mdata.classes):
            self.classes.append(cdnet_mdata.names[i].decode("utf-8"))

    def get_classes(self):
        return deepcopy(self.classes)


cdef class Network:
    cdef dn.network* _c_network

    @staticmethod
    def open(config_url, weights_url):
        with fsspec_cache_open(config_url, mode="rt") as config:
            with fsspec_cache_open(weights_url, mode="rb") as weights:
                return Network(config.name, weights.name)

    def __cinit__(self, config_file, weights_file, clear = True, batch_size = 1):
        self._c_network = dn.load_network_custom(config_file.encode(),
                                                 weights_file.encode(),
                                                 1 if clear else 0,
                                                 batch_size)
        if self._c_network is NULL:
            raise RuntimeError("Failed to create the DarkNet Network...")

    def __dealloc__(self):
        if self._c_network is not NULL:
            dn.free_network(self._c_network[0])
            free(self._c_network)

    @property
    def shape(self):
        return dn.network_width(self._c_network), dn.network_height(self._c_network)

    def input_size(self):
        return dn.network_input_size(self._c_network)

    def output_size(self):
        return dn.network_output_size(self._c_network)

    def predict(self, np.ndarray[dtype=np.float32_t, ndim=1, mode="c"] input) -> np.ndarray:
        cdef int input_size
        input_size = self.input_size()
        if input.size != input_size:
            raise TypeError("The input array size does not match the network input size. "
                            f"({input.size} != {input_size})")

        cdef float* output
        output = dn.network_predict(self._c_network[0], <float *>input.data)

        cdef np.npy_intp output_shape[1]
        output_shape[0] = self.output_size()
        return np.PyArray_SimpleNewFromData(1, output_shape, np.NPY_FLOAT32, output)

    def predict_image(self, np.ndarray[dtype=np.float32_t, ndim=3, mode="c"] img) -> np.ndarray:
        cdef dn.image imr
        imr.c = img.shape[0]
        imr.h = img.shape[1]
        imr.w = img.shape[2]
        imr.data = <float *> img.data

        cdef float* output
        output = dn.network_predict_image(self._c_network, imr)

        cdef np.npy_intp output_shape[1]
        output_shape[0] = self.output_size()
        return np.PyArray_SimpleNewFromData(1, output_shape, np.NPY_FLOAT32, output)

    def detect(self,
               frame_size=None,
               float threshold=.5,
               float hierarchical_threshold=.5,
               int relative=0,
               int letterbox=1,
               str nms_type="sort",
               float nms_threshold=.45,
               ):
        pred_width, pred_height =  self.shape if frame_size is None else frame_size

        cdef int num_dets = 0
        cdef dn.detection* detections
        detections = dn.get_network_boxes(self._c_network,
                                          pred_width,
                                          pred_height,
                                          threshold,
                                          hierarchical_threshold,
                                          <int*>0,
                                          relative,
                                          &num_dets,
                                          letterbox)
        rv = convert_detections_to_tuples(detections, num_dets, nms_type, nms_threshold)
        dn.free_detections(detections, num_dets)
        return sorted(rv, key=lambda x: x[1], reverse=True)
