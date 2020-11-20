# __init__.pxd

cdef extern from "darknet.h":
    """
    /*
     * darknet.h forgot to extern some useful network functions
     */
     static int network_depth(network* net) {
         return net->c;
     }
     static int network_batch_size(network* net) {
         return net->batch;
     }
     static int network_input_size(network* net) {
         return net->layers[0].inputs;
     }
     static int network_output_size(network* net) {
         int i;
         for(i = net->n-1; i > 0; --i) if(net->layers[i].type != COST) break;
         return net->layers[i].outputs;
     }
    """

    # The input vector
    # An RGB or Gray-scale image.
    ctypedef struct image:
        int w;
        int h;
        int c;
        float* data;

    void free_image(image self)

    # The output vector
    ctypedef struct metadata:
        int classes
        char** names

    metadata get_metadata(char* filename)

    # Detection Output
    ctypedef struct box:
        float x;
        float y;
        float w;
        float h;

    ctypedef struct detection:
        box bbox;
        int classes;
        float* prob;
        float* mask;
        float objectness;
        int sort_class;
        float *uc;  # Gaussian_YOLOv3 - tx, ty, tw, th uncertainty
        int points; # bit-0 - center, bit-1 top-left-corner, bit-2 bottom-right-corner

    void free_detections(detection* detections, int len)

    ctypedef struct det_num_pair:
        int num;
        detection* dets;

    void free_batch_detections(det_num_pair* det_num_pairs, int len)


    void do_nms_sort(detection* detections, int len, int num_classes, float thresh)
    void do_nms_obj(detection* detections, int len, int num_classes, float thresh)

    # The model
    ctypedef struct network:
        pass

    network* load_network(char* cfg_filename, char* weights_filename, int clear)
    network* load_network_custom(char* cfg_filename, char* weights_filename, int clear, int batch_size)
    void free_network(network self)

    int network_batch_size(network *self);
    int network_width(network *self);
    int network_height(network *self);
    int network_depth(network *self);
    int network_input_size(network* self);
    int network_output_size(network* self);
    float* network_predict(network, float* input)
    float* network_predict_image(network*, image)

    detection* get_network_boxes(network* self, int width, int height, float thresh, float hier_thresh, int* map, int relative, int* out_len, int letter)
    det_num_pair* network_predict_batch(network* self, image, int batch_size, int width, int height, float thresh, float hier_thresh, int* map, int relative, int letter)


