import numpy as np
from donkeycar.utils import normalize_image, throttle as compute_throttle
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType
)

class HailoModelRunner:
    def __init__(self, hef_path):
        self.hef_path = hef_path

        self.device = VDevice()
        self.hef = HEF(hef_path)

        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        network_groups = self.device.configure(self.hef, self.configure_params)

        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

        self.image_height, self.image_width, self.channels = self.input_vstream_info.shape
        self.input_batch = np.empty((1, self.image_height, self.image_width, self.channels), dtype=np.float32)

    def run(self, image):
        image = image.astype(np.uint8)
        
        image_normalized = normalize_image(image).astype(np.float32)
        self.input_batch[0] = image_normalized

        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            input_data = {self.input_vstream_info.name: self.input_batch}
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)

        outputs = infer_results[self.output_vstream_info.name]

        if outputs.shape[1] == 1:
            steering = outputs[0, 0]
            throttle = compute_throttle(steering)
        else:
            steering = outputs[0, 0]
            throttle = outputs[0, 1]
        return steering, throttle
    
    