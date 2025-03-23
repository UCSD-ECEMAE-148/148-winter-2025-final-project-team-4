import numpy as np
from donkeycar.utils import normalize_image, throttle as compute_throttle
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
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

        network_name = self.network_group.name
        self.infer_model = self.device.create_infer_model(self.hef_path, network_name)
        self.configured_model = self.infer_model.configure()

        print("Persistent infer model created.")

    def run(self, image):
        image = image.astype(np.uint8)
        image_normalized = normalize_image(image).astype(np.float32)
        dataset = np.expand_dims(image_normalized, axis=0)

        bindings = self.configured_model.create_bindings()

        try:
            self.configured_model.run([bindings], int(100))
        except Exception as e:
            print("Error during persistent run:", e)
            raise

        outputs = bindings.output(self.output_vstream_info.name).get_buffer()
        print("Inference output:", outputs)

        if outputs.ndim == 2 and outputs.shape[1] == 1:
            steering = outputs[0, 0]
            throttle = compute_throttle(steering)
        else:
            steering = outputs[0, 0]
            throttle = outputs[0, 1]
        return steering, throttle

    def shutdown(self):
        self.infer_model.shutdown()
