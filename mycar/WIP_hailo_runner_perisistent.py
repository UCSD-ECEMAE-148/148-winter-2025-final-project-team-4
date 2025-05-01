from donkeycar.utils import throttle as compute_throttle

from hailo_platform import VDevice, HEF, ConfigureParams, HailoStreamInterface
from hailo_platform import InputVStreamParams, OutputVStreamParams, FormatType


class HailoRunner:
    def __init__(self, hef_path, interface=HailoStreamInterface.PCIe):
        self.device = VDevice()
        self.hef = HEF(hef_path)
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=interface)
        
        network_groups = self.device.configure(self.hef, self.configure_params)
        self.network_group = network_groups[0]
        
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
        self.vstreams = self.network_group.activate(self.input_vstreams_params, self.output_vstreams_params)
        
        self.input_vstream = self.vstreams.inputs[0]
        self.output_vstream = self.vstreams.outputs[0]

    def run(self, input_data):
        self.input_vstream.write(input_data)
        output_data = self.output_vstream.read()

        print("Inference output:", output_data)

        if output_data.ndim == 2 and output_data.shape[1] == 1:
            steering = float(output_data[0, 0])
            throttle = compute_throttle(steering)
        else:
            steering = float(output_data[0, 0])
            throttle = float(output_data[0, 1])

        return steering, throttle

    def __del__(self):
        try:
            if hasattr(self, 'vstreams'):
                self.vstreams.close()
            if hasattr(self, 'network_group'):
                self.network_group.release()
            if hasattr(self, 'device'):
                self.device.release()
        except Exception as e:
            print(f"[HailoRunner] Warning during cleanup: {e}")

