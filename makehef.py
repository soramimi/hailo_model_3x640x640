import numpy as np
import os

from hailo_sdk_client import ClientRunner

#Define model information
model_name = 'model_3x640x640'
onnx_path = 'model_3x640x640.onnx'
start_node = 'input'
end_node = ['output']
input_shape = {'input': [1, 3, 640, 640]}
chosen_hw_arch = 'hailo8l'
input_height = 640
input_width = 640
input_ch = 3

alls_lines = [
    'model_optimization_flavor(optimization_level=0, compression_level=1)\n',
    'resources_param(max_control_utilization=1.0, max_compute_utilization=1.0,max_memory_utilization=1.0)\n',
    'performance_param(fps=250)\n'
]

#Parsing
runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(onnx_path, model_name, start_node_names=[start_node], end_node_names=end_node, net_input_shapes=input_shape)

parsed_model_har_path = f'{model_name}_parsed_model.har'
runner.save_har(parsed_model_har_path)

#Optimize
calibData = np.random.randint(0, 255, (1024, input_height, input_width, input_ch))

runner.load_model_script(''.join(alls_lines))
runner.optimize(calibData)

quantized_model_har_path = f'{model_name}_quantized_model.har'
runner.save_har(quantized_model_har_path)

#Compile
hef = runner.compile()

file_name = f'{model_name}.hef'
with open(file_name, 'wb') as f:
    f.write(hef)

compiled_model_har_path = f'{model_name}_compiled_model.har'
runner.save_har(compiled_model_har_path)
