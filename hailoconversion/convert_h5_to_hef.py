import os, shutil
import argparse, glob, random
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext

def load_images(paths, h=120, w=160):
    arr = np.zeros((len(paths),h,w,3),dtype=np.float32)
    for i,p in enumerate(paths):
        arr[i] = np.asarray(Image.open(p).convert("RGB").resize((w,h)))
    return arr

parser = argparse.ArgumentParser()
parser.add_argument("--input_h5", required=True)
parser.add_argument("--output_hef", required=True)
parser.add_argument("--calib_images", required=True)
args = parser.parse_args()

print("Loading Keras model:", args.input_h5)
model = tf.keras.models.load_model(args.input_h5)

saved_model_dir = args.input_h5.replace(".h5", "_savedmodel")
model.save(saved_model_dir, include_optimizer=False)

loaded = tf.saved_model.load(saved_model_dir)
infer = loaded.signatures["serving_default"]
end_node_names = [tensor.op.name for tensor in infer.outputs]

runner = ClientRunner(hw_arch="hailo8")

print("Translating TF SavedModel into Hailo runner")
hn_json, _ = runner.translate_tf_model(
    model_path=f"{saved_model_dir}/saved_model.pb",
    end_node_names=end_node_names,
)

hn = json.loads(hn_json)
hn_output_names = [
    layer_name
    for layer_name, layer in hn["layers"].items()
    if layer.get("type") == "output_layer"
]
print("Detected HN outputs:", hn_output_names)

script = "\n".join([
    *(f"quantization_param({name},precision_mode=a16_w16)" for name in hn_output_names),
    "allocator_param(enable_muxer=True)",
    "hef_param(should_use_sequencer=True,params_load_time_compression=True)"
]) + "\n"

with open("model_script.alls","w") as f:
    f.write(script)

paths = sorted(glob.glob(f"{args.calib_images}/*.jpg") + glob.glob(f"{args.calib_images}/*.png"))
random.seed(0)
dataset = load_images(random.sample(paths, min(len(paths), 1024)))
print(f"Calibration dataset loaded ({len(dataset)} images)")

print("Optimizing (quantize to 16-bit)")
runner.load_model_script("model_script.alls")
runner.optimize(dataset)

print("Compiling HEF")
runner.compile()

hef_content = runner.hef

with open(args.output_hef, "wb") as f:
    f.write(hef_content)
print("HEF saved to", args.output_hef)

har_path = args.output_hef.replace(".hef", ".har")
runner.save_har(har_path)
print("HAR saved to", har_path)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

for fname in ("acceleras.log", "hailo_sdk.client.log", "hailo_sdk.core.log", "allocator.log"):
    if os.path.isfile(fname):
        shutil.move(fname, os.path.join(log_dir, fname))
        print(f"Moved {fname} → {log_dir}/{fname}")
    else:
        print(f"Log not found (skipping): {fname}")

with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
    out = runner.infer(ctx, dataset[:1])
    for name, tensor in zip(hn_output_names, out):
        print(f"{name} → shape={tensor.shape}, dtype={tensor.dtype}")
