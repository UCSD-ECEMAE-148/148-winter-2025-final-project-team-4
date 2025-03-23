from hailo_sdk_client import ClientRunner

model_name = "suarez"
quantized_model_har_path = f"{model_name}_quantized_model.har"
runner = ClientRunner(har=quantized_model_har_path)

hef = runner.compile()
file_name = f"{model_name}.hef"
with open(file_name, "wb") as f:
    f.write(hef)
