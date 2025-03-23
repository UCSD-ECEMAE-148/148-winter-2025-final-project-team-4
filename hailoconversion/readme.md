# Convert DonkeyCar Keras .h5 files to Hailo .hef files (Can be used with a Persistent Model)
1) Go to https://hailo.ai/developer-zone/
2) Make an account
3) Download Hailo Dataflow Compiler <br>
https://hailo.ai/?dl_dev=1&file=b36a5d77c29bbff62c86ede1ae193065 <br>
(or if the above link no longer works, download from here:<br>https://hailo.ai/developer-zone/software-downloads/)
4) Start up a Linux terminal
5) Create a conda virtual environment using Python 3.9-3.11
6) cd into hailoconversion
7) Transfer the .whl file into hailoconversion 
8) Run the following in the terminal (or whichever version you have): <br> 
`pip install hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl`
9) Then run the following command, while ensuring you fill in the INPUT_NAME, OUTPUT_NAME and DATA_PATH sections: <br>
`python3 WIP_convert_donkeyh5_to_hef.py --input_h5 INPUT_PATH.h5 --output_hef OUTPUT_NAME.hef --calib_images DATA_PATH/images`
10) If you see the following at the end of execution, then your conversion was fully successful: <br>
`model/output_layer1 → shape=(1, 1, 1, 1), dtype=float32`<br>
`model/output_layer2 → shape=(1, 1, 1, 1), dtype=float32`
11) Place your .hef in your models folder on your DonkeyCar with Raspberry Pi and AI-HAT+
