# Machine-Centric Video Streaming Optimization for AI Inference Pipelines
use this to make venv first, the requirements should have everything needed for the
program to work
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# FIRST
run server-h264.py, and wait for the confirmation message that the server is active

# SECOND
run camera-h264.py on a separate terminal, and the video preview should automatically open
to close, press 'q' to end the camera script.

the program in its current state is looking for moving objects and adjusting quality based
on its confidence when identifying that object. if there is no movement, the entire screen
will be black to conserve the most bandwidth

future iterations will adjust masking logic to prioritize humans alongside motion,
optimize and smoothen CRF switching, and separating camera and server devices
