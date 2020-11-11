# microburst_ann
Repository to identify microbursts using convolutional artificial neural networks and bouncing packet microbursts in the future. 

## Installation
Run these shell commands to install the dependencies into a virtual 
environment and configure the SAMPEX data paths:

```
# cd into the top project directory
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 -m microburst_ann init # and answer the promps.
```

Only tested with Python 3.8.