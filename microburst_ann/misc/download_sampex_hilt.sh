#!/bin/bash

# Download the SAMPEX HILT and attitude data to DATA_DIR/hilt and 
# DATA_DIR/attitude.

DATA_DIR = "/home/mike/research/sampex/data"

# Download the HILT data
wget -r --no-parent --reject "*html*" --wait=3 --directory-prefix=$DATA_DIR http://www.srl.caltech.edu/sampex/DataCenter/DATA/HILThires/

# Move the data.
cd $DATA_DIR
mkdir hilt
cd www.srl.caltech.edu/sampex/DataCenter/DATA/HILThires/
mv * "${DATA_DIR}/hilt"
cd $DATA_DIR
rm -r www.srl.caltech.edu/

# Download the attitude data
wget -r --no-parent --reject "*html*" --wait=3 --directory-prefix=$DATA_DIR http://www.srl.caltech.edu/sampex/DataCenter/DATA/PSSet/Text/

# Move the data.
cd $DATA_DIR
mkdir attitude
cd www.srl.caltech.edu/sampex/DataCenter/DATA/PSSet/Text/
mv * "${DATA_DIR}/attitude"
cd $DATA_DIR
rm -r www.srl.caltech.edu/
