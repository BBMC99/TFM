# ReadMe

## Folder **rttov**:
Includes the scripts necessary to parse the input data, execute RTTOV, collect the outputs and create a CSV dataset file.
The Python scripts can be placed inside the directory RTTOV/wrapper. Each of them has its purpose:
- **pyrttov_tfm.py** --> This is the main script to launch the execution of RTTOV. In the code it is possible to specify the routes to the input files:
  * mhs_data_path: for NC files
  * ecmwf_data_path: for GRIB files
  The variable nprofiles defines the number of profiles to be calculated by RTTOV, which will be saved in the dataset.
- **prepare_data.py** --> Is the first step necessary for pyrttov_tfm.py. This script prepares the data from both sources and parses it in a way that in which RTTOV can manage the input data. 
- **grib2np.py** --> Is used to parse the GRIB files.
- **netCDF4toNumpy.py** --> Is used to parse the NC files.
- **create_dataset.py** --> This script is used to create the output CSV file after the computation of the results by RTTOV.

The script **prepare_env.sh** can be used as help to activate the conda enviroment and update the PYTHONPATH.

## Folder **rrnn**:
Includes the Notebooks that implement Neural Network models trained using the dataset:
- **rrnn_classic_autoParams.ipynb** --> This model can choose hyperparameters automatically using Keras Tuner.
- **rrnn_classic.ipynb** --> This model is created selecting manually the hyperparameters.
- **DeepONet.ipynb** --> This model is created using DeepONet strategy.




