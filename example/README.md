# Example raw braidz data analysis

### 1. Preprocess the data

First preprocess the data, run this command from inside the analysis directory: 
  * `python ./preprocess_braidz.py ../20241203_led_demo/ preprocessing_parameters.yaml`
  
The preprocessing script will do some minimal trajectory filtering to throw out very short trajectories, and calculate some useful trajectory features. The statistics of trajectories before and after are saved to a yaml file. 

### 2. Run diagnostic plotting notebook

Run the jupyter notebook titled `braid_diagnostic_plots.ipynb`
