# Scripts helpful with braidz datasets

## Compress (particularly for Azure)

This script will take our standard data directory structure and create a compressed version intended for long term storage. It removes the 2d data from the .braidz (this is not needed unless you plan to retrack/rekalmanize your data post-hoc, for example, with a different calibration), which saves as much as 6 GB on large datasets. All other raw data and experimental metadata is also saved, and everything is zipped into a `.zip` file to make it easily transportable. The script will *not copy preprocessed_data*, only the raw data and everything in the directory `exp_code` (if it exists).

Note that this script can take a few minutes to run.

### Usage

```
bash compress_for_aws.sh PATH_TO_DATAFILE
bash compress_for_aws.sh --help
```

**Example:**

```bash
bash compress_for_aws.sh /media/username/data/20241217_cool_experiment
```

This will create:
```
/media/caveman/username/data/
├── 20241217_cool_experiment/         # original, untouched
├── 20241217_cool_experiment_no2d/    # stripped copy
└── 20241217_cool_experiment_no2d.zip # zipped version of stripped copy
```
