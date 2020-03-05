# Master thesis

Scripts for data preprocessing in master/ma/scripts

- **file_changer:** change names so they can be processed
- **visualization:** visualized 2d json keypoints to 2d pane with opencv
- **save_files:** Save multiple json files from multiple folders in a directory to a single numpy file
  - save_files.py path_to_json_dir path_to_target_dir
  - if no path_to_target_dir, create path_to_json_dir + _saved_numpy dir
- **centralization:** centralizes keypoint files (subtarct all points from the middle (neck))
- **normalization:** normalizes keypoint files (subtract mean and divide by stdev) 
  - centralize_normalize.py path_to_numpy_file path_to_target_dir 
  - path_to_target_dir is optional, if not specified use dir of numpy file
- rescaling: TBD: rescaling all speakers to the same size (started 25.02.2020, not finished - standby)
- data_loader: loading data into a model

