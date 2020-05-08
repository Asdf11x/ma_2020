# Master thesis

Scripts for data preprocessing in master/ma/scripts

- **file_changer:** change names so they can be processed
- **visualization:** visualized 2d json keypoints to 2d pane with opencv
- **save_files:** Save multiple json files from multiple folders in a directory to a single numpy file
  - [save_files.py](https://github.com/Asdf11x/mt_2020/blob/master/ma/scripts/20-03-04_save_files/save_files.py) path_to_json_dir path_to_target_dir
  - if no path_to_target_dir, create path_to_json_dir + _saved_numpy dir
- **centralization:** centralizes keypoint files (subtarct all points from the middle (neck))
- **normalization:** normalizes keypoint files (subtract mean and divide by stdev) 
  - [centralize_normalize.py](https://github.com/Asdf11x/mt_2020/blob/master/ma/scripts/20-03-02_norm_cent/centralize_normalize.py) path_to_numpy_file path_to_target_dir 
  - path_to_target_dir is optional, if not specified use dir of numpy file
  - [show_json_of_npy_file.py](https://github.com/Asdf11x/mt_2020/blob/master/ma/scripts/20-03-04_save_files/show_json_of_npy_file.py) Show the contains of npy files in a directory as json files
- **rescaling**: TBD: rescaling all speakers to the same size (started 25.02.2020, not finished - standby)
- **data_loader**: Dataloader/text_to_kp - loading data into a model
  - [text_to_kps_dataset.py](https://github.com/Asdf11x/mt_2020/blob/master/ma/scripts/Dataloader/text_to_kp/text_to_kps_dataset.py): Data loaders for text to keypoints using a npy (keypoints) and a csv file (holding text and the links from text to keypoints)
  - [text_to_kps_test.py](https://github.com/Asdf11x/mt_2020/blob/master/ma/scripts/Dataloader/text_to_kp/text_to_kps_test.py): test text_to_kps_dataset.py print the obtained data
  - [gru99_model_own.py](https://github.com/Asdf11x/mt_2020/blob/master/ma/scripts/Dataloader/text_to_kp/gru99_model_own.py): basic seq2seq model using text_to_kps_dataset.py
 - [signs2text](https://github.com/Asdf11x/mt_2020/tree/master/ma/scripts/keypoints2text/kp_to_text_real_data): Current (08.05.20) implementation of basic seq2seq, seq2seq with attention and the beginning of a transformer model. 
 - [Data Utils](https://github.com/Asdf11x/mt_2020/tree/master/ma/scripts/keypoints2text/utils) contains a few scripts for data preprocessing
