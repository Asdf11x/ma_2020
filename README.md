# mt_2020

So far:

- file_changer: change names so they can be processed
- visualization: visualized 2d json keypoints to 2d pane with opencv
- normalization: normalizes keypoint files (subtract mean and divide by stdev) 
- centralization: centralizes keypoint files (subtarct all points from the middle (neck))
- rescaling: TBD: rescaling all speakers to the same size (started 25.02.2020, not finished - standby)
- data_loader: loading data into a model
- save_files: Save multiple json files from multiple folders in a directory to a single numpy file
