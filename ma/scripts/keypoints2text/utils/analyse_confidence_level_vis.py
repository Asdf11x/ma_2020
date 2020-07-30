"""
analyse_confidence_level_vis.py: visualizes results of analyse_confidence_level script

results look like:

{'pose_keypoints_2d': 0.3986607017707151, 'face_keypoints_2d': 0.8399530071912715, 'hand_left_keypoints_2d': 0.49033687098093925, 'hand_right_keypoints_2d': 0.4357413572924184}

Goal: Comparision of Phoenix and How2Sign dataset

"""

import pandas as pd
import matplotlib.pyplot as plt

# add confidence means by hand from other source

data = {
    'Segment': ['val_pose', 'val_face','val_hand_left','val_hand_right'],
    'How2Sign': [0.39363903234030806, 0.8378197056001734, 0.46477502350261896, 0.4246017309293574],
    'PHOENIX': [0.31319216500405533, 0.7697564108682671, 0.3104184979114336, 0.2979657303039666]
}

df = pd.DataFrame (data, columns = ['Segment','How2Sign', 'PHOENIX'])

print(df)

ordered_df = df.sort_values(by='Segment')
my_range=range(1,len(df.index)+1)

plt.hlines(y=my_range, xmin=ordered_df['How2Sign'], xmax=ordered_df['PHOENIX'], color='grey', alpha=0.4)
plt.scatter(ordered_df['How2Sign'], my_range, color='navy', alpha=1, label='How2Sign')
plt.scatter(ordered_df['PHOENIX'], my_range, color='gold', alpha=0.8 , label='PHOENIX')
plt.legend()

# Add title and axis names
plt.yticks(my_range, ordered_df['Segment'])
plt.title('OpenPose Confidence Value Comparision\n of PHOENIX14T and How2Sign', loc='left')
plt.xlabel('Confidence Value')
plt.ylabel('Segments')
plt.xlim(0,1)
plt.show()