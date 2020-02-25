# Rescaling
Rescaling limbs of each speaker towards the mean of all speakers.
file naming: uid-speaker_id-...

# Approach
- Computing mean for each limb from 9 speakers. For each speaker take 1 video from train, test, val -> total 3 videos per speaker.
- obtaining getting mean length for each limb
- use old x,y values of the origin skeletons and save their angles
- start from neck and use old angles with new mean length and draw a new skeleton recursively from the neck
- output: new json files and an animation of new rescaled skeletons
