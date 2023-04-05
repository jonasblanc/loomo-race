# dlav-group1

We implemented a person tracking pipeline based on video. This pipeline was used in a human-robot race with a [Loomo Robot](https://store.segway.com/segway-loomo-mini-transporter-robot-sidekick) following a person around a track based on the camera feed. A specific hand gesture is all that is needed to be selected as the person of interest to be then detected, tracked and followed by the robot. The main challenge consists of guiding the robot through many people without loosing the person of interest.

* Milestone 1 implements a detector, for each frame it detects specific hand gestures as well as people in the frame.
* Milestone 2 add memory between frame by tracking a person that made the specific hand gesture in a previous frame (green bounding box vs white)
* Milestone 3 adapts milestone 2 to be used as the driving unit for the Loomo robot.

For the first two milestones we used google colab, for easy sharing. Please note that most of the delay comes from rendering the frame using javascript, indeed the local python script is smoother.

For all the milestones the specific hand gesture is a triangle with both hands.
