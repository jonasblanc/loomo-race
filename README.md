# Loomo Person Tracking

<p align="center">
  <img src="https://user-images.githubusercontent.com/44334351/230029037-00378230-6015-4f03-88b2-d1e4c514b8d5.jpg" alt="loomo" width="600"/>
</p>

Source: [Segway](https://ch-de.segway.com)

We implemented a person tracking pipeline based on video. This pipeline was used in a human-robot race with a [Loomo Robot](https://store.segway.com/segway-loomo-mini-transporter-robot-sidekick) following a person around a track based on the camera feed. A specific hand gesture is all that is needed to be selected as the person of interest to be then detected, tracked and followed by the robot. The main challenge consists of guiding the robot through many people without loosing the person of interest.

The complete pipeline and implementation details are explained in the [report](/report.pdf).


## Milestone 1
The [first milestone](/m1) implements a detector, for each frame it detects specific hand gestures as well as people in the frame.

## Milestone 2
The [second milestone](/m2) add memory between frame by tracking a person that made the specific hand gesture in a previous frame (green vs white bounding box)

## Milestone 3 
The [third and last milestone](/m3) adapts the previous milestone to be used as the driving unit for the Loomo robot. The tracking pipeline is running on the camera feed of the robot. The bounding box is then used as a target for the control unit of the robot.

## Notes
The first two milestones have a noteboob version. Please note that most of the delay comes from rendering the frame using javascript, indeed the local python script is smoother.

For all the milestones the specific hand gesture is a triangle with both hands.
