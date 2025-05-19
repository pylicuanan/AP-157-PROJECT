# Coin Face Detector using OpenCV Template Matching 
AP 157 PROJECT\
Philip Jovence Licuanan, David Johann Magpusao, Darwin James Tangonan

This project implements a **coin face detection system** using **OpenCV's template matching** technique. While it is based on the concept of identifying "heads" or "tails" of a coin, we simplified the setup by covering each side with distinct **solid colors**, similar to a poker chip. A single-source LED as a light source and a box contraption were used for uniform lighting and minimized light noise. This allowed for more consistent and robust template matching.

## Overview

- Uses live camera input to continuously scan for coins.
- Automatically captures images every few seconds when enabled.
- Matches against pre-defined "Heads" and "Tails" templates using OpenCV's `matchTemplate`.
- Visual output includes rectangles drawn around detected matches:
  - Green for heads
  - Blue for tails
- Records the number of detected heads and tails in each frame.
- Saves captured frames and detection results to files.
- Logs detection statistics to CSV files for analysis.

## Instructions

- Set up the camera in a fixed place. Ensure uniform lighting across the span of the camera.
- Capture the setup once with sample coins facing "heads" and "tails" included in the picture.
- Crop and save the coin images, each serving as the "heads" and "tails" template.
- The program is now ready for the automatic coin face counter.

## Controls

- `c` – Toggle automatic image capture (every 10 seconds by default)
- `r` – Reset the heads and tails counters
- `q` – Quit the program and save data

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- pandas

## File Structure

- `CoinFaceDetector.py` – Main script handling camera input, capture logic, and user interface.
- `TemplateMatching.py` – Contains the `coin_match` class used for detecting heads and tails in each captured image.
- `Heads.jpg` / `Tails.jpg` – Template images representing the solid-colored heads and tails.
- `captures/` – Folder where captured images are saved.
- `results/` – Folder where images with detection overlays are saved.
- `datasets/` – Folder where CSV logs of detection results are stored.

## Example Use Case

This project can serve as a prototype for basic object detection using template matching. By simplifying real-world objects into solid-color representations, it makes the detection process more reliable for educational demonstrations or controlled-environment projects.
