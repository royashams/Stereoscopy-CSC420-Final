## Stereoscopy-CSC420-Final
Final project for CSC420 introduction to computer vision course at the University of Toronto

## Usage
- All of the code can be run using `main.py`
- Browse for available images in `/image_pairs.`
- The first argument for running main.py is the image name prefix, followed by a number.
- The number can be only `1` or `2`, where `1` indicates running the Anaglyph generation code, and `2` runs the Disparity Map Generation. 
- Example: `python main.py veronica 1` will run Anaglyph code for the `veronica_left` and `veronica_right` image pair.
- Example: `python main.py foot 2` will run Stereo Disparity code for the `foot_left` and `foot_right` image pair. There are more prompts following this initial command, that ask to run the Middlebury evaluation vs. StereoBM computation, and the Gaussian kernel size. 

## Important Notes
- Before running the Middlebury Evaluation, make sure that `imagemagick` is installed. 
- Video and blenderfiles of final results using disparity are in 3D_model_results.

Enjoy! :)
