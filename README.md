# Indoor-scene-vanish-point-detection-and-line-labeling
We estimate three mutually orthogonal vanishing directions points in the following steps:
- Detect straight line segments
- Find intersection points (vanishing point candidates)
- Score and rank vanishing point candidates
- Choose the triplet with the highest combined score that also leads to reasonable camera parameters
- Label the line segments based on the estimate

## Prerequisites

- cv2


## Usage 
```
$python main.py "filename"
```
Input files should be stored in "/input"

Output files will be stored in "/output"

## Example

Input: 
![alt text](https://github.com/hyc96/Indoor-scene-vanish-point-detection-and-line-labeling/blob/master/input/1.jpg)
Output:
![alt text](https://github.com/hyc96/Indoor-scene-vanish-point-detection-and-line-labeling/blob/master/output/membership_1.jpg)

Vanishing lines are labeled with three directions, the thrid label indicates irrelavent lines.
## License

see the [LICENSE.md](LICENSE.md) file for details

## References 
This project is generally implemented based on:
1. Mallya, A. & Lazebnik, S. (2015). Learning Informative Edge Maps for Indoor Scene Layout Prediction
2. Schwing, A. G. & Urtasun, R. (2012). Efficient Exact Inference for 3D Indoor Scene Understanding
3. Hedau, V., Hoiem, D. & Forsyth, D. A. (2009). Recovering the spatial layout of cluttered rooms
4. Rother, C. (2000). A New Approach for Vanishing Point Detection in Architectural Environments
5. Tardif, J.-P. (2009). Non-iterative approach for fast and accurate vanishing point detection
6. Denis, P., Elder, J. H. & Estrada, F. J. (2008). Efficient Edge-Based Methods for Estimating Manhattan Frames in Urban Imagery

Contact me for more detailed report on the implementation.

## Contact
huaiyuc@seas.upenn.edu
