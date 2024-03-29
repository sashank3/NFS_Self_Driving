# Self-Driving Car in Need for Speed : Most Wanted 2012 Game

## Summary:

In my project, I've created a functional computer-vision model for autonomous driving, utilizing a realistic video game as a simulation platform. The model incorporates fundamental object detection and lane-finding algorithms similar to those employed in Full Self-Driving Tesla vehicles. Combining these algorithms with a sophisticated driving algorithm, my system is capable of maneuvering autonomously, successfully navigating around obstacles, obeying traffic signs and signals, and ensuring a safe journey from point A to B.

#### Dependencies : 

```
Numpy                    1.14.5
Pyautogui                0.9.42
OpenCV                   4.1.2
Tensorflow               1.10.0
```

+ Run the game on 800 x 600 resolution ( Changing this requires you to reshape the Region Of Interest )
+ Run the COMBINED.py File on any Python IDE once in-game.
+ The Car will drive itself and stay within the road limits while detecting and avoiding incoming traffic.


#### Game View 

![Image 1](View.png)


#### Edge Detection Algorithm to acquire necessary data

![Image 2](Edges.png)


#### Lane Finder Algorithm to help ensure the car stays within bounds

![Image 3](LaneDetect.png)

Thank you!
