# Navigating the Unknown
In this project a perception stack for the DJI Tello EDU quadcopter is designed to enabe it to autonomously
navigate through irregular, unknown-shaped windows. The primary goal is to identify and fly through the largest gap in a wall. The process begins by maneuvering the quadcopter to a position where the full gap is visible, followed by detecting the optical flow of the window. Subsequently, this flow data is postprocessed
to outline the largest gap’s contour and pinpoint its center. With the center identified we employ visual servoing to guide the quadcopter to align its image center with the gap’s center, facilitating a successful flight through the gap.

(Check the full problem statements here [project 4](https://rbe549.github.io/rbe595/fall2023/proj/p4/))


## Steps to run the code
- Install Numpy, OpenCV, djitellopy, torch, cudatoolkit, matplotlib libraries before running the code.
- Install all the library dependencies mentioned [here](https://github.com/princeton-vl/RAFT)
- Turn the drone on and connect to it.
- To run the main code run the `main.py` file after installing all dependancies. This will save the final output in `repository` folder itself.
- In repository folder:
  ```bash
  python3 main.py --model=RAFT/models/raft-sintel.pth
  ```
- In our testing we found the weights for sintel dataset are giving better results. Try other weights if you want to by changing the weight file accordingly.

## Report
For detailed description see the report [here](Report.pdf).

## Plots and Animations
### Blender simulation
Testing the flow detection in blender simulation:

Case 1:
<p float="middle">
	<img src="media/test1.gif" width="250" height="250" title="frames"/> 
	<img src="media/processed_flow/test1h_frame000.png" width="250" height="250" title="flow"/>
	<img src="media/processed_frame/test1h_frame000.png" width="250" height="250" title="real frame"/>
</p>

Case 2:
<p float="middle">
	<img src="media/test2.gif" width="250" height="250" title="frames"/> 
	<img src="media/processed_flow/test2h_frame000.png" width="250" height="250" title="flow"/>
	<img src="media/processed_frame/test2h_frame000.png" width="250" height="250" title="real frame"/>
</p>

### Real world run
Gaussian splat of the real window in the lab:
<p float="middle">
	<img src="media/gausssplat_gif.gif" width="375" height="200" />
</p>

Live demo runs:

Watch the good quality video of demo run 1 on the real tello drone here ([link1](https://youtu.be/wt_jdC7YsPk) and [link2](https://youtu.be/kCroe-EPg3U)).

<p float="middle">
	<img src="media/output3.gif" width="350" height="350" title="Drone POV"/> 
	<img src="media/output3dronepov.gif" width="350" height="350" title="Cameraman POV"/>
</p>

Watch the good quality video of demo run 2 on the real tello drone here ([link1](https://youtu.be/JmVFpYKcHsI) and [link2](https://youtu.be/AGB9YBB8y6k)).

<p float="middle">
	<img src="media/output4.gif" width="350" height="350" title="Drone POV"/> 
	<img src="media/output4dronepov.gif" width="350" height="350" title="Cameraman POV"/>
</p>


## References
1. [https://github.com/princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)

## Collaborators
Ankit Talele - amtalele@wpi.edu

Chaitanya Sriram Gaddipati - cgaddipati@wpi.edu

Shiva Surya Lolla - slolla@wpi.edu




  
