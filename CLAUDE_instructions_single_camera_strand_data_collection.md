I now need a python script that does the following. 

I have a full braid camera array running on a separate computer. The 3D tracking from that system is available from a given URL (the default URL in the braid_ros_listener.py node). We can either grab it directly, or plan to use the braid_ros_listener.py node to get the data into ROS2. 

I have an auxiliary camera attached to the computer where we are working now. I want a simple script to launch a strand-camera instance that will perform 2D tracking with this auxiliary camera. This camera is synchronized with the braid array -- they are all attached to the same trigger box. 

I am going to collet a dataset where I move an LED around the space seen by both the braid array and the auxiliary strand-cam. 

I want the script to save a dataset that records both the braid 3D positions (which again, are available from the braid server, or ROS), and the 2D pixel locations on the auxiliary camera. 

In a subsequent script, we will use this dataset to calibrate the auxiliary camera so that 3D positions can be mapped into the 2D field of view of the auxiliary camera. 