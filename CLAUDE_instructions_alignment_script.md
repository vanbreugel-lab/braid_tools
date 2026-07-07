I need a Python script that applies an affine transform to an "unaligned" camera calibration so that 3D data collected with the "unaligned" calibration is rotated, scaled, and translated so that it best fits a user-provided geometry. 


Here is my current workflow:

1. I use ROS to perform a checkerboard calibration for each camera
2. I collect a dataset where I wave an LED wand around the space and record braid tracking data
3. I run the braidz-mcsc multicamselfcal routine to get an unaligned camera calibration. 
4. Collect a new dataset (with the LED wand) with the unaligned calibration loaded. This dataset traces the outside edges and surfaces of my flight arena. Some of the edges and surfaces may not be visible to enough cameras to be tracked, but most should be. I also "draw" a set of three perpendicular arrows in the middle of the space that I will use to specify the positive x,y,z directions. 
5. Use a gui (flydra_analysis_calibration_align_gui) to visualize the dataset from 4 in relation to a xml file that defines the arena geometry. I manually tune the rotation, scaling, translation (and flipping) components of an affine transform until the data best aligns with the geometry of the arena. Save the resulting calibration as a stand alone xml file.


Files provided for reference: 
In this directory:
'/home/lobula/ros2_ws/src/braid_tools/braid_calibration_mcsc_alignment/bigtunnel_calibration'
There are some files:
checkerboard_calibration: checkerboard calibration data for all the cameras (used by mcsc)
20251120_calibration: the unaligned directory has braidz data and a calibration xml produced by the mcsc script. The aligned directory has a dataset corresponding to step 4 above. 
bigwindtunnel_stim.xml: xml file describing the geometry of the rectangle corresponding to step 4 and 5 above. 

I will keep steps 1-4 the same. I want your help building a script that can automatically perform step 5. The goal of the script is to find the affine transform that best warps the data to match the arena xml geometry, and then save the new calibration file. To allow me to verify that it is working properly, I want the script to show a simple (not fancy!) visualization of the xml arena, the data overlaid on it, and the location/orientation of each camera relative to the arena and data (very simple rectangle + cone model for each camera). The visualization should be a live window that pops up that I can zoom/rotate the view in using standard libraries (in the past I have used mayavi for such things, but use whatever lightweight modern visualization tool is appropriate for this). I would like the option to tweak the alignment within this visualization (much like flydra_analysis_calibration_align_gui), and save the calibration file. 

The gui should also have options for manual fine tuning of the scale, rotation, and translation, and -- critically -- an option to flip the positive direction of the axes. The gui needs to show positive x,y,z axes so I can align the arrows. 

Note that you will need to be careful in using the data because of the arrows on the inside, don't let that throw off the alignment of the planes of the rectangle. 
