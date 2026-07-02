I need a Python script that applies an affine transform to an "unaligned" camera calibration so that 3D data collected with the "unaligned" calibration is rotated, scaled, and translated so that it best fits a user-provided geometry. 


Here is my current workflow:

1. I use ROS to perform a checkerboard calibration for each camera
2. I collect a dataset where I wave an LED wand around the space and record braid tracking data
3. I run the braidz-mcsc multicamselfcal routine to get an unaligned camera calibration. 
4. Collect a new dataset (with the LED wand) with the unaligned calibration loaded. This dataset traces the outside edges and surfaces of my flight arena. Some of the edges and surfaces may not be visible to enough cameras to be tracked.
5. Use a gui (flydra_analysis_calibration_align_gui) to visualize the dataset from 4 in relation to a xml file that defines the arena geometry. I manually tune the rotation, scaling, translation (and flipping) components of an affine transform until the data best aligns with the geometry of the arena. Save the resulting calibration as a stand alone xml file.
6. With the new calibration loaded, collect a dataset where I draw arrows in the positive x, y, z directions. Load that in the alignment GUI and flip axes as needed to make sure that the coordinate frame is oriented as I want it to be relative to the physical arena. 


Files provided for reference: 
an unaligned camera calibration that comes out of braidz-mcsc
an aligned camera calibration from after the GUI alignment
an example arena xml file
the flydra_analysis_calibration_align_gui should be accessible to you on this machine


I will keep steps 1-4 the same. I want your help building a script that can automatically perform step 5. The goal of the script is to find the affine transform that best warps the data to match the arena xml geometry, and then save the new calibration file. To allow me to verify that it is working properly, I want the script to show a simple (not fancy!) visualization of the xml arena, the data overlaid on it, and the location/orientation of each camera relative to the arena and data (very simple rectangle + cone model for each camera). The visualization should be a live window that pops up that I can zoom/rotate the view in using standard libraries (in the past I have used mayavi for such things, but use whatever lightweight modern visualization tool is appropriate for this). I would like the option to tweak the alignment within this visualization (much like flydra_analysis_calibration_align_gui), and save the calibration file. 

I also need help with step 6, which can piggyback off of the work done in step 5. Step 6 should reuse the GUI from step 5, I will load it with a new dataset that has the arrows drawn in it with the LED wand. I will then manually flip x,y,z axes so that the positive arrow goes in the right direction, and save the updated calibration.  

For validation, you should be able to use the example unaligned camera calibration and the arena xml file. 