I built a toy app that allows the user to label time series segments using key presses. It may be used in scenarios such as labeling a signal by second-level segments, or correcting predicted segment labels output by a machine learning model. There are three feature highlights of this toy app:

1. It allows the user to navigate the time series data using the Left/Right arrow keys.
2. It allows the user to label the selected segments using number keys.
3. It has a Undo button that can undo the user’s previous labeling.

These three features speed up the manual labeling process by eliminating the need to switch between the Pan and the Select mode and by adding a quick way to undo an action should a mistake occur. The toy app is self-contained and apart from plotly and dash, you only need to have dash_extensions and numpy installed to run it.

To build on the toy app, for example, one can add a Save functionality to let the user save the annotation. Interested readers are welcome to check out a similar but bigger-scale app I built here: https://github.com/yzhaoinuw/sleep_scoring.
It is an app I built for a research project in neuroscience that studies the waste clearance mechanism in brain during sleep. Specifically, given the synchronized EEG, EMG, and norepinephrine recording of a mouse subject, this app can 1) predict (using a deep learning model integrated into the app) its sleep stage by each second, 2) visualize the data along with the predicted sleep labels, 3) let the user correct the predicted labels or manually label from scratch. In addition, this app includes more features such as letting the user save the results and also deals with more than just one signal.