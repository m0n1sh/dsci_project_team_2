# ReadMe - Team 2 DSCI 633

## testbed.ipynb
### Configuring the File
This notebook is used to test various models against each other. To get started with this file, the constants 
must be adjusted to suit your specific environment. By default, the application will search for the dataset 
in the local directory. The dataset can be downloaded from https://drive.google.com/file/d/1fnbYf9cHIiCUBNzAVFAVTIuTaTjxX7pi/view?usp=sharing

The dataset must be extracted prior to use. Similarly, the application will search for the saved h5 version of the AlexNet
model in the local directory. If properly extracted, class detection and file loading should be automated.

The AlexNet model must also be downloaded (If not being created manually first) 
from https://drive.google.com/file/d/1fZpCxQXom_mlQgzHs67EYUF8n_1Ggzl7/view?usp=sharing Like in the case of the 
dataset, the application will expect this file to be in the same directory as the notebook file by default.

### Running the File
With constants properly configured, the application should be able to run without any further interaction other than beginning execution. 
In local testing, this took roughly 14 hours to fully evaluate the 6 models in total. To avoid the need to run, there is a cell (the second to last) that 
can be executed that will load the result dictionary with the results. If running the full application, just skip executing this specific cell. 

For more rapid testing, there is a commented out line in the third from last cell that will shrink the dataset to a small fraction of the size. 
If utilzing this commented out line, the line below must instead be commented out. This utilizes training/testing split, "training" data is
what is used for testing in this case in terms of the variable name in use to allow for uniform behavior regardless of the subset in use.


### Understanding the Output
For each model, the accuracy will be displayed, and for models that were pretrained through Keras, a top 5 accuracy will also be 
displayed. There will be no progress markers in order to prevent flooding the output cells. The final code cell can be ran to 
produce a bar chart displaying the relative accuracies of the tested models. 
