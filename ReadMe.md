# ReadMe - Team 2 DSCI 633

## General Workflow 
The logical workflow of running the full project begins with final_submission_base_model.ipynb as this notebook is used to create and train our implementation of AlexNet. Next in the logical flow is testbed.ipynb, this notebook orchestrates the testing of our AlexNet implemeentation in comparison to 5 pre-trained models. After this, Transfer_Learning.ipynb, Transfer_learning_VGG16.ipynb, and Alexnet_Model.ipynb can be ran in any order as these notebooks work to perform transfer learning on the selected pre-trained models and our AlexNet implementation.

The sections that follow describe the specific operation guidance for these notebooks. 

## Final_submission_base_model.ipynb


### Configuring the File
This notebook is used to build the base AlexNet model and observe the performace of it. Basic variable are editable as per our need and the class will reflect on initilization. Dataset is available in this link below, it has to be extracted and the training/validation path variable need to be updated to run the model. https://drive.google.com/file/d/1ArknCm4w1dpf7-G1_5TEPiPcb44AIxcJ/view?usp=sharing

### Running the File
It's a simple single cell run, as the code the modularised in that way. Each function is clearly explained and in the main class.

### Understanding the Output
* No of Images in Test and Validation dataset.
* Outputs from this file in the order on appreance. Model sumary which explaing the different layers of the models, kernals, and paramaters used.
* Metrics of Each epoch.
* Graph of the accuracy and loss performance of the model vs epoch.
* Random set of images and their predicteion embedded in the image.

## Testbed.ipynb

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

## Transfer_Learning.ipynb
### Configuring the File
This notebook is used to test feasibility of transfer learning of ResNet50 and InceptionV3 models against each other. To get started with this file, the constants 
must be adjusted to suit your specific environment. By default, the application will search for the dataset in the local directory, you will not be needed to mount your drive. The dataset can be downloaded from https://drive.google.com/file/d/1NJP2e9HpdlPSPwN8NAM5swCeBYlQZyS2/view?usp=share_link

You will be needed to download the data from the above link no need to extract the data from the zip folder, you will be needed to copy the path of the zip folder and update it in the ZipFilePath variable and update the DownloadedPath variable with the path of where you would like to download your extracted zip folder for example:
ZipFilePath = "path to the zip folder", DownloadedPath = "path to where you like to extract the zip folder".

Additionally, update the train_dir and valid_dir variable to the path of the train folder  and test folder in the extracted zip folder respectively.

### Running the File
With constants properly configured, the application should be able to run without any further interaction other than beginning execution. 
In local testing, this took roughly 1 hours to fully evaluate the ResNet50 and InceptionV3 model.

### Understanding the Output
For each model, the accuracy will be displayed and a accuracy and loss plot will be plotted.
