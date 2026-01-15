
### **Experiment Setup and Execution Guide**

#### **Folder Structure**
1. **DataSets**: All datasets required for the experiment are stored in this folder.
2. **Codes**: This folder contains all the code files necessary for the experiment.
3. **NotebookFiles**: This folder contains 8 IPython notebook files (`.ipynb`), each corresponding to one dataset experiment.

#### **Steps to Run the Experiment**

1. **Download the Repository**
   - Clone or download the entire GitHub repository to your local machine.

2. **Open Jupyter Notebook**
   - Launch Jupyter Notebook on your local system.
   - Navigate to the `NotebookFiles` folder and open the relevant `.ipynb` file for the dataset you want to work with.

3. **Install Necessary Packages**
   - Install the necessary packages:

     pip install category_encoders
     pip install tensorflow
     pip install researchpy
     pip install ucimlrepo


   - If the following error occurs when downloading UCI dataset:

     SSLError: [X509: NO_CERTIFICATE_OR_CRL_FOUND] no certificate or crl found (_ssl.c:4353)

     run the following command to fix it:

     pip install --upgrade --force-reinstall certify


4. **Verify Adjustments**
   - Double-check that all file paths in the notebook are correctly mapped to your local setup.

5. **Run the Notebook**
   - Once the paths are updated, execute the notebook step by step or run all cells to perform the experiment.

6. **Repeat for Other Experiments**
   - For experiments related to other datasets, repeat the above steps with the corresponding `.ipynb` file.


This version corrects grammatical errors, improves clarity, and ensures consistency in formatting.