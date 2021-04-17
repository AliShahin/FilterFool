# This is the PyTorch implementation of "Semantically Adversarial Learnable Filters". 
(The code has been tested on Ubunttu 18.04 with 1 GPU.)



## Setup
    1. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
       ```
        module load python3/anaconda
        conda create --name FilterFool python=3.6.8
       ```

    2. Activate conda environment
       ```
        conda activate FilterFool
       ```

    3. Extract the tar file
       ```   
       tar -zxvf FilterFool_code.tar.gz
       ```

    4. Go to the working directory
       ```
       cd FilterFool_code
       ```

    5. Install requirements (please make sure your GPU is enabled)
       ```
        pip install -r requirements.txt
       ```



## Generate adversarial images of Figure 2 of the manuscript (Dataset: ImageNet, Classifier: ResNet50)
 
    1. In the script.sh set the desired filter among "Nonlinear_detail", "Gamma" or "Log" 

    2. Generate the FilterFool adversarial image
       ```
       bash script.sh
       ```

    3. The FilterFool adversarial image and log file are stored in the Results_{filter} (within the root directory) with the same name as their corresponding original images



## Authors

Ali Shahin Shamsabadi
Changjae Oh
Andrea Cavallaro


