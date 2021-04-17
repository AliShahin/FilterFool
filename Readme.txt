# FilterFool

This is the official repository of [Semantically Adversarial Learnable Filters](https://arxiv.org/pdf/2008.06069.pdf).


<b>Example of results</b>

| Original Image | Adversarial Image | Original Image | Adversarial Image |
|---|---|---|---|
| ![Original Image](https://github.com/AliShahin/FilterFool/tree/master/FilterFoolExamples/ILSVRC2012_val_00000328.png) | ![Adversarial Image](https://github.com/AliShahin/FilterFool/tree/master/FilterFoolExamples/ILSVRC2012_val_00043794.png) |![Original Image](https://github.com/AliShahin/FilterFool/tree/master/FilterFoolExamples/ILSVRC2012_val_00030569.png) | ![Adversarial Image](https://github.com/AliShahin/FilterFool/tree/master/FilterFoolExamples/ILSVRC2012_val_00014005.png) |
| ![Original Image](https://github.com/AliShahin/FilterFool/tree/master/Dataset/ILSVRC2012_val_00002437.png) | ![Adversarial Image](https://github.com/AliShahin/FilterFool/tree/master/FilterFoolExamples/ILSVRC2012_val_00011184.png) |


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


