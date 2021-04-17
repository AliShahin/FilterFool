# FilterFool

This is the official repository of [Semantically Adversarial Learnable Filters](https://arxiv.org/pdf/2008.06069.pdf).


<b>Example of results</b>

| Original Image | Adversarial Image |  Original Image | Adversarial Image |  Original Image | Adversarial Image | 
|---|---|---|---|---|---|
| ![Original Image](https://github.com/AliShahin/FilterFool/blob/master/CleanImgs/Nonlinear_Detail/ILSVRC2012_val_00043794.png)| ![Adversarial Image](https://github.com/AliShahin/FilterFool/blob/master/FilteredImages/Nonlinear_Detail/ILSVRC2012_val_00043794.png) |![Original Image](https://github.com/AliShahin/FilterFool/blob/master/FilteredImages/Gamma/ILSVRC2012_val_00014005.png)|![Adversarial Image](https://github.com/AliShahin/FilterFool/blob/master/FilterFoolExamples/ILSVRC2012_val_00014005.png) |![Original Image](https://github.com/AliShahin/FilterFool/blob/master/CleanImgs/Log/ILSVRC2012_val_00011184.png)|![Adversarial Image](https://github.com/AliShahin/FilterFool/blob/master/FilterFoolExamples/ILSVRC2012_val_00011184.png) |
| macaw | Irish setter | crane | mower | Irish terrier | orang |

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
   tar -zxvf https://github.com/AliShahin/FilterFool.git
   ```

4. Go to the working directory
   ```
   cd FilterFool_code
   ```

5. Install requirements (please make sure your GPU is enabled)
   ```
    pip install -r requirements.txt
   ```



## Generate adversarial images 
 
 1. In the script.sh set the desired filter among "Nonlinear_detail", "Gamma" or "Log" 

 2. Generate the FilterFool adversarial image
    ```
    bash script.sh
    ```

 3. The FilterFool adversarial image and log file are stored in the Results_{filter} (within the root directory) with the same name as their corresponding original images



## Authors
* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Changjae Oh](mailto:c.oh@qmul.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)


## References
If you use our code, please cite the following paper:

      @article{shamsabadi2021filterfool,
        title = {Semantically Adversarial Learnable Filters},
        author = {Shamsabadi, Ali Shahin and Oh, Changjae and Cavallaro, Andrea},
        journal={arXiv preprint arXiv:2008.06069},
        year = {2021}
      }
## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).
