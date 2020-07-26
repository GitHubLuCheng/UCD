# UCD - Unsupervised Cyberbullying Detection 


Implementation of Unsupervised Cyberbullying Detection via Time-Informed Gaussian Mixture Model (CIKM20)


## Code usage
We provide three quickstart bash scripts:
1. [run_all_wrangling.sh](/wrangling/run_all_wrangling.sh)
2. [run_all_measures.sh](/measures/run_all_measures.sh)
3. [run_all_models.sh](/models/run_all_models.sh)

Please put all the files in the same folder, and run UCD.py. The output is the averaged Precision, Recall, F1, AUC, and the corresponding standard deviations over 10 replications for the Instagram dataset.

Download and place data in the [data](/data) directory, then uncompress them.
First run `run_all_wrangling.sh` to create formatted data, then run `run_all_temporal_analysis.sh` to conduct the temporal analysis or `run_all_predictors.sh` to reproduce the results of prediction tasks.
Detailed usage and running time are documented in the corresponding python scripts.

Note the datasets are large, so the quickstart scripts will take up to 24 hours to finish.
Check the estimated running time in each python script before you run the quickstart scripts.

## Data
The data is hosted on [Google Drive](https://drive.google.com/drive/folders/19R3_2hRMVqlMGELZm47ruk8D9kqJvAmL?usp=sharing) and [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TORICY).
See more details in this [data description](/data/README.md).


## Python packages version
Tensorflow==1.12.0
Keras==2.2.4
networkx==2.2
numpy==1.16.5
pandas==0.24.2


### Reference
> [Lu Cheng](http://www.public.asu.edu/~lcheng35/), [Kai Shu](http://www.cs.iit.edu/~kshu/), [Siqi Wu](https://avalanchesiqi.github.io/), [Yasin N. Silva](http://www.public.asu.edu/~ynsilva/), [Deborah L. Hall](http://www.dhallpsych.com/index), and [Huan Liu](http://www.public.asu.edu/~huanliu/). Unsupervised Cyberbullying Detection via Time-Informed Gaussian Mixture Model. *ACM Conference on Information and Knowledge Management (CIKM)*, 2020. \[[paper](https://avalanchesiqi.github.io/files/cscw2019network.pdf)\]
