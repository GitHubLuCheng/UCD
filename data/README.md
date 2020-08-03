# Dataset for UCD

Our experiments use two public datasets crawled from Instagram and Vine.
The datasets were respectively introduced and released in [1] and [2]. 
Please reach out to the wonderful [CU Boulder team](https://sites.google.com/site/cucybersafety/home/cyberbullying-detection-project/dataset) for the datasets.

We provide our data wrangling script [preprocess.py](preprocess.py).
After running it, the input files for the main program are generated in current folder.
* newdata.pickle
* user_friend_follower.csv
* source_target.csv

### Reference
> \[1\] Homa Hosseinmardi, Sabrina Arredondo Mattson, Rahat Ibn Rafiq, Richard Han,Qin Lv, and Shivakant Mishra. 2015. Analyzing labeled cyberbullying incidentson the instagram social network. InSocinfo. Springer, 49–66.
> 
> \[2\] Rahat Ibn Rafiq, Homa Hosseinmardi, Richard Han, Qin Lv, Shivakant Mishra,and Sabrina Arredondo Mattson. 2015. Careful what you share in six seconds:Detecting cyberbullying instances in Vine. InASONAM. ACM, 617–622.