# Dataset for UCD

Our experiments use two public datasets crawled from Instagram and Vine.
The datasets were respectively introduced and released in [1] and [2]. 
Please reach out to the wonderful [CU Boulder team](https://sites.google.com/site/cucybersafety/home/cyberbullying-detection-project/dataset) for the datasets.

* `instagram.pickle`, 2218 session

The CU Boulder team provides 3 csv files for the Instagram dataset -- `sessions_0plus_to_10_metadata.csv`, `sessions_10plus_to_40_metadata.csv`, and `sessions_40plus_metadata.csv`.
First concatenate these 3 files into 1 csv file, name it `instagram.csv`, and place it in current folder.

We use [glove.twitter.27B.50d.txt](https://nlp.stanford.edu/projects/glove/) for word representation.
Next, run our data wrangling script [preprocess.py](preprocess.py), an input file `instagram.pickle` for the main program is generated.

The CU Boulder team also provide the "follows" and "followed" list for each user.
Extract the network statistics and detailed network into the following 2 files, and place them in the current folder.

* `source_target.csv`, 3181 rows including header
```csv
Source,Target
littlemixofficial,vevo
littlemixofficial,girlslifemag
...,...
```

* `user_friend_follower.csv`, 605 rows including header
```csv
user,follows,followed
surfer_magazine,552490,554
reneeyoungwwe,257641,499
...,...
```

### Reference
> \[1\] Homa Hosseinmardi, Sabrina Arredondo Mattson, Rahat Ibn Rafiq, Richard Han,Qin Lv, and Shivakant Mishra. 2015. Analyzing labeled cyberbullying incidentson the instagram social network. InSocinfo. Springer, 49–66.
> 
> \[2\] Rahat Ibn Rafiq, Homa Hosseinmardi, Richard Han, Qin Lv, Shivakant Mishra,and Sabrina Arredondo Mattson. 2015. Careful what you share in six seconds:Detecting cyberbullying instances in Vine. InASONAM. ACM, 617–622.