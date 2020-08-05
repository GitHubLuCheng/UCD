# Dataset for UCD

Our experiments use two public datasets crawled from Instagram and Vine.
The datasets were respectively introduced and released in [1] and [2]. 
Please reach out to the wonderful [CU Boulder team](https://sites.google.com/site/cucybersafety/home/cyberbullying-detection-project/dataset) for the datasets.

* `instagram.pickle`, 2219 rows including header
```csv
_unit_id,_golden,_unit_state,_trusted_judgments,_last_judgment_at,question1,question1:confidence,question2,question2:confidence,clmn1,clmn10,clmn100,clmn101,clmn102,clmn103,clmn104,clmn105,clmn106,clmn107,clmn108,clmn109,clmn11,clmn110,clmn111,clmn112,clmn113,clmn114,clmn115,clmn116,clmn117,clmn118,clmn119,clmn12,clmn120,clmn121,clmn122,clmn123,clmn124,clmn125,clmn126,clmn127,clmn128,clmn129,clmn13,clmn130,clmn131,clmn132,clmn133,clmn134,clmn135,clmn136,clmn137,clmn138,clmn139,clmn14,clmn140,clmn141,clmn142,clmn143,clmn144,clmn145,clmn146,clmn147,clmn148,clmn149,clmn15,clmn150,clmn151,clmn152,clmn153,clmn154,clmn155,clmn156,clmn157,clmn158,clmn159,clmn16,clmn160,clmn161,clmn162,clmn163,clmn164,clmn165,clmn166,clmn167,clmn168,clmn169,clmn17,clmn170,clmn171,clmn172,clmn173,clmn174,clmn175,clmn176,clmn177,clmn178,clmn179,clmn18,clmn180,clmn181,clmn182,clmn183,clmn184,clmn185,clmn186,clmn187,clmn188,clmn189,clmn19,clmn190,clmn191,clmn192,clmn193,clmn194,clmn195,clmn2,clmn20,clmn21,clmn22,clmn23,clmn24,clmn25,clmn26,clmn27,clmn28,clmn29,clmn3,clmn30,clmn31,clmn32,clmn33,clmn34,clmn35,clmn36,clmn37,clmn38,clmn39,clmn4,clmn40,clmn41,clmn42,clmn43,clmn44,clmn45,clmn46,clmn47,clmn48,clmn49,clmn5,clmn50,clmn51,clmn52,clmn53,clmn54,clmn55,clmn56,clmn57,clmn58,clmn59,clmn6,clmn60,clmn61,clmn62,clmn63,clmn64,clmn65,clmn66,clmn67,clmn68,clmn69,clmn7,clmn70,clmn71,clmn72,clmn73,clmn74,clmn75,clmn76,clmn77,clmn78,clmn79,clmn8,clmn80,clmn81,clmn82,clmn83,clmn84,clmn85,clmn86,clmn87,clmn88,clmn89,clmn9,clmn90,clmn91,clmn92,clmn93,clmn94,clmn95,clmn96,clmn97,clmn98,clmn99,cptn_time,img_url,likes,owner_cmnt,owner_id,shared media,followed_by,follows,id,cyberaggression,cyberbullying
...,...
```

The CU Boulder team provides 3 csv files for the Instagram dataset -- `sessions_0plus_to_10_metadata.csv`, `sessions_10plus_to_40_metadata.csv`, and `sessions_40plus_metadata.csv`.
First concatenate these 3 files into 1 csv file, name it `instagram.csv`, and place it in current folder.

We use [glove.twitter.27B.50d.txt](https://nlp.stanford.edu/projects/glove/) for word representation.
Next, run our data wrangling script [preprocess.py](preprocess.py), an input file `instagram.pickle` for the main program is generated.

The CU Boulder team also provide the user "follows" and "followed" list in files `Instagram_common_users.zip` and `Instagram_normal_users.zip`.
Extract the network statistics and detailed network into the following 2 files, and place them in the current folder.

* `source_target.csv`, 3181 rows including header
```csv
Source,Target
littlemixofficial,vevo
...,...
```

* `user_friend_follower.csv`, 605 rows including header
```csv
user,follows,followed
surfer_magazine,552490,554
...,...
```

### Reference
> \[1\] Homa Hosseinmardi, Sabrina Arredondo Mattson, Rahat Ibn Rafiq, Richard Han, Qin Lv, and Shivakant Mishra. 2015. Analyzing labeled cyberbullying incidents on the instagram social network. In Socinfo. Springer, 49–66.
> 
> \[2\] Rahat Ibn Rafiq, Homa Hosseinmardi, Richard Han, Qin Lv, Shivakant Mishra, and Sabrina Arredondo Mattson. 2015. Careful what you share in six seconds: Detecting cyberbullying instances in Vine. In ASONAM. ACM, 617–622.