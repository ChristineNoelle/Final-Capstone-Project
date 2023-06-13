# Final-Capstone-Project
## Capstone Project: Using Musical Features to Classify Songs

**Christine R. Jimenez**

### Executive Summary


**Project Summary and Background**

**Background:** For a number of years, I have observed the effects of recorded music in persons with dementia and Alzheimer’s disease - enhancing remembering and social interaction.  Hence, one of my end targets would be to meld song classification and music recommendation to optimize quality of life for such clinical populations.  For instance, one method to classify songs is via perceived emotion and one usage is to recommend music based on this.

**Music Information Retrieval (MIR) and Song Classification:** Music Information Retrievial (MIR) can be described as extracting information from music.  My goal in this project is to classify music, by positive and negative valence, using audio features.  

* What is the valence of a song?  As a brief description, think of a song that sounds happy.  Valence for that song would likely be classified as positive.  In the models built from this data set, audio features will predict valence.  Songs will be classified as having positive valence, sounding happy or cheerful.  Songs will also be classified as having negative valence, sounding sad, depressing, or angry.

* What is an audio feature?  An audio feature is a characteristic of a song.  One example of an audio feature is the tempo of a song, measured in beats per minute.  Song classification by musical valence, predicted from audio features, might enhance the building of future music recommendation systems.

**Music Recommendation Systems and Clinical Populations:** Music recommendation systems are important to particular clinical populations in persons that consider music highly important. For example, one person with dementia may benefit from the positive feelings and recollections that music evokes.  Another could experience the calming effect of a slow tempo Rhythm and Blues song.  Song classification by valence may enhance existing music recommendation systems and help build these systems for such groups.

**Building Song Classification Models:** There is a Chinese proverb, "A journey of a thousand miles begins with a single step".  Much effort is required to build music recommendation systems for clinical populations.  This project uses audio features to predict the variable termed "valence" and is one small step in enhancing music recommendation systems.


**Findings**

**K-Nearest Neighbors:** The optimal model for classifying valence of songs from musical characteristics was K-Nearest Neighbors. The K-Nearest Neighbors optimal model included three features (Danceability, Energy, Release Year) and two outcome (binary) classes (Valence Upper Positive, Valence Lower Negative). Best parameter was n_neighbors = 21 and time to train was 18 seconds.

* The model had improved from 90.17% (n_neighbors = 5) to 91.13% (n_neighbors = 21). Rise in test accuracy score indicated increased effectiveness of model, with prediction according to majority class of 21 closest data points. GridSearch CV was the tool used find the best k parameter. It was also notable that both the training and test accuracy scores for this model were high, 91.57% and 91.13%, respectively. The final model was both sensitive to training data points and generalizable to test data points.

* Closely following in performance was the unpruned Decision Tree model, with the same three features and outcome (test accuracy score 86.90%). Both three-feature KNN and Decision Tree models had higher test accuracy scores when compared with the three-feature, binary outcome Logistic Regression model (test accuracy score 38.41%). The reason for the success of the K-Nearest Neighbor and Decision Tree models when compared to the Logistic Regression models might be that test data were more clustered than separated linearly.

* Three-feature K-Nearest Neighbor and Decision Tree models not only had higher test accuracy scores than the three-feature Logistic Regression model (38.41%), but also the 13-feature Logistic Regression baseline model (39.16%). 

**Results and Conclusion**: 

* L1 Regularization in Logistic Regression revealed three priority audio features.  Danceability, Energy, and Release Year are displayed in partial dependence plots (https://github.com/ChristineNoelle/Final-Capstone-Project/blob/7da5c533653463fe22aa39fafb8c84052a902721/Partial%20Dependence%20Plot%20.png). These plots visualized the influence of each audio feature value to the average prediction (Valence) in the optimized K-Nearest Neighbors model (n_neighbors = 21). 

* Higher danceability and higher energy ratings contributed to positive valence classification. Lower danceability and lower energy ratings contributed to negative valence classification. 

* Regarding year of song release, it appeared that older songs contributed to positive valence while newer songs contributed to negative valence.

**Future Research and Development:** While how a song is perceived is important, how a person experiences a song is equally valuable information. The experience and perception of emotion may not always match in regards to music.  Memories, for instance, might make listening to a sad-sounding song, a happy experience.  One future direction is to collect data on emotional experience during music listening, in addition to musical valence, and to study the interaction between the two. 

**Next Step and Recommendations:** Proceeding forward, it seems that one can interpret that danceability ratings, energy ratings, and year of song release might be optimal audio features for data collection if one would like to classify songs by valence. The next step is to determine how experience of emotions during song listening relate with the valence of a song. It seems standardized methods need to be devised to collect these audio and emotion features, so that they might be comparable across studies.

### Rationale: Research Task and Question

**Valence:** In this data set, the variable Valence is a measure from 0.0 to 1.0 describing the musical “positiveness” conveyed by a track.  The Kaggle.com website, from which the data was sourced, describes tracks with high valence as sounding more positive (e.g. happy, cheerful, and euphoric).  Tracks with low valence would then sound more negative (e.g. sad, depressed, angry).

**Predicting Valence (Positive or Negative):** Is it possible to predict valence from audio features, and what would be the optimal model to predict the valence of a song? Improved utilization of audio features to determine if a song is cheerful, happy, sad, or depressing, might enhance the building of future music recommendation systems.

**Uses for Predicting Valence:** The optimal model in this analysis could enhance the building of a future, highly personalized, music recommendation system for one in a clinical population. For instance, one person experiencing dementia - that considers music highly important - might benefit from the positive feelings that music can evoke. If audio features can better predict positive valence, for example, these features can be used to recommend music that may benefit quality of life for this person.

### Data Sourcing for Research Question

The following is the link to the data set: https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset/input

**Understanding the Data:** This data set, from website Kaggle.com, consists of 170,653 rows (samples) and 19 columns (features). Features and entailing concepts will be outlined during data exploration. As a brief introduction, the data set title is "Music Recommendation System Using Spotify Dataset". There are 15 features - in addition to Artist, Song Title, and Unique ID - that I would partition into the following categories:

•	"General Features of Music": Danceability, Acousticness, Energy, Key, Liveness, Loudness, Mode, Duration

•	"Lyric-Related Features of Music": Speechiness, Instrumentalness, Explicit

•	"Time-Related Features of Music": Tempo, Release Year, Release Date, Popularity

Valence is also listed as a feature, but for this analysis, I will revise this and make Valence the target variable, or outcome variable. Thus, my goal for in this analysis will be to correctly classify songs by Valence. The remaining features in the provided data set will then be assessed to predict Valence during model-building

### Data Exploration, Cleaning, and Preparation 

•	Null values were removed

•	Duplicate samples were removed

•	Columns were evaluated and renamed (removed if carrying redundant information)

•	Valence was reformatted for Multinomial Logistic Regression (four outcome classes: Upper Positive, Lower Positive, Upper Negative, and Lower Negative) and Logistic Regression (two outcome classes: Upper Positive and Lower Negative)

•	Outliers were removed for particular audio features to ensure sample inclusion more closely reflected typical songs (as well as for model-building)

•	Data was shuffled and split, with a test size of 0.30

### Methodology
•	Baseline models’ test accuracy scores were compared, and selected baseline model was closely balanced binary (two-class outcome) model

•	Remaining 13 dimensions (audio features) were reduced to three dimesions using L1 Regularization tool for Logistic Regression (L1 Regularization is a method that inherently highlights priority features and therefore is a plausible solution for reducing dimensions)

•	Logistic Regression, K-Nearest Neighbors, and Decision Tree models were built with priority three audio features and binary outcome, trained on the training set, and validated with the test set

•	Models were built by comparing test accuracy scores, as top-performing models were optimized with GridSearch CV

### Model Evaluation and Results:

#### Key Improvements in Test Accuracy Scores 

•	Please click to see diagram summary: https://github.com/ChristineNoelle/Final-Capstone-Project/blob/53e4ba6af93542312ead3bd7338279487973837e/Model-Building%20Summary%20(Steps).png

#### Key Comparisons in Model-Building 

•	Please click to see diagram summary: https://github.com/ChristineNoelle/Final-Capstone-Project/blob/53e4ba6af93542312ead3bd7338279487973837e/Main%20Model%20Comparisons.png

### Conclusion
In reality, there are infinite dimensions and boundless methods in classifying songs.  This particular analysis provided one optimal model in song classification.  An optimal K-Nearest Neighbors model was built from comparing test accuracy scores, so that the final model had a test accuracy score of 91.13%.  Using audio features to predict musical valence is akin to one step in enhancing music recommendation systems for clinical populations.  According to Chinese proverb, "A journey of a thousand miles begins with a single step".

Link to Jupyter notebook: 
https://github.com/ChristineNoelle/Final-Capstone-Project/blob/53e4ba6af93542312ead3bd7338279487973837e/Final-Capstone-Project.ipynb

Link to download data directly:
https://drive.google.com/file/d/1eorj0yHVcsShq77JPADnzzftxNrG37Tc/view?usp=sharing

Link to data set in Kaggle Website: 
https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset/input

### Contact and Further Information: 
Christine Jimenez 
Email: christinerapadas@gmail.com
