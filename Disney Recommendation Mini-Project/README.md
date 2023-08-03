# Disney+ Recommendation System
### Content-based recommender system for TV shows and movies on the Disney+ streaming service. 

Disney's recent addition to the streaming service environment, Disney+, has been steadily growing with many additions in content across regions. With an enormous library of more than 500 films, 15,000 TV episodes, and 80 Disney+ originals, it's easy to get lost in the plethora of content available for consumation. 

This project aims to analyze bag-of-words text chunks from movie and TV show information, including cast, director, description, etc. Using this information, a TFIDF vectorizer is used to compare cosine similarity between different content entries, thus providing a recommendation closest to the given content entry. This will be performed after initially analyzing the data, noting key aspects of the entries, and cleaning the dataset.

### Project Overview:
<u>Problem:</u> Make recommendations for TV shows/movies from Disney's Disney+ streaming service based on user preferences.
<br>Topics:
* Data exploration and analysis: clean and analyse existing movie/TV data to be used for predictions
* TFIDF Vectorizer and cosine similarity used to make content-based recommendations
* Recommend related movies based on similarities in movie details and potential interests of user
