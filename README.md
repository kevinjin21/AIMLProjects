# My AI and ML Projects
### A collection of personal projects developed using skills built while working at Apache Corporation and during my UCSD ML Bootcamp.

Apache Corporation is an independent energy company headquartered in Houston, Texas. It is primarily engaged in the exploration, development, and production of natural gas, crude oil, and natural gas liquids in the United States, Egypt, and offshore the UK. My work with Apache centered around power and energy forecasting (focusing on electricity usage, optimization, etc.), and in building AI tools to leverage LLMs in our workflow. 

The UCSD Machine Learning Course is a 6 month program intended to be used as a 'crash course' for learning machine learning. During this course, students are required to create a *capstone project*, which takes real-world data, and formulates a problem to solve or predict using machine learning algorithms and methods.

Aside from projects developed in association with these organizations, I have independently researched and developed a few other "passion projects", fusing topics I am personally interested in with techniques learned throughout these experiences. These 'mini-projects' vary in length and depth, but nevertheless each explore unique aspects of AI, machine learning, and data analysis.

# Table of Contents:

## [💎 Capstone Project: Predicting Sales of Pharmaceutical Drugs in South Korea](https://github.com/kevinjin21/SpringboardProjects/tree/main/Capstone)
<u>Problem:</u> Predict drug sales in Korea using data from various facets of South Korean life, including (but not limited to): weather, healthcare, travel, building construction, etc.
<br>Topics: 
* Data analysis: EDA and feature extraction; recursive feature elimination and correlation studies
* Modeling: Basic regressors, tree-based algorithms, neural networks, ensembling, boosting
* Metric evaluation and analysis; optimization and best-model selection
* Model Deployment: create REST API for the final model, containerize and push to Docker container for deployment

[Access the model and AWS deployment](https://github.com/kevinjin21/SpringboardProjects/tree/main/Capstone/Pharm_Deploy)
<br>[Initial project proposal](https://docs.google.com/document/d/1n_RRZgfwl0WT2p3aCEYIY8RU9nsb2mGosM1jT3U_WT0/edit)
<br>[Full project write-up](https://docs.google.com/document/d/10khUmjzLq3PH_gnmZfJjBF86JT7S8hG7s1BtfL9th5A/edit)

## Projects
### [✍️ YouTube Style Generator](https://github.com/kevinjin21/AIMLProjects/tree/main/Youtube%20Style%20Generator)
<u>Problem:</u> Create original content that mimics the writing style and format of YouTube creators by analyzing their video transcripts and generating new stories in their distinctive style. 
<br>Topics:
* YouTube transcript extraction and text processing using LangChain and YouTube API
* AI-powered style analysis to identify narrative structure, dialogue patterns, and stylistic elements
* Story generation with Claude 3.5 Sonnet (can be changed to LLM of choice) that follows extracted format guides while incorporating user-specified elements
* Template system for quick story generation with different character/setting combinations

### [💸 Finance Invoice Parser](https://github.com/kevinjin21/AIMLProjects/tree/main/Finance%20Invoice%20Parser)
<u>Problem:</u> Automatically process financial statements (banking and credit card) to extract transaction details and categorize spending for better financial management.
<br>Topics:
* PDF data extraction and processing using PyPDF and PyMuPDF libraries; data storage and management via Pandas and SQLite
* AI-powered transaction categorization using LangChain and local LLM (Ollama) integration
* PowerBI visualization for more user-friendly finance dashboard

### [💬 Ghibli Text Generation](https://github.com/kevinjin21/SpringboardProjects/tree/main/Ghibli%20Dialogue%20Generation%20Mini-Project)
<u>Problem:</u> Generate dialogue text in the style of renowned film studio, Studio Ghibli. 
<br>Topics:
* Pytorch 'long short-term memory network' (LSTM) using character chunks to predict future characters
* Data collection and manipulation, namely reshaping and creating tensors for analysis
* Pytorch LSTM construction and training

### [🧞 Disney+ Movie Recommendation](https://github.com/kevinjin21/SpringboardProjects/tree/main/Disney%20Recommendation%20Mini-Project)
<u>Problem:</u> Make recommendations for TV shows/movies from Disney's Disney+ streaming service based on user preferences.
<br>Topics:
* Data exploration and analysis: clean and analyse existing movie/TV data to be used for predictions
* TFIDF Vectorizer and cosine similarity used to make content-based recommendations
* Recommend related movies based on similarities in movie details and potential interests of user

### [🎬 Web Scraping Film Recommendation](https://github.com/kevinjin21/SpringboardProjects/tree/main/Animated%20Film%20Web%20Scraping%20Recommendation%20Mini-Project)
<u>Problem:</u> Scrape anime film data from 2022, 2023 off of Wikipedia webiste and build a simple recommendation system using the data available.
<br>Topics:
* Web scraping and HTML parsing; BeautifulSoup
* Data cleaning and analysis - focus on realistic, incomplete and inconsistent data
* Simple recommendation system comparing content similarities (cosine) 

### [📈 Korea Surgery Time Series Prediction](https://github.com/kevinjin21/SpringboardProjects/tree/main/Health%20Data%20Mini-Project)
<u>Problem:</u> Predict cases of South Korean surgeries using time-series analysis of existing medical data.
<br>Topics:
* EDA of health data statistics; analyze existing trends in time series data
* Use auto ARIMA to train and predict future cases

