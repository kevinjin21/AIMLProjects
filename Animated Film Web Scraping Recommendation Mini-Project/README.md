# Animation Film Web Scraping Recommender
### Build a simple recommendation system using web-scraped data on anime films in 2022 and 2023.

Working with cleaned or well-documented data is the dream of most data scientists. However, this is often far from the truth, and many times information must be taken directly from websites, articles, transcripts, etc. 

This project focuses on using web-scraping techniques to take data from a Wikipedia database, with the goal of using that data to build a simple ML design. Using BeautifulSoup4, we can scrape HTML data from websites and move it into a pandas dataframe for analysis and modeling. 

This project is loosely related to the Disney+ recommendation project, in that it is also building a simple recommender based on content similarities for films. However, the main focus should be on effectively and efficiently obtaining and cleaning data directly from the web. The final recommender is lacking in depth (due to the nature of the data), but thus can represent real-world data in the sense that more data gathering is required for a more complete model.

### Project Overview:
<u>Problem:</u> Scrape anime film data from 2022, 2023 off of Wikipedia webiste and build a simple recommendation system using the data available.
<br>Topics:
* Web scraping and HTML parsing; BeautifulSoup
* Data cleaning and analysis - focus on realistic, incomplete and inconsistent data
* Simple recommendation system comparing content similarities (cosine) 

### Reproducing this project:
The environment.yml file is provided for this project, so it can be easily reproduced on your machine. Create a new conda environment using the provided file and the libraries/dependencies will be ready to use.

For more information on environment creation, please refer to the Capstone README detailing how to recreate an environment from a .yml file: https://github.com/kevinjin21/SpringboardProjects/tree/main/Capstone/Pharm_Deploy (Refer to part 1., Training the model).