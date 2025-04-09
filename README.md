# Pizza Price Scraper 
For this group project we scraped the prices of pizza Margheritas in ten major Swiss cities from an online delivery service. Then we analysed and visualised the data. In the following chapters we are going to discuss the steps in detail and show the main findings of the project.

Pizza is a widely affordable dish, with equal popularity for eating in, takeaway or delivery. The simplicity of the ingredients and the ubiquity of the dish make it an ideal candidate for price comparison across different locations.

The chosen online delivery service posed an interesting source for the data scraping since the data was collected from several dynamic pages. 

As guidelines for the project, we set ourselves the following three research questions:
1.	In which big Swiss cities is the price for a pizza Margherita the highest or the lowest respectively?
2.	What factors (e.g. price of pizza Margherita, cuisine, delivery time) influence a restaurant’s rating?
3.	How does the price of a pizza Margherita in big Swiss cities relate to the regional median income (“Pizza Margherita Index”)?

## Module
- main.py
  - This modules is for running the whole project in one go.
  - Main contributors: Omar, Valérie
- scraper.py
  - This modules scrapes product information from a online delivery service. The data is output as a csv file.
  - Main contributor: Omar
- get_api.py
  - This module gets the wage data from the BFS API
  - Main contributor: Valérie
- cleanertransformer.py
  - This modules cleans the data of a provided csv file and does certain transformations.
  - Main contributors: Valérie, Omar
- visualizer.py
  - This modules is for analysing and creation visualizations based on the pizza price data.
  - Visualizations: Pizza-Price to Wage Ratio Map
  - Main contributor: Omar
- analyze.py
  - This modules is for analysing the pizza price data and visualizing the results.
  - Visualizations: Violin Plot (Pizza Prices vs. Cities), Facet Grid Scatterplot (Number of Ratings vs. Rating Score)
  - Main contributor: Valérie
 
