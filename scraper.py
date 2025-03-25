import requests
from bs4 import BeautifulSoup
import pandas as pd
from seleniumbase import Driver
from selenium import webdriver
import time
import os

class restaurant:
    def __init__(self, restaurantName, restaurantLocation, restaurantRating, restaurantCuisine, restProdPrice):
        self.name = restaurantName
        self.location = restaurantLocation
        self.rating = restaurantRating
        self.cuisine = restaurantCuisine
        self.productPrice = restProdPrice

    def __str__(self):
        return f"{self.name}, {self.location}, {self.rating}, {self.cuisine}, {self.productPrice}"

    def csvCreator(self):
        """Creates for each product of restaurant a line of comma separated values (CSV)"""
        lines = []
        for product, price in self.productPrice.items():
            lines.append(f"{self.name};{self.location};{self.rating};{self.cuisine};{product};{price}")
        return lines

def getPage(url):
    """Gets the page content for a given URL and returns it as "soup" """
    # instantiate a Chrome browser
    driver = Driver(uc = True)
    # driver =  uc.Chrome(
    #     use_subprocess=False,
    # )

    # visit the target URL
    driver.get(url)

    # sleep for visibility of change
    time.sleep(2);

    # TODO: Find a better method for scrolling? Identifying the bottom?
    # scroll to the bottom
    stopScrolling = 0
    while True:
        stopScrolling += 1
        driver.execute_script( "window.scrollBy(0, 500)")
        time.sleep(0.5)
        if stopScrolling > 120:
            break

    time.sleep(3);

    page = driver.page_source
    driver.quit()

    # get page content with bs
    soup = BeautifulSoup(page, 'html.parser')

    return soup

def createRestObject(restaurantList):
    """Creates Restaurant objects from restaurant list."""

    # Initialize dictionary for info of products and prices
    productPriceDict = {}
    # Initialize list of restaurant objects
    restaurantObjects = []

    # Loop through all restaurants in soup list
    for restaurantItem in restaurantList:

        # Get different data from restaurantItem
        restaurantName = textValidator(restaurantItem.find('div', class_='restaurant-card_restaurant-name__lEVVi'))
        productName = restaurantItem.find_all('div', class_='restaurant-nested-dishes_product-name__I4Cam')
        productPrice = restaurantItem.find_all("div", class_='restaurant-nested-dishes_product-price__R1kWT')
        for i in range(len(productName)):
            productPriceDict.update({textValidator(productName[i]) : textValidator(productPrice[i])})
        restaurantRating = textValidator(restaurantItem.find("div", class_="restaurant-ratings_rating__yW5tR"))
        restaurantCuisine = textValidator(restaurantItem.find("div", class_ = "restaurant-cuisine_cuisine__SQ_Dc"))

        # Create new restaurant object with data
        restaurantObjects.append(restaurant(restaurantName, location.text, restaurantRating, restaurantCuisine, productPriceDict))

        # Clear dictionary with products and prices for new restaurant
        productPriceDict = {}

    return restaurantObjects

def textValidator(htmlBlock):
    """Method to check if soup object/html code is not empty. Returns "NA" if it is else the text."""

    if htmlBlock != None:
        return htmlBlock.text
    else:
        return "NA"

# Initialize list with all restaurant objects
restaurantExtList = []

# TODO: Method for constructing URLs

# set target url
url = "https://www.just-eat.ch/lieferservice/essen/zuerich-8001?serpRedirect=true&q=Pizza+Margherita"

# get soup of url
soup = getPage(url)

# get location of current soup
location = soup.find("span", class_ ="ygmw2")
# get the list of the restaurants in the soup
restaurantList = soup.find_all('div', class_='restaurant-card-shell_card__IMerh')

# create restaurant objects based on the soup list and add them to a global list of all restaurants
restaurantExtList.extend(createRestObject(restaurantList))

# Check if CSV already exists
if os.path.exists("data/pizza.csv"):
  os.remove("data/pizza.csv")

# create new CSV
with open("data/pizza.csv", "w") as csvFile:
    csvFile.write("restaurant;location;rating (number of ratings);cuisine;product;price\n")
    for restaurantObject in restaurantExtList:
      lines = restaurantObject.csvCreator()
      for line in lines:
         csvFile.write(f"{line}\n")
