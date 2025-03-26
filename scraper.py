import requests # not used
from bs4 import BeautifulSoup
import pandas as pd
from seleniumbase import Driver
from selenium import webdriver # not used
import time
import os

class Restaurant: # class naming convention PascalCase
    """Represents a restaurant and its attributes"""
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
            # Ensure to correctly format the product and price line
            line = f"{self.name};{self.location};{self.rating};{self.cuisine};{product};{price}"
            lines.append(line)
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

def extract_restaurants(soup, location):
    """Creates Restaurant objects from restaurant list."""
    # Initialize list of restaurant objects
    restaurantObjects = []
    restaurant_items = soup.find_all("div", class_ = "restaurant-card-shell_card__IMerh")

    # print only for debugging
    # print(f"Found {len(restaurant_items)} restaurants in {location}")

    if not restaurant_items:
        print("No restaurants found! The structure might have changed.")
        return []

    # Loop through all restaurants in soup list
    for restaurantItem in restaurant_items:
        productPriceDict = {}
        # Get different data from restaurantItem
        restaurantName = textValidator(restaurantItem.find('div', class_='restaurant-card_restaurant-name__lEVVi'))
        productName = restaurantItem.find_all('div', class_='restaurant-nested-dishes_product-name__I4Cam')
        productPrice = restaurantItem.find_all("div", class_='restaurant-nested-dishes_product-price__R1kWT')

        for i in range(len(productName)):
            productPriceDict.update({textValidator(productName[i]) : textValidator(productPrice[i])})

        restaurantRating = textValidator(restaurantItem.find("div", class_="restaurant-ratings_rating__yW5tR"))
        restaurantCuisine = textValidator(restaurantItem.find("div", class_ = "restaurant-cuisine_cuisine__SQ_Dc"))

        # Print only for debugging
        # print(f"Scraped: {restaurantName}, Rating: {restaurantRating}, Cuisine: {restaurantCuisine}")

        # Create new restaurant object with data
        restaurantObjects.append(Restaurant(restaurantName, location, restaurantRating, restaurantCuisine, productPriceDict))

        # Clear dictionary with products and prices for new restaurant
        productPriceDict = {}

    return restaurantObjects

def textValidator(htmlBlock):
    """Method to check if soup object/html code is not empty. Returns "NA" if it is else the text."""
    return htmlBlock.text if htmlBlock else "NA"


def save_to_csv(restaurants, filname):
    # Check if CSV already exists
    if os.path.exists("data/pizza.csv"):
        os.remove("data/pizza.csv")

    # create new CSV
    with open("data/pizza.csv", "w") as csvFile:
        csvFile.write("restaurant;location;rating (number of ratings);cuisine;product;price\n")
        for restaurantObject in restaurants:
            lines = restaurantObject.csvCreator()
            for line in lines:
                csvFile.write(f"{line}\n")

def main():
    """Main function to scrape multiple locations and save data"""
    # set target url and list of postal codes to scrape
    base_url = "https://www.just-eat.ch/lieferservice/essen/{}?serpRedirect=true&q=Pizza+Margherita"
    postal_codes = ["basel-4051", "bern-3011"]

    all_restaurants = []

    # loop through the different urls
    for postal_code in postal_codes:
        # format URL with postal code and search term variations
        url = base_url.format(postal_code)
        print(f"Scraping: {url}")  # only as check

        # get soup of url
        soup = getPage(url)
        # get location of current soup
        location_elem = soup.find("span", class_="ygmw2")
        location = textValidator(location_elem)

        all_restaurants.extend(extract_restaurants(soup, location))

    save_to_csv(all_restaurants, "data/pizza.csv")
    print("Scraping complete") # check

if __name__ == "__main__":
    main()