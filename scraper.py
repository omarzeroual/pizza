"""
Module Name: scraper.py
Description: This modules scrapes product information from a online delivery service. The data is output as a csv file.
Author:
Date: 2025-03-27
Version: 0.1
"""

from bs4 import BeautifulSoup
from seleniumbase import Driver
import time
import os

class Restaurant: # class naming convention PascalCase
    """Represents a restaurant and its attributes"""
    def __init__(self, restaurantName, restaurantLocation, restaurantRating, restaurantCuisine, restProdPrice, restaurantDelTime, restaurantDelCost, restaurantMinOrd):
        self.name = restaurantName
        self.location = restaurantLocation
        self.rating = restaurantRating
        self.cuisine = restaurantCuisine
        self.productPrice = restProdPrice
        self.delTime = restaurantDelTime
        self.delCost = restaurantDelCost
        self.minOrd = restaurantMinOrd

    def __str__(self):
        return f"{self.name}, {self.location}, {self.rating}, {self.cuisine}, {self.productPrice}, {self.delTime}, {self.delCost}, {self.minOrd}"

    def csvCreator(self):
        """Creates for each product of restaurant a line of comma separated values (CSV)"""
        lines = []
        for product, price in self.productPrice.items():
            # Ensure to correctly format the product and price line
            line = f"{self.name};{self.location};{self.rating};{self.cuisine};{product};{price};{self.delTime};{self.delCost};{self.minOrd}"
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

    # Slowly scroll to the bottom to uncover new restaurants
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
        restaurantDelTime = textValidator(restaurantItem.find("div", attrs={"data-qa": "restaurant-eta"}))
        restaurantDelCost = textValidator(restaurantItem.find("div", attrs={"data-qa": "restaurant-delivery-fee"}))
        restaurantMinOrd = textValidator(restaurantItem.find("div", attrs={"data-qa":"restaurant-mov"}))


        # Print only for debugging
        # print(f"Scraped: {restaurantName}, Rating: {restaurantRating}, Cuisine: {restaurantCuisine}")

        # Create new restaurant object with data
        restaurantObjects.append(Restaurant(restaurantName, location, restaurantRating, restaurantCuisine, productPriceDict, restaurantDelTime, restaurantDelCost, restaurantMinOrd))

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
        csvFile.write("restaurant;location;rating (number of ratings);cuisine;product;price;delivery time;delivery fee;minimum order value\n")
        for restaurantObject in restaurants:
            lines = restaurantObject.csvCreator()
            for line in lines:
                csvFile.write(f"{line}\n")

def main():
    """Main function to scrape multiple locations and save data"""
    # set target url and list of postal codes to scrape
    base_url = "https://www.just-eat.ch/lieferservice/essen/{}?serpRedirect=true&q=Pizza+Margherita"
    postal_codes = ["zuerich-8001", "genf-1204", "basel-4051", "lausanne-1003", "bern-3011", "winterthur-8400", "luzern-6003", "sankt-gallen-9000", "lugano-6900", "biel-2502"]

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