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
    def __init__(self, restaurant_name, restaurant_location, restaurant_rating, restaurant_cuisine, rest_prod_price, restaurant_del_time, restaurant_del_cost, restaurant_min_ord):
        self.name = restaurant_name
        self.location = restaurant_location
        self.rating = restaurant_rating
        self.cuisine = restaurant_cuisine
        self.product_price = rest_prod_price
        self.del_time = restaurant_del_time
        self.del_cost = restaurant_del_cost
        self.min_ord = restaurant_min_ord

    def __str__(self):
        return f"{self.name}, {self.location}, {self.rating}, {self.cuisine}, {self.product_price}, {self.del_time}, {self.del_cost}, {self.min_ord}"

    def csv_creator(self):
        """Creates for each product of restaurant a line of comma separated values (CSV)"""
        lines = []
        for product, price in self.product_price.items():
            # Ensure to correctly format the product and price line
            line = f"{self.name};{self.location};{self.rating};{self.cuisine};{product};{price};{self.del_time};{self.del_cost};{self.min_ord}"
            lines.append(line)
        return lines

def get_page(url):
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

    # Scroll to the bottom to uncover new restaurants
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    time.sleep(3);

    page = driver.page_source
    driver.quit()

    # get page content with bs
    soup = BeautifulSoup(page, 'html.parser')

    return soup

def extract_restaurants(soup, location):
    """Creates Restaurant objects from restaurant list."""
    # Initialize list of restaurant objects
    restaurant_objects = []
    restaurant_items = soup.find_all("div", class_ = "restaurant-card-shell_card__IMerh")

    # print only for debugging
    # print(f"Found {len(restaurant_items)} restaurants in {location}")

    if not restaurant_items:
        print("No restaurants found! The structure might have changed.")
        return []

    # Loop through all restaurants in soup list
    for restaurantItem in restaurant_items:
        product_price_dict = {}
        # Get different data from restaurantItem
        restaurant_name = text_validator(restaurantItem.find('div', class_='restaurant-card_restaurant-name__lEVVi'))
        product_name = restaurantItem.find_all('div', class_='restaurant-nested-dishes_product-name__I4Cam')
        product_price = restaurantItem.find_all("div", class_='restaurant-nested-dishes_product-price__R1kWT')

        for i in range(len(product_name)):
            product_price_dict.update({text_validator(product_name[i]) : text_validator(product_price[i])})

        restaurant_rating = text_validator(restaurantItem.find("div", class_="restaurant-ratings_rating__yW5tR"))
        restaurant_cuisine = text_validator(restaurantItem.find("div", class_ ="restaurant-cuisine_cuisine__SQ_Dc"))
        restaurant_del_time = text_validator(restaurantItem.find("div", attrs={"data-qa": "restaurant-eta"}))
        restaurant_del_cost = text_validator(restaurantItem.find("div", attrs={"data-qa": "restaurant-delivery-fee"}))
        restaurant_min_ord = text_validator(restaurantItem.find("div", attrs={"data-qa": "restaurant-mov"}))

        # Print only for debugging
        # print(f"Scraped: {restaurant_name}, Rating: {restaurant_rating}, Cuisine: {restaurantCuisine}")

        # Create new restaurant object with data
        restaurant_objects.append(Restaurant(restaurant_name, location, restaurant_rating, restaurant_cuisine, product_price_dict, restaurant_del_time, restaurant_del_cost, restaurant_min_ord))

        # Clear dictionary with products and prices for new restaurant
        product_price_dict = {}

    return restaurant_objects

def text_validator(html_block):
    """Method to check if soup object/html code is not empty. Returns "NA" if it is else the text."""
    return html_block.text if html_block else "NA"


def save_to_csv(restaurants):
    # Check if CSV already exists
    if os.path.exists("data/pizza.csv"):
        os.remove("data/pizza.csv")

    # create new CSV
    with open("data/pizza.csv", "w") as csvFile:
        csvFile.write("restaurant;location;rating (number of ratings);cuisine;product;price;delivery time;delivery fee;minimum order value\n")
        for restaurantObject in restaurants:
            lines = restaurantObject.csv_creator()
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
        soup = get_page(url)
        # get location of current soup
        location_elem = soup.find("span", class_="ygmw2")
        location = text_validator(location_elem)

        all_restaurants.extend(extract_restaurants(soup, location))

    save_to_csv(all_restaurants, "data/pizza.csv")
    print("Scraping complete") # check

if __name__ == "__main__":
    main()