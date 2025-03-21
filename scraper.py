import requests
from bs4 import BeautifulSoup
import pandas as pd
from seleniumbase import Driver
from selenium import webdriver
import time
#import undetected_chromedriver as uc


# set target url
url = "https://www.just-eat.ch/lieferservice/essen/zuerich-8001?serpRedirect=true&q=Pizza+Margherita"

# instantiate a Chrome browser
driver = Driver(uc = True)
# driver =  uc.Chrome(
#     use_subprocess=False,
# )

# visit the target URL
driver.get(url)

# sleep for visibility of change
time.sleep(2);

#element = driver.find_element("footer-style_footer__NogL5")

# scroll to the bottom
#driver.execute_script("arguments[0].scrollIntoView(true)", element)
time.sleep(3);

page = driver.page_source
driver.quit()

# get page content with bs
soup = BeautifulSoup(page, 'html.parser')

restaurantList = soup.find_all('div', class_='restaurant-card-shell_card__IMerh')

for restaurant in restaurantList:
    restaurantName = restaurant.find('div', class_='restaurant-card_restaurant-name__lEVVi')
    productName = restaurant.find('div', class_='restaurant-nested-dishes_product-name__I4Cam')
    productPrice = restaurant.find("div", class_='restaurant-nested-dishes_product-price__R1kWT')

    print(restaurantName.text)
    print(productName.text)
    print(productPrice.text)

#restaurantNames = soup.find_all("div", class_="restaurant-card_restaurant-name__lEVVi")
