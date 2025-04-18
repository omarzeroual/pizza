"""
Module Name: main.py
Description: This modules is for running the whole project in one go.
Author: Omar, Valérie
Date: 2025-04-08
Version: 1.0
"""
import scraper
import get_api
import cleanertransformer
import analyze
import visualizer

def main():
    get_api.main()
    scraper.main()
    cleanertransformer.main()
    analyze.main()
    visualizer.main()

if __name__ == "__main__":
    main()
