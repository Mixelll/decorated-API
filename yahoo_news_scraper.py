import logging
import random
import time
from datetime import datetime

import dateparser
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager


logging.getLogger('fake_useragent').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class YahooFinanceTickerNewsScraper:
    def __init__(self, ticker, start_date, end_date, max_retries=3, proxy_list=None, fetch_full_text=False, max_scrolls=5):
        self.base_url = f"https://finance.yahoo.com/quote/{ticker}/news"
        self.start_date = self.parse_date(start_date)
        self.end_date = self.parse_date(end_date)
        self.max_retries = max_retries
        self.proxy_list = proxy_list or []
        self.ua = UserAgent()
        self.fetch_full_text = fetch_full_text
        self.max_scrolls = max_scrolls
        self.news_items = []
        self.seen_links = set()

    def parse_date(self, date_input):
        if isinstance(date_input, str):
            return dateparser.parse(date_input)
        elif isinstance(date_input, datetime):
            return date_input
        else:
            raise ValueError("Date input must be a string or datetime object")

    def get_random_proxy(self):
        if self.proxy_list:
            return random.choice(self.proxy_list)
        return None

    def fetch_page(self, url):
        headers = {'User-Agent': self.ua.random}
        proxy = self.get_random_proxy()
        proxies = {'http': proxy, 'https': proxy} if proxy else None

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
                if response.status_code == 200:
                    return response.content
                else:
                    logging.warning(f"Non-200 status code: {response.status_code}")
            except requests.RequestException as e:
                logging.error(f"Request failed: {e}")
            # Exponential backoff with jitter
            time.sleep((2 ** attempt) + random.uniform(0, 1))
        return None

    def parse_news(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        items = soup.find_all('div', class_='content svelte-1v1zaak')
        articles = [x for x in items if '<span>Ad</span>' not in str(x)]

        for article in articles:
            headline = article.find('h3').text
            link = article.find('a')['href']
            full_link = link

            if full_link in self.seen_links:
                continue
            self.seen_links.add(full_link)

            full_text, article_date = self.fetch_full_article(full_link) if self.fetch_full_text else (None, None)
            self.news_items.append({'headline': headline, 'link': full_link, 'date': article_date, 'full_text': full_text})

    def fetch_full_article(self, url):
        content = self.fetch_page(url)
        if content:
            soup = BeautifulSoup(content, 'html.parser')

            # Extract the full text
            paragraphs = soup.find_all('p')
            full_text = ' '.join([para.get_text() for para in paragraphs])

            # Extract the publication date
            date_span = soup.find('time')  # Adjust the selector based on the actual HTML structure
            article_date = dateparser.parse(
                date_span['datetime']) if date_span and 'datetime' in date_span.attrs else None

            return full_text, article_date
        return None, None

    def scroll_and_scrape(self):
        options = Options()
        options.headless = False  # Run in visible mode to simulate real user behavior
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

        logging.info(f"Opening {self.base_url}")
        driver.get(self.base_url)
        time.sleep(random.uniform(3, 5))  # Wait for the page to load

        last_height = driver.execute_script("return document.body.scrollHeight")

        scroll_count = 0
        while scroll_count < self.max_scrolls:
            scroll_count += 1
            # Scroll down to the bottom of the page
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(random.uniform(0.5, 1))  # Random pause for more realistic scrolling behavior

            # Parse the loaded page content
            html_content = driver.page_source

            print(scroll_count)
            # Check if the last news item's date is before the start date
            # if self.news_items and self.news_items[-1]['date'] < self.start_date:
            #     break

            # Check if the scroll position has reached the bottom
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        t = time.time()
        self.parse_news(html_content)
        print(f'Parsed news in {time.time() - t:.2f} seconds')
        driver.quit()

    def scrape_news(self):
        self.scroll_and_scrape()
        return self.news_items


if __name__ == "__main__":
    ticker = "AAPL"  # Replace with the ticker symbol you want to scrape news for
    start_date = "2022-01-01"  # Start date in YYYY-MM-DD format or as a datetime object
    end_date = datetime(2022, 12, 31)  # End date in YYYY-MM-DD format or as a datetime object
    proxy_list = []  # Add your proxy list here if needed
    fetch_full_text = True  # Set to True to fetch full text of each article
    max_scrolls = 5  # Set the maximum number of scrolls
    scraper = YahooFinanceTickerNewsScraper(ticker, start_date, end_date, proxy_list=proxy_list,
                                            fetch_full_text=fetch_full_text, max_scrolls=max_scrolls)
    news = scraper.scrape_news()

    for item in news:
        print(f"Date: {item['date']}")
        print(f"Headline: {item['headline']}")
        print(f"Link: {item['link']}")
        if item['full_text']:
            print(f"Full Text: {item['full_text'][:200]}...")  # Print the first 200 characters of the full text
        print()