import random
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from requests.exceptions import RequestException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class NewsScraper:
    def __init__(self, base_delay=3, random_factor=10):
        self.user_agent = UserAgent()  # Creates a UserAgent object to generate random user-agent
        self.fetch_history = {}  # Dictionary to track last fetch times by domain
        self.base_delay = base_delay  # Base delay in seconds
        self.random_factor = random_factor  # Additional random delay to add variability

    def fetch_html(self, url):
        """Fetch the HTML of the news page, respecting domain-specific delays and error handling."""
        domain = urlparse(url).netloc
        current_time = time.time()

        # Apply random delay for the same domain to respect server load and appear more human-like
        if domain in self.fetch_history:
            elapsed_time = current_time - self.fetch_history[domain]
            delay_time = self.base_delay + random.uniform(0, self.random_factor)
            # print(domain, elapsed_time, delay_time)
            if elapsed_time < delay_time:
                time.sleep(delay_time - elapsed_time)

        headers = {'User-Agent': self.user_agent.random,
                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                   'Accept-Language': 'en-US,en;q=0.5'}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except RequestException:
            # Fallback to Selenium if Requests fails
            return self.selenium_fetch_html(url)
        finally:
            # Update the fetch history with the current time
            self.fetch_history[domain] = time.time()

    def selenium_fetch_html(self, url):
        """Fetch HTML using Selenium as a fallback method."""
        options = Options()
        options.headless = True  # Use headless browser for automation
        options.add_argument(f'user-agent={self.user_agent.random}')
        driver = webdriver.Chrome(options=options)

        try:
            driver.get(url)
            time.sleep(2)  # Allow some time for the page to load dynamically
            return driver.page_source
        except Exception as e:
            print(f"Error using Selenium to fetch page: {e}")
            return None
        finally:
            driver.quit()

    def parse_article(self, url):
        """Parse the HTML to retrieve full news article text. Customization needed per site."""
        html = self.fetch_html(url)
        if html is not None:
            soup = BeautifulSoup(html, 'html.parser')
            article_text = ' '.join(p.get_text().strip() for p in soup.find_all('p'))
            print(urlparse(url).netloc, (article_text[:100] + '...' + article_text[-100:]).replace('\n', ' '))
            return article_text
        else:
            return "Failed to retrieve article."

    def parse_articles_batch(self, urls):
        """Parse multiple articles in a batch from a list of URLs."""
        results = {}
        for url in urls:
            article_text = self.parse_article(url)
            results[url] = article_text
        return results
