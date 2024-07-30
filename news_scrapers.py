import logging
import random
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from requests.exceptions import RequestException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import html_functions


class NewsScraper:
    def __init__(self, base_delay=3, random_factor=7, max_retries=2, domain=None):
        if domain is not None:
            _base_delay = html_functions.delay_times.get(domain, {}).get('base_delay', base_delay)
            _random_factor = html_functions.delay_times.get(domain, {}).get('random_factor', random_factor)
            base_delay = _base_delay if _base_delay is not None else base_delay
            random_factor = _random_factor if _random_factor is not None else random_factor
        self.user_agent = UserAgent()  # Creates a UserAgent object to generate random user-agent
        self.fetch_history = {}  # Dictionary to track last fetch times by domain
        self.failure_count = {}  # Track failure counts by domain
        self.base_delay = base_delay  # Base delay in seconds
        self.random_factor = random_factor  # Additional random delay to add variability
        self.max_retries = max_retries  # Maximum number of retries per domain

    def fetch_html(self, url):
        """Fetch the HTML of the news page, respecting domain-specific delays and error handling."""
        split = urlparse(url).netloc.split('.')
        domain = split[1] if len(split) > 2 else split[0]

        if domain in html_functions.skipped_domains:
            return None

        current_time = time.time()
        retry_count = self.failure_count.get(domain, 0)

        while retry_count < self.max_retries:
            if domain in self.fetch_history:
                elapsed_time = current_time - self.fetch_history[domain]
                delay_time = self.base_delay + random.uniform(0, self.random_factor)
                if elapsed_time < delay_time:
                    time.sleep(delay_time - elapsed_time)

            headers = {
                'User-Agent': self.user_agent.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                html = response.text

                if html_functions.page_not_available(html, domain):
                    logging.error(f"Page not available at {url}")
                    return None

                if html_functions.captcha_encountered(html, domain):
                    logging.error(f"Access issue at {url}")
                    retry_count += 1
                    self.failure_count[domain] = retry_count
                    time.sleep(retry_count * 10)  # Increase wait time for each retry
                    continue
                self.failure_count[domain] = 0
                return html
            except requests.RequestException as e:
                html = self.selenium_fetch_html(url)
                if html:
                    self.failure_count[domain] = 0
                    return html
                logging.error(f"Request failed for {url}: {str(e)}")
                print(f"Request failed for {url}: {str(e)}")
                retry_count += 1
                self.failure_count[domain] = retry_count
                continue
            finally:
                self.fetch_history[domain] = time.time()
        return None  # Return None after max retries

    def selenium_fetch_html(self, url):
        options = Options()
        options.headless = True
        options.add_argument("--headless=new")
        options.add_argument(f'user-agent={self.user_agent.random}')
        driver = webdriver.Chrome(options=options)

        split = urlparse(url).netloc.split('.')
        domain = split[1] if len(split) > 2 else split[0]

        try:
            driver.get(url)
            time.sleep(2)
            html = driver.page_source

            if html_functions.page_not_available(html, domain):
                logging.error(f"Page not available at {url}")
                return None

            if html_functions.captcha_encountered(html, domain):
                logging.error(f"Access issue at {url} using Selenium")
                return None
            return html
        except Exception as e:
            logging.error(f"Selenium fetch error for {url}: {str(e)}")
            return None
        finally:
            driver.quit()

    def parse_article(self, url, html_dict=None, _html=None):
        """Parse the HTML to retrieve full news article text."""
        split = urlparse(url).netloc.split('.')
        domain = split[1] if len(split) > 2 else split[0]
        html = self.fetch_html(url) if _html is None else _html
        if html_dict is not None:
            html_dict[url] = html
        if html:
            prc_txt = lambda x: ' '.join(p.get_text().strip() for p in x)
            fn_dict = html_functions.domain_specific_full_text_select_mixed_functions
            find_what = fn_dict.get(domain)
            if callable(find_what):
                article_text = find_what(html)
                if not article_text:
                    return ''
                if isinstance(article_text, tuple):
                    article_text = [prc_txt(x) for x in article_text]
                    len_lst = [len(x) for x in article_text]
                    article_text = article_text[len_lst.index(max(len_lst))]
                elif not isinstance(article_text, str):
                    article_text = ' '.join([prc_txt(x) for x in article_text])
                else:
                    article_text = prc_txt(article_text)
            else:
                tag_with_most_p_tags = html_functions.find_tag_with_most_tags(html=html, tags=('p',))
                if tag_with_most_p_tags is None:
                    return ''
                get_txt_from_tag = lambda *args: prc_txt(tag_with_most_p_tags.find_all(*args))

                # text_all = get_txt_from_tag()
                # force_all_text_factor = 3
                # force_all_text_if_short = lambda x: text_all if len(text_all) > force_all_text_factor * len(x) else x
                if find_what is None:
                    article_text = get_txt_from_tag('p')
                elif isinstance(find_what, str):
                    article_text = get_txt_from_tag(find_what)
                else:
                    article_text = get_txt_from_tag(*find_what)
            # article_text = force_all_text_if_short(article_text)
            ln2 = len(article_text) // 2
            print(url, f'len:{len(article_text)}', (article_text[:100] + '...' + article_text[ln2:ln2 + 100] + '...' + article_text[-100:]).replace('\n', ' '))
            return article_text
        return ''

    def parse_articles_batch(self, urls, **kwargs):
        """Parse multiple articles in a batch from a list of URLs."""
        results = {}
        for url in urls:
            article_text = self.parse_article(url, **kwargs)
            if article_text != "Failed to retrieve article.":
                results[url] = article_text
            else:
                logging.info(f"Article retrieval failed for {url}")
        return results


if __name__ == '__main__':
    scraper = NewsScraper()
    # articles = scraper.parse_articles_batch(["https://www.benzinga.com/news/23/10/35190005/apple-amazon-and-2-other-stocks-insiders-are-selling"])  # X
    # articles = scraper.parse_articles_batch(["https://www.benzinga.com/pressreleases/24/05/n38876463/embracing-the-ai-era-imf-highlights-massive-potential-for-global-workforce-ai-innovation"])
    # articles = scraper.parse_articles_batch(["https://edition.cnn.com/2024/03/25/tech/digital-markets-act-apple-google-meta"])
    # articles = scraper.parse_articles_batch(["https://www.business-standard.com/technology/tech-news/apple-debuts-long-awaited-ai-tools-including-chatgpt-tie-up-at-wwdc-124061100082_1.html"])
    # articles = scraper.parse_articles_batch(["https://pro.thestreet.com/story/16135076/1/let-s-look-at-amazon-s-big-day-why-this-company-is-no-small-fry-for-mcdonald-s.html"])
    # articles = scraper.parse_articles_batch(["https://www.investors.com/research/ibd-stock-analysis/amazon-stock-ai-bedrock-aws/"])
    # articles = scraper.parse_articles_batch(["https://www.reuters.com/business/aerospace-defense/who-will-save-struggling-airline-sas-2023-09-27/"])
    # articles = scraper.parse_articles_batch(["https://www.cnn.com/2023/09/27/sport/henrikh-mkhitaryan-armenia-nagorno-karabakh-spt-intl/index.html"])
    # articles = scraper.parse_articles_batch(["https://www.cnbc.com/2024/07/05/crypto-market-bloodbath-as-mt-gox-bitcoin-btc-payout-approaches.html"])
    # articles = scraper.parse_articles_batch(["https://decrypt.co/238283/south-korean-crypto-exchanges-guidelines-mass-delistings"])
    # articles = scraper.parse_articles_batch(["https://investingnews.com/illumina-appoints-everett-cunningham-chief-commercial-officer/"])
    print(articles)


 # def __init__(self, base_delay=3, random_factor=10):
 #        self.user_agent = UserAgent()  # Creates a UserAgent object to generate random user-agent
 #        self.fetch_history = {}  # Dictionary to track last fetch times by domain
 #        self.base_delay = base_delay  # Base delay in seconds
 #        self.random_factor = random_factor  # Additional random delay to add variability
 #
 #    def fetch_html(self, url):
 #        """Fetch the HTML of the news page, respecting domain-specific delays and error handling."""
 #        domain = urlparse(url).netloc
 #        name = domain.split('.')[1] if len(domain.split('.')) > 2 else domain.split('.')[0]
 #        current_time = time.time()
 #
 #        # Apply random delay for the same domain to respect server load and appear more human-like
 #        if domain in self.fetch_history:
 #            elapsed_time = current_time - self.fetch_history[domain]
 #            delay_time = self.base_delay + random.uniform(0, self.random_factor)
 #            # print(domain, elapsed_time, delay_time)
 #            if elapsed_time < delay_time:
 #                time.sleep(delay_time - elapsed_time)
 #
 #        headers = {'User-Agent': self.user_agent.random,
 #                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
 #                   'Accept-Language': 'en-US,en;q=0.5'}
 #        try:
 #            response = requests.get(url, headers=headers)
 #            response.raise_for_status()
 #            return response.text
 #        except RequestException:
 #            # Fallback to Selenium if Requests fails
 #            return self.selenium_fetch_html(url)
 #        finally:
 #            # Update the fetch history with the current time
 #            self.fetch_history[domain] = time.time()