import inspect
import json
import logging
import random
from datetime import datetime

import pandas as pd
import requests
from fake_useragent import UserAgent

import decortools as dt
import m_db as mdb
from news_scrapers import NewsScraper
from api_keys import ALPHA_VANTAGE_KEY
import datetime_functions as dtf


logging.getLogger('fake_useragent').setLevel(logging.ERROR)
datetime_column = 'time_published'

TICKER = 'GSPC'  # 'SPY'
TABLE_ID = dict(schema='news_data', table_name='all_news', primary_keys=['url'])
urls_in_db_col = 'get_full_text'
attrs_date_range = 'date_range'


@dt.df_manipulator_decorator(dtf.series_str2datetime, apply_func_to_series=datetime_column, after=True)
@mdb.return_df_rows_not_in_table(**TABLE_ID, suppress_error_no_table_exists=True, add_column_instead=urls_in_db_col)
def get_historical_news(api_key, symbol, limit=1000, topics=None, time_from=None, time_to=None, sort=None, test_domains=False, domain=None, debug=False):
    print(f"Fetching news for {symbol} from {time_from} to {time_to}.")
    """
    Fetch historical news articles for a given stock symbol using the Alpha Vantage API and optionally test domain access.

    Args:
        api_key (str): Your Alpha Vantage API key.
        symbol (str): The stock symbol (e.g., 'AAPL').
        limit (int): Maximum number of articles to fetch.
        topics (str): Comma-separated list of topics to filter by.
        time_from (str): Start time in the format 'YYYYMMDDTHHMM'.
        time_to (str): End time in the format 'YYYYMMDDTHHMM'.
        sort (str): Order of articles ('LATEST', 'EARLIEST', or 'RELEVANCE').
        test_domains (bool): If True, performs accessibility test on the domains of fetched articles.
        domain (str): If provided, only tests the specified domain.

    Returns:
        None: Prints the retrieved news articles or test results.
    """
    base_url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
    params = {
        'tickers': symbol,
        'apikey': api_key,
        'limit': limit,
        'topics': topics,
        'time_from': time_from,
        'time_to': time_to,
        'sort': sort
    }
    if debug:
        table_id = TABLE_ID.copy()
        table_id.pop('primary_keys')
        return mdb.get_table_as_df(**table_id)
    try:
        response = requests.get(base_url, params={k: v for k, v in params.items() if v is not None})
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return

    if "feed" in data and data["feed"]:
        news_articles = data["feed"]

        if test_domains:
            return test_article_accessibility(news_articles, select_domain=domain)
        else:
            print(len(news_articles), "news articles found.")
            news_articles = pd.DataFrame(news_articles)
            news_articles.drop_duplicates(subset='url', inplace=True)
            return news_articles
    else:
        logging.error(data)
        if test_domains:
            print("No news articles found.")
        else:
            raise ValueError(data)


def test_article_accessibility(news_articles, select_domain=None):
    domain_articles = {}
    for article in news_articles:
        domain = article.get('source_domain', 'Unknown domain')
        if domain not in domain_articles:
            domain_articles[domain] = []
        domain_articles[domain].append(article)

    results = {domain: {'accessible': [], 'inaccessible': []} for domain in domain_articles.keys()}

    user_agent = UserAgent()

    for domain, articles in domain_articles.items():
        if select_domain is not None and select_domain != domain:
            continue
        test_articles = random.sample(articles, min(2, len(articles)))
        print(f"Testing domain: {domain}")
        for article in test_articles:
            url = article['url']
            try:
                # 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
                headers = {'User-Agent': user_agent.random, 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                           'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://www.google.com/'}
                article_response = requests.get(url, headers=headers)
                article_response.raise_for_status()
                results[domain]['accessible'].append(url)
                print(f"Accessible: {url}")
            except requests.exceptions.RequestException:
                results[domain]['inaccessible'].append(url)
                print(f"Inaccessible: {url}")
        print("\n" + "=" * 50 + "\n")

    accessible_domains = []
    partial_domains = {}
    inaccessible_domains = []
    for domain, access_info in results.items():
        acs = access_info['accessible']
        inacs = access_info['inaccessible']
        if not inacs:
            accessible_domains.append(domain)
        elif acs and inacs:
            partial_domains[domain] = f'{len(acs)}/{len(inacs)}'
        else:
            inaccessible_domains.append(domain)
    print("Accessible domains:", accessible_domains)
    print("Partially accessible domains:", partial_domains)
    print("Inaccessible domains:", inaccessible_domains)
    print("\n" + "=" * 50 + "\n")


@dt.dynamic_date_range_decorator(start_name='time_from', end_name='time_to', max_period='12h', result_date_accessor_fn=lambda x: x.attrs[attrs_date_range], aggregate_fn=None)  # lambda x: pd.concat(x))
@mdb.upsert_df2db_decorator(**TABLE_ID)
# @dt.df_manipulator_decorator(dt.concurrent_groupby_apply, groupby='domain', after=False, pass_function=True)
@dt.copy_signature(get_historical_news)
def get_historical_news_full(*args, **kwargs):
    t = datetime.now()
    print(kwargs.get('time_from'), kwargs.get('time_to'))
    news_articles = get_historical_news(*args, **kwargs)
    date_range = news_articles[datetime_column].agg(['min', 'max'])
    if urls_in_db_col in news_articles.columns:
        news_articles = news_articles[news_articles[urls_in_db_col]]
        news_articles.drop(columns=[urls_in_db_col], inplace=True)
    news_articles.attrs[attrs_date_range] = date_range
    # if news_articles is None or isinstance(news_articles, pd.DataFrame) and news_articles.empty:
    #     return news_articles
    domains = set(news_articles['source_domain'].str.replace('www.', ''))
    if len(domains) == 1:
        domain = domains.pop()
        scraper = NewsScraper(domain=domain)
    else:
        scraper = NewsScraper()
    html_dict = {}
    # Use the batch processing method provided by the NewsScraper class
    if hasattr(scraper, 'parse_articles_batch'):
        urls = news_articles['url'].tolist()
        print(f'Fetching full articles for {len(urls)} news articles.')
        full_texts_dict = scraper.parse_articles_batch(urls, html_dict=html_dict)
        # Convert dictionary results to align with the DataFrame
        news_articles['full_text'] = news_articles['url'].map(full_texts_dict)
    else:
        # Fall back to row-wise processing if batch processing is not available
        news_articles['full_text'] = news_articles['url'].apply(scraper.parse_article, html_dict=html_dict)
    news_articles['html'] = news_articles['url'].map(html_dict).apply(lambda x: x.replace('\0', '') if x is not None else None)
    print(f'Fetched full articles for {len(news_articles)} news articles in {datetime.now() - t}.')
    return news_articles


# Example usage
if __name__ == "__main__":
    api_key_ = ALPHA_VANTAGE_KEY
    # df = get_historical_news(api_key_, TICKER, limit=50, topics='technology',  time_from='20230101T0000',
    #                            time_to=None, test_domains=False)
    # df
    # , topics='technology'
    df = get_historical_news_full(api_key_, None, limit=1000, time_from='20230101T0000',
                               time_to='20240115T1120', test_domains=False)
    df
#     20240125T1700
