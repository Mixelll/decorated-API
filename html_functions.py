from bs4 import BeautifulSoup


domain_specific_full_text_select_mixed_functions = {
    'marketwatch': ('div', 'article__content'),
    'business-standard': lambda x: BeautifulSoup(x, 'html.parser').select('div.MainStory_storycontent__Pe3ys > div:not([id]):not([class])'),
    'benzinga': lambda x: (find_tag_with_most_tags(x, tags=('p',)).find_all('p'), find_tag_with_most_tags(x, tags=('ul',)).find_all('ul')),
    'cnn': lambda x: BeautifulSoup(x, 'html.parser').select('div.article__content p'),
    # 'barrons': lambda x: BeautifulSoup(x, 'html.parser').select('div.article-content > p'),
    'reuters': lambda x: find_tag_with_most_tags(x, tags=('div',)).find_all('div'),
    'cnbc': lambda x: BeautifulSoup(x, 'html.parser').select('div.ArticleBody-articleBody p'),
}


domain_specific_captcha_encountered = {
    'investors': ['PerimeterX'],
    'reuters': ['captcha-delivery']
}

domain_specific_min_html_length = {
    'reuters': 1600
}

domain_specific_length_and_captcha = {
    'reuters',
}

domain_specific_page_not_available = {}

delay_times = {
    'marketwatch': dict(base_delay=15, random_factor=30),
    'investingnews': dict(base_delay=5, random_factor=10)
}

skipped_domains = ['aljazeera']


def find_tag_with_most_tags(html=None, parser=None, tags=()):
    """
    Find the tag with the most <p> tags within the HTML content or BeautifulSoup parser object.
    html (str, optional): HTML content of the webpage.
    parser (BeautifulSoup, optional): BeautifulSoup parser object of the webpage.
    recursive (bool, optional): If True, the function will search for nested tags with the most <p> tags. More efficient
    """
    if html is None and parser is None:
        raise ValueError("Either HTML content or a BeautifulSoup parser object must be provided.")
    if html is not None and parser is not None:
        raise ValueError("Only one of HTML content or a BeautifulSoup parser object should be provided.")

    if html is not None:
        soup = BeautifulSoup(html, 'lxml')
    else:
        soup = parser

    # Find the tag with the maximum number of <p> tags
    max_child_count = 0
    tag_with_most_child_tags = None
    # Iterate over all tags in the soup
    for tag in soup.find_all():
        # Count direct <p> children
        tag_count = len(tag.find_all(*tags, recursive=False))
        # Update if this tag has more <p> tags than any previous ones
        if tag_count > max_child_count:
            max_child_count = tag_count
            tag_with_most_child_tags = tag
    return tag_with_most_child_tags


def check_html_indicators(html, indicators):
    """
    Base function to check HTML content against specified indicators and a minimum length.

    Args:
        html (str): HTML content of the webpage.
        indicators (list): List of strings that indicate an issue when found in HTML.
    Returns:
        bool: True if any indicator is found or if the content is shorter than the minimum length, False otherwise.

    Raises:
        ValueError: If the html or indicators are not properly provided.
    """
    html = html.lower()
    if not html or not isinstance(html, str):
        return True
    if not isinstance(indicators, list) or not all(isinstance(i, str) for i in indicators):
        raise ValueError("Indicators must be a list of strings.")
    return any(indicator in html for indicator in indicators)


def check_html_length(html, min_html_length=500):
    """
    Base function to check HTML content against specified indicators and a minimum length.

    Args:
        html (str): HTML content of the webpage.
        min_html_length (int): Minimum acceptable length of HTML content to consider the page valid.

    Returns:
        bool: True if any indicator is found or if the content is shorter than the minimum length, False otherwise.

    Raises:
        ValueError: If the html or indicators are not properly provided.
    """
    if not html or not isinstance(html, str):
        return True
    if min_html_length is not None and isinstance(min_html_length, int):
        if len(html) < min_html_length:
            return True  # Content is too short, potentially indicating an issue
    return False


def captcha_encountered(html, domain=None, domain_only=False):
    """
    Check if a captcha challenge is present on a webpage.

    Args:
        html (str): HTML content of the webpage.
        domain (str, optional): The domain of the webpage for domain-specific rules.
        domain_only (bool, optional): If True, only domain-specific checks will be performed.

    Returns:
        bool: True if a captcha is detected, False otherwise.
    """
    general_captcha_indicators = []
    domain_specific_indicators = domain_specific_captcha_encountered

    indicators = general_captcha_indicators
    if domain and domain in domain_specific_indicators:
        if domain_only:
            indicators = domain_specific_indicators[domain]
        else:
            indicators.extend(domain_specific_indicators[domain])
    length_kw = {'min_html_length': domain_specific_min_html_length[domain]} if domain and domain in domain_specific_min_html_length else {}
    _and = True if domain and domain in domain_specific_length_and_captcha else False
    if _and:
        return check_html_indicators(html, indicators) and check_html_length(html, **length_kw)
    else:
        return check_html_indicators(html, indicators) or check_html_length(html, **length_kw)


def page_not_available(html, domain=None, domain_only=False):
    """
    Check if the webpage is not available.

    Args:
        html (str): HTML content of the webpage.
        domain (str, optional): The domain of the webpage for domain-specific rules.
        domain_only (bool, optional): If True, only domain-specific checks will be performed.

    Returns:
        bool: True if the page is not available, False otherwise.
    """
    general_availability_indicators = ["this page doesn't exist or was removed", "page not found", 'the article you are looking for could not be found']
    domain_specific_indicators = domain_specific_page_not_available

    indicators = general_availability_indicators
    if domain and domain in domain_specific_indicators:
        if domain_only:
            indicators = domain_specific_indicators[domain]
        else:
            indicators.extend(domain_specific_indicators[domain])

    return check_html_indicators(html, indicators)

