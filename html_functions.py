from bs4 import BeautifulSoup


domain_specific_full_text_select_mixed_functions = {
    'marketwatch.com': ('div', 'article__content'),
    'business-standard.com': lambda x: BeautifulSoup(x, 'html.parser').select('div.MainStory_storycontent__Pe3ys > div:not([id]):not([class])'),
    'benzinga.com': lambda x: (find_tag_with_most_tags(x, tags=('p',), recursive=True).find_all('p'), find_tag_with_most_tags(x, tags=('ul',), recursive=False).find_all('ul')),
    'cnn.com': lambda x: BeautifulSoup(x, 'html.parser').select('div.article__content p'),
}


domain_specific_captcha_encountered = {
    'investors.com': ['PerimeterX'],
}

domain_specific_page_not_available = {}

delay_times = {
    'marketwatch.com': dict(base_delay=10, random_factor=30)
}


def find_tag_with_most_tags(html=None, parser=None, tags=(), recursive=False, return_any=True):
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

    def count_p_tags(tag):
        """ Counts <p> tags within the given tag, including nested ones. """
        return len(tag.find_all(*tags))

    # Find the tag with the maximum number of <p> tags
    max_p_count = 0
    tag_with_most_ps = None
    final_tag = None
    if not recursive:
        # Iterate over all tags in the soup
        for tag in soup.find_all():
            # Count direct <p> children
            p_count = len(tag.find_all(*tags, recursive=False))
            # Update if this tag has more <p> tags than any previous ones
            if p_count > max_p_count:
                max_p_count = p_count
                tag_with_most_ps = tag
        return tag_with_most_ps

    else:

        # Iterate over all tags in the current level
        for tag in soup.find_all(recursive=False):
            p_count = count_p_tags(tag)
            if p_count > max_p_count:
                max_p_count = p_count
                tag_with_most_ps = tag
                tag_with_most_ps_global = tag

        # Check within the tag found for a tag that has more than half its <p>s as direct descendants
        while final_tag is None and max_p_count > 0:
            max_p_count_local = -1
            final_tag = None
            for child_tag in tag_with_most_ps.find_all(recursive=False):
                total_p = count_p_tags(child_tag)
                direct_p = len(child_tag.find_all(*tags, recursive=False))
                if total_p > max_p_count_local:
                    tag_with_most_ps = child_tag
                if direct_p > total_p / 2 and direct_p > max_p_count / 3:
                    tag_with_most_ps = child_tag
                    final_tag = child_tag
                    break
                max_p_count_local = max(max_p_count_local, total_p)

        if final_tag:
            print("Found the desired tag:")
            print("HTML of the tag:")
        else:
            print("No such tag found.")

        if return_any:
            return tag_with_most_ps
        return final_tag


def check_html_content(html, indicators, min_html_length=2000):
    """
    Base function to check HTML content against specified indicators and a minimum length.

    Args:
        html (str): HTML content of the webpage.
        indicators (list): List of strings that indicate an issue when found in HTML.
        min_html_length (int): Minimum acceptable length of HTML content to consider the page valid.

    Returns:
        bool: True if any indicator is found or if the content is shorter than the minimum length, False otherwise.

    Raises:
        ValueError: If the html or indicators are not properly provided.
    """
    if not html or not isinstance(html, str):
        raise ValueError("Invalid HTML content provided.")
    if not isinstance(indicators, list) or not all(isinstance(i, str) for i in indicators):
        raise ValueError("Indicators must be a list of strings.")
    if min_html_length is not None and not isinstance(min_html_length, int):
        if len(html) < min_html_length:
            return True  # Content is too short, potentially indicating an issue
    return any(indicator in html for indicator in indicators)


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

    return check_html_content(html, indicators)


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
    general_availability_indicators = ["This page doesn't exist or was removed", "Page not found"]
    domain_specific_indicators = domain_specific_page_not_available

    indicators = general_availability_indicators
    if domain and domain in domain_specific_indicators:
        if domain_only:
            indicators = domain_specific_indicators[domain]
        else:
            indicators.extend(domain_specific_indicators[domain])

    return check_html_content(html, indicators, min_html_length=None)

