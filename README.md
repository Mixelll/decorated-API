**Consists of the following modules:**
- postgresql_db - database queries and utilities built with psycopg3 and SQLAlchemy.
- m_db - generated ETL decorators.
- news_scrapers - news scraper class with fake_useragent and fallback to selenium.
- alpha_vantage_news - get urls and sentiment from Alpha Vantage API, compared against DB, scrape full articles and load to table. Uses decorators defined in m_db.

The decorators usually act as a shell with the core logic implemented by functions. If the decorators are not to your liking, use the functions directly.

**Requires the decortools library**
