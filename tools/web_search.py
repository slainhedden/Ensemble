import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from typing import List, Dict
import asyncio
from urllib.parse import quote_plus

class WebSearchSpider(scrapy.Spider):
    name = 'web_search_spider'
    
    def __init__(self, query: str, *args, **kwargs):
        super(WebSearchSpider, self).__init__(*args, **kwargs)
        self.start_urls = [f'https://www.google.com/search?q={quote_plus(query)}']
        self.results = []

    def parse(self, response):
        # Extract search result snippets
        for result in response.css('div.g'):
            title = result.css('h3.r a::text').get()
            snippet = result.css('div.s::text').get()
            if title and snippet:
                self.results.append({
                    'title': title,
                    'snippet': snippet
                })

        # Follow the "Next" page link if it exists
        next_page = response.css('div#foot table#nav td.b a::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

class WebSearch:
    def __init__(self):
        self.process = CrawlerProcess(get_project_settings())

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        spider = WebSearchSpider(query=query)
        self.process.crawl(spider)
        
        # Run the crawler in a separate thread
        await asyncio.get_event_loop().run_in_executor(None, self.process.start)
        
        # Return the results, limited to max_results
        print('Scrapping the web...')
        return spider.results[:max_results]
