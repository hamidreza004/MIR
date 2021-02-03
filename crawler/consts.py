TOTAL_PAPERS_COUNT = 5000
BASE_URL = 'https://academic.microsoft.com/paper/'

PAGE_SELECTOR = "#mainArea router-view router-view div.results div.results ma-card .primary_paper"
TITLE_SELECTOR = "#mainArea h1.name"
ABSTRACT_SELECTOR = "#mainArea > router-view > div > div > div > div > p"
DATE_SELECTOR = "#mainArea > router-view > div > div > div > div > a > span.year"
AUTHORS_SELECTOR = "#mainArea > router-view > div > div > div > div > ma-author-string-collection > div > div.authors .author-item.au-target a.au-target.author.link"
REFERENCES_SELECTOR = "#mainArea > router-view > router-view div.results > div > compose > div > div.results > ma-card div.primary_paper > a.title.au-target"
