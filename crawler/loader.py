from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from consts import BASE_URL,\
    PAGE_SELECTOR, \
    TITLE_SELECTOR, \
    ABSTRACT_SELECTOR, \
    DATE_SELECTOR, \
    AUTHORS_SELECTOR, \
    REFERENCES_SELECTOR


class FaultyPageException(Exception):
    pass


def get_paper_url(paper_id):
    return BASE_URL + str(paper_id)


chrome_options = Options()
# chrome_options.add_argument("--headless")

driver = webdriver.Chrome(executable_path='./chromedriver', options=chrome_options)

number_of_tries = 0


def get_paper(paper_id):
    paper_url = get_paper_url(paper_id)
    driver.get(paper_url)

    global number_of_tries

    try:
        number_of_tries += 1

        _ = WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, PAGE_SELECTOR))
        )

    except TimeoutException:
        if number_of_tries <= 2:
            get_paper(paper_id)
        else:
            number_of_tries = 0
            raise FaultyPageException

    title = driver.find_element_by_css_selector(TITLE_SELECTOR).text
    abstract = driver.find_element_by_css_selector(ABSTRACT_SELECTOR).text
    date = driver.find_element_by_css_selector(DATE_SELECTOR).text

    authors = driver.find_elements_by_css_selector(AUTHORS_SELECTOR)
    authors = [author.text for author in authors]

    references = driver.find_elements_by_css_selector(REFERENCES_SELECTOR)
    references = [ref.get_attribute("href").split('/')[4] for ref in references]

    result = {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "date": date,
        "authors": authors,
        "references": references,
    }

    number_of_tries = 0

    return result
