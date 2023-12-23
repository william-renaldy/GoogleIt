"""
GoogleIt Converter Module

This module provides functionality to convert HTML files or websites into PDF format using Selenium.

Usage:
    - Import the module: `from GoogleIt import converter`
    - Call the `convert` function with appropriate parameters.

Example:
    ```python
    converter.convert(source='https://example.com', target='output.pdf', timeout=5)
    ```

Functions:
    - `convert(source: str, target: str, timeout: int = 2, print_options: dict = {}) -> None`:
        Converts a given HTML file or website into PDF.

        Parameters:
            - `source` (str): Source HTML file or website link.
            - `target` (str): Target location to save the PDF.
            - `timeout` (int, optional): Timeout in seconds. Default is set to 2 seconds.
            - `print_options` (dict, optional): Options for PDF printing. Refer to https://vanilla.aslushnikov.com/?Page.printToPDF for available options.

        Raises:
            - Exception: If an error occurs during PDF conversion.

Note:
    This module relies on the Selenium library and requires a compatible WebDriver (e.g., ChromeDriver) to be installed.

"""


import json
import base64

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import staleness_of
from selenium.webdriver.common.by import By


def convert(
    source: str,
    target: str,
    timeout: int = 2,
    print_options: dict = {},
):
    """
    Convert a given html file or website into PDF

    :param str source: source html file or website link
    :param str target: target location to save the PDF
    :param int timeout: timeout in seconds. Default value is set to 2 seconds
    :param bool compress: whether PDF is compressed or not. Default value is False
    :param int power: power of the compression. Default value is 0. This can be 0: default, 1: prepress, 2: printer, 3: ebook, 4: screen
    :param dict print_options: options for the printing of the PDF. This can be any of the params in here:https://vanilla.aslushnikov.com/?Page.printToPDF
    """

    result = __get_pdf_from_html(
        source, timeout, print_options)


    with open(target, "wb") as file:
        file.write(result)



def __send_devtools(driver, cmd, params={}):
    resource = "/session/%s/chromium/send_command_and_get_result" % driver.session_id
    url = driver.command_executor._url + resource
    body = json.dumps({"cmd": cmd, "params": params})
    response = driver.command_executor._request("POST", url, body)

    if not response:
        raise Exception(response.get("value"))

    return response.get("value")


def __get_pdf_from_html(
    path: str, timeout: int, print_options: dict
):
    webdriver_options = Options()
    webdriver_prefs = {}
    driver = None

    webdriver_options.add_argument("--headless")
    webdriver_options.add_argument("--disable-gpu")
    webdriver_options.add_argument("--no-sandbox")
    webdriver_options.add_argument("--disable-dev-shm-usage")
    webdriver_options.experimental_options["prefs"] = webdriver_prefs

    webdriver_prefs["profile.default_content_settings"] = {"images": 2}

    driver = webdriver.Chrome(options=webdriver_options)

    driver.get(path)

    try:
        element = driver.find_element(by=By.TAG_NAME, value="body")
        element.send_keys('Ctrl+A')
        WebDriverWait(driver, timeout).until(
            staleness_of(element)
        )
    except TimeoutException:
        calculated_print_options = {
            "landscape": False,
            "displayHeaderFooter": False,
            "printBackground": True,
            "preferCSSPageSize": True,
        }
        calculated_print_options.update(print_options)
        result = __send_devtools(
            driver, "Page.printToPDF", calculated_print_options)
        driver.quit()
        return base64.b64decode(result["data"])