import requests
from lxml import etree
from bs4 import BeautifulSoup

# Define the URL
url = 'http://127.0.0.1:5500/index.html'

# Make a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the HTML content from the response
    html_content = response.text

    # Parse the HTML content
    tree = etree.HTML(html_content)

    # Find the element using the provided XPath
    result = tree.xpath("/html/body/div/nav/")

    # Check if the element was found
    if result:
        print("Element found:", result)
    else:
        print("Element not found using the given XPath.")
else:
    print("Failed to retrieve the web page. Status code:", response.status_code)
