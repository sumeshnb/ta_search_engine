from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor

from urlparse import urlparse

import uuid
import os


class UTACrawler(CrawlSpider):
    # Name of the the crawler to identify to UTA Website
    name = "cse5334crawler"

    #######################I have given two sets of variables below
    #   Use the cse one for testing your code.
    #   It is relatively small (cse.uta.edu has only around 200 pages and crawls in few minutes
    #   Once your code is stable, run it on entire uta by uncommenting the UTA variables

    #   Scrapy based crawlers are quite aggressive - so make sure that your code is bug-free
    #       before unleashing it on uta.edu or cse.uta.edu
    #   There are many simple ways to test your code - function by function

    #Domain that we will restrict the crawler to
    allowed_domains = ["cse.uta.edu"]
    #Starting URL for crawling
    start_urls = ["http://cse.uta.edu/"]

    #Domain that we will restrict the crawler to
    #allowed_domains = ["uta.edu"]
    #Starting URL for crawling
    #start_urls = ["http://www.uta.edu/uta"]



    #Do not change this line
    # It determines what to do when you get a page
    #   We use SgmlLinkExtractor to get all <a> tags from allowed_domains
    #   and then call function filter_links to do further post processing
    #   once the web page is filtered and downloaded, we call parse_item to process it
    rules = ( Rule(
        SgmlLinkExtractor(
            allow_domains=tuple(allowed_domains),
            unique=True),
        callback='parse_item',
        process_links="filter_links",
        follow=True
    ),
    )



    # This functions accepts an array of Link objects and returns another array of Link objects
    #   you can use this function to ignore certain urls from processing
    # The Link object will look like the following
    #   Link(url='http://cse.uta.edu/', text=u'Skip to content', fragment='', nofollow=False)          
    #   url: gives the url to crawl
    #   text: is the anchor text associated with the url
    #   ignore the fragment and nofollow
    def filter_links(self, links):
        #filtered_links is the output you are going to return.
        filtered_links = []

        #Uta.edu has close to 200,000 web pages. We dont want to download many of them
        # So we will use the following set of keywords to ignore some of them
        # If the url has some word from the following list, ignore it
        words_to_ignore = ["calendar", "timebegin", "book.php", "archive", "catalog", "eventid",
                           "maverickdiscounts", "devel.uta.edu", "library.uta.edu", "mymav.uta.edu",
                           "/m.", "archives"]
        #Uta.edu has many types of files: html, php, pdf, docx etc
        #  For our purpose, we want to use the webpages with the following url only
        valid_extensions = [".htm", ".html", ".php"]

        for link in links:

            #Process each link object
            # ignore the link if the url has some word in words_to_ignore
            #   hint: do a simple string search, dont complicate it by using regular expressions
            #####################Task t1a: your code below#######################				
            #####################Task t1a: your code below#######################

            #print link.url
            flag = 0
            for word in words_to_ignore:
                if word in link.url:
                    flag = 1
                    #print word
                    break

            if flag == 0:
                parsedUrl = urlparse(link.url)
                #print parsedUrl

                #We now want to consider only htm, html and php files
                # naively searching for it in the url is a bad idea
                # here is a better idea:
                #   the urlparse function parses url into 6 variables
                #   specifically look at the path variable
                #   use it to determine if the url has one of the extensions from valid_extensions
                # if so, append the link variable (NOT THE URL) to filtered_links
                #   hint: use endswith function
                #####################Task t1b: your code below#######################
                #####################Task t1b: your code below#######################
                for ext in valid_extensions:
                    if parsedUrl.path.endswith(str(ext)):
                        filtered_links.append(link)
                        break

        return filtered_links


    def get_unique_file_name(self):
        #This function is called to get a unique filename
        # Do the following:
        #  1. Generate a string using uuid module (specifically use uuid4) . say abcd
        #  2. The filename will be downloads/abcd.html
        #  3. Check if some file with this name already exists
        #  4. If so, go to step 1
        #  5. If not, return the file name (both the directory name and the html extension
        #####################Task t1c: your code below#######################
        #####################Task t1c: your code below#######################
        file_name = os.getcwd() + "/downloads/" + str(uuid.uuid4()) + ".html"
        #print file_name

        while os.path.isfile(file_name):
            file_name = os.getcwd() + "/downloads/" + str(uuid.uuid4()) + ".html"

        return file_name

    #Do not change anything in this function
    def write_to_file(self, file_name, response):
        open(file_name, 'wb').write(response.body)


        #Do not change anything in this function

    def parse_item(self, response):
        file_name = self.get_unique_file_name()
        f = open("fileNamesToUUID.txt", "a")
        f.write(file_name + "|" + response.url + "\n")
        f.flush()
        f.close()

        self.write_to_file(file_name, response)                              
