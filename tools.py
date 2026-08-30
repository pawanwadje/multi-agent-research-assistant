from langchain.tools import tool 
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query:str)-> str:
    """ Search the web for recent and reliable information on a topic. Returns: A formatted list containing titles, URLs, and snippets. """
    try:
        results=tavily.search(query=query,max_results=5)

        # return results
        out =[]
        for r in results['results']:
            out.append(
                f"Title:{r['title']}\nURL:{r['url']}\nSnippet:{r['content'][:500]}\n"
                
            )
        # return "\n-----\n".join(out)
        if not out: 
            return "No relevant search results were found." 
        return "\n-----\n".join(out) 
    except Exception as e: 
        return f"Web search failed: {str(e)}"
# print(web_search.invoke("what are the recent news of war"))

@tool
def scrape_url(url:str)->str:
    """Scrape and return clean text content from a given """
    try:
        resp=requests.get(url,timeout=8,headers={"User-Agent":"Mozilla/5.0"})
        soup=BeautifulSoup(resp.text,"lxml")
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        return soup.get_text(separator=" ",strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL:{str(e)}"


# print( scrape_url.invoke( "https://en.wikipedia.org/wiki/Artificial_intelligence" ) )

