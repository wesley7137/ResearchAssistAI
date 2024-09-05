import requests
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from rdflib import Graph, Namespace, RDF, URIRef, Literal
from rdflib.namespace import RDFS, OWL, XSD
import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from transformers import pipeline
import time

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLP models

# Rate limiting parameters
RATE_LIMIT = 2  # requests per second
RATE_LIMIT_PERIOD = 1  # second

# Phase 1: Data Ingestion and Search Functionality
def search_academic_sources(keywords, max_results=10):
    """
    Searches multiple academic sources (PubMed, arXiv, CrossRef) for articles based on keywords.
    
    :param keywords: List of keywords for the search query.
    :param max_results: Maximum number of articles to retrieve from each source per keyword.
    :return: List of articles with abstracts and metadata.
    """
    pubmed_articles = search_pubmed(keywords, max_results)
    time.sleep(RATE_LIMIT_PERIOD)  # Rate limiting between different API calls
    arxiv_articles = search_arxiv(keywords, max_results)
    time.sleep(RATE_LIMIT_PERIOD)  # Rate limiting between different API calls
    crossref_articles = search_crossref(keywords, max_results)
    
    all_articles = pubmed_articles + arxiv_articles + crossref_articles
    return all_articles


def search_pubmed(keywords, max_results=5):
    """
    Searches PubMed for articles based on keywords.

    :param keywords: List of keywords for the search query.
    :param max_results: Maximum number of articles to retrieve per keyword.
    :return: List of article abstracts and metadata.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    all_articles = []

    for keyword in keywords:
        params = {
            "db": "pubmed",
            "term": keyword,
            "retmax": max_results,
            "retmode": "json"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"PubMed search response for '{keyword}': {data}")
            article_ids = data.get('esearchresult', {}).get('idlist', [])
            
            if not article_ids:
                logger.warning(f"No PubMed articles found for the keyword: {keyword}")
                continue

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(article_ids),
                "retmode": "xml",
                "rettype": "abstract"
            }
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()

            soup = BeautifulSoup(fetch_response.content, "lxml-xml")
            for article in soup.find_all("PubmedArticle"):
                metadata = {
                    "title": article.find("ArticleTitle").text if article.find("ArticleTitle") else "N/A",
                    "abstract": " ".join([abstract.text for abstract in article.find_all("AbstractText")]),
                    "authors": [author.find("LastName").text for author in article.find_all("Author") if author.find("LastName")],
                    "journal": article.find("Title").text if article.find("Title") else "N/A",
                    "doi": article.find("ELocationID", {"EIdType": "doi"}).text if article.find("ELocationID", {"EIdType": "doi"}) else "N/A",
                    "source": "PubMed",
                    "keyword": keyword
                }
                all_articles.append(metadata)

            time.sleep(1 / RATE_LIMIT)  # Rate limiting

        except requests.RequestException as e:
            logger.error(f"Error fetching PubMed articles for keyword '{keyword}': {e}")

    logger.info(f"Total PubMed articles found: {len(all_articles)}")
    return all_articles

    
def search_arxiv(keywords, max_results=5):
    """
    Searches arXiv for articles based on keywords.

    :param keywords: List of keywords for the search query.
    :param max_results: Maximum number of articles to retrieve.
    :return: List of article abstracts and metadata.
    """
    base_url = "http://export.arxiv.org/api/query"
    query = "+AND+".join(keywords)
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        # Use 'lxml' parser explicitly
        soup = BeautifulSoup(response.content, "lxml-xml")
        articles = []

        for entry in soup.find_all("entry"):
            metadata = {
                "title": entry.find("title").text if entry.find("title") else "N/A",
                "abstract": entry.find("summary").text if entry.find("summary") else "N/A",
                "authors": [author.find("name").text for author in entry.find_all("author")],
                "journal": "arXiv",
                "doi": entry.find("id").text if entry.find("id") else "N/A",
                "source": "arXiv"
            }
            articles.append(metadata)

        time.sleep(1 / RATE_LIMIT)  # Rate limiting
        return articles

    except requests.RequestException as e:
        print(f"Error fetching arXiv articles: {e}")
        return []
    
    
def search_crossref(keywords, max_results=5):
    """
    Searches CrossRef for articles based on keywords.

    :param keywords: List of keywords for the search query.
    :param max_results: Maximum number of articles to retrieve.
    :return: List of article abstracts and metadata.
    """
    base_url = "https://api.crossref.org/works"
    params = {
        "query": " ".join(keywords),
        "rows": max_results
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get('message', {}).get('items', [])
        articles = []

        for item in items:
            metadata = {
                "title": item.get("title", ["N/A"])[0],
                "abstract": item.get("abstract", "N/A"),
                "authors": [author.get("family", "Unknown") for author in item.get("author", [])],
                "journal": item.get("container-title", ["N/A"])[0],
                "doi": item.get("DOI", "N/A"),
                "source": "CrossRef"
            }
            articles.append(metadata)

        time.sleep(1 / RATE_LIMIT)  # Rate limiting
        return articles

    except requests.RequestException as e:
        print(f"Error fetching CrossRef articles: {e}")
        return []
