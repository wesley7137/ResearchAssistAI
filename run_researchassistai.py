from Phase1 import search_academic_sources
from Phase2 import extract_key_information, summarize_text, generate_description
from Phase3 import harmonize_data, ontology_mapping
from Phase4 import preprocess_image, extract_data_from_image
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import datetime


def main():
    keywords = ["longevity", "mitochondrial", "aging", "protein folding", "autophagy", "bio multi-modal datasets", "machine learning"]
    max_results = 5  # or whatever number you want

    articles = search_academic_sources(keywords, max_results)

    if not articles:
        print("No articles found.")
        return


    # Display the articles retrieved for debugging and validation
    print(f"Retrieved {len(articles)} articles from multiple sources.")
    for i, article in enumerate(articles, 1):
        print(f"Article {i}:")
        print(f"Title: {article['title']}")
        print(f"Abstract: {article['abstract'][:200]}...")  # Print a snippet of the abstract
        print(f"Authors: {', '.join(article['authors'])}")
        print(f"Journal: {article['journal']}")
        print(f"DOI: {article['doi']}")
        print(f"Source: {article['source']}\n")

    # Phase 2: NLP for Information Extraction and Summarization
    print("\n\n\n**Phase 2: NLP for Information Extraction and Summarization**\n\n\n")
    summarized_articles = []

    for index, article in enumerate(articles, start=1):
        # Extract key information
        key_info = extract_key_information(article['abstract'])
        # Summarize the abstract
        summary = summarize_text(article['abstract'])
        # Generate descriptions using the extracted key information
        generated_desc = generate_description(key_info['METHODS'], key_info['RESULTS'])
        
        # Create a citable format
        citable_info = {
            'authors': article.get('authors', []),
            'year': article.get('year', ''),
            'title': article['title'],
            'journal': article.get('journal', 'N/A'),
            'volume': article.get('volume', 'N/A'),
            'issue': article.get('issue', 'N/A'),
            'pages': article.get('pages', 'N/A'),
            'doi': article.get('doi', 'N/A'),
            'url': article.get('url', 'N/A'),
            'publication_date': article.get('publication_date', 'N/A'),
            'accessed_date': datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        summarized_article = {
            'title': article['title'],
            'abstract': article['abstract'],
            'summary': summary,
            'extracted_info': key_info,
            'generated_description': generated_desc,
            'citation_info': citable_info,
            'source': article.get('source', 'N/A')
        }
        
        summarized_articles.append(summarized_article)

        # Display summarized information for debugging
        print(f"Summary for '{article['title']}':")
        print(f"Abstract Summary: {summary}")
        print(f"Generated Description: {generated_desc}")
        print(f"Citation Info: {citable_info}\n")

        # Save after every 5 articles or at the end
        if index % 5 == 0 or index == len(articles):
            # Ensure the directory exists
            os.makedirs("processed_articles", exist_ok=True)
            
            # Save to a file using the index instead of 'id'
            with open(f"processed_articles/articles_{index-4}_to_{index}.json", 'w') as f:
                json.dump(summarized_articles, f, indent=2)
            
            print(f"Completed processing batch of articles (up to article {index})")
            
            # Clear the list for the next batch
            summarized_articles = []



    # Phase 3: Data Integration
    print("\n\n\n**Phase 3: Data Integration**\n\n\n")
    # Convert article data to pandas DataFrame format
    df = pd.DataFrame(articles)
    harmonized_df = harmonize_data([df])
    rdf_graph = ontology_mapping(harmonized_df)

    # Display harmonized data and RDF graph snippet
    print("Harmonized Data Frame:")
    print(harmonized_df.head())
    print("\nRDF Graph (Turtle format):")
    print(rdf_graph.serialize(format="turtle").decode("utf-8"))

    # Phase 4: Image and Multimedia Analysis
    print("\n\n\n**Phase 4: Image and Multimedia Analysis**\n\n\n")
    image_path = "example_chart.png"  # Replace with actual path to your image
    preprocessed_image = preprocess_image(image_path)

    # Display preprocessed image for validation
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title("Preprocessed Image")
    plt.show()

    # Extract text and data from the image
    extracted_text = extract_data_from_image(preprocessed_image)
    print("Extracted Text Data from Image:")
    print(extracted_text)

    # Phase 5: Natural Language Generation
    print("\n\n\n**Phase 5: Natural Language Generation**\n\n\n")
    # Using the descriptions generated in Phase 2 as the final output for simplicity
    for article in summarized_articles:
        print(f"Generated Description for '{article['title']}':")
        print(article['generated_description'])
        print("\n")

    print("Pipeline execution completed.")

if __name__ == "__main__":
    main()