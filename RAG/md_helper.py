import re

def strip_markdown(text):
    # Remove headings, bold, italics, links, images, inline code, blockquotes, lists
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # remove images
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # remove links
    text = re.sub(r'[`*_>#-]', '', text)        # remove MD symbols
    return text.strip()
