import os
import re
import pymupdf
import ast

dialogue_files = []
directory = 'trainingdata/'

# Initializing PDF reader, looping through local directory, then going page-by-page
# through each pdf and scraping relevant dialogue

for file in os.listdir(directory):
    path = os.path.join(directory, file)
    doc = pymupdf.open(path)
    for page in doc:
        nix_pat = r"PRESIDENT:\s*((?:[^\n]+(?:\n(?![A-Z]+:))?)*)"
        page_text = page.get_text()
        nixonspeech = re.findall(nix_pat, page_text)
        dialogue_files.append(nixonspeech)


clean = []

# print(dialogue_files)
# print(len(dialogue_files))


for page in dialogue_files:
    page = str(page)
    page1 = page.replace(r"\\", " ")
    page2 = page1.replace(r"\n", " ")
    clean.append(page2.strip())
print(clean)