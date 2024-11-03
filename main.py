import os
import re
import pymupdf


dialogue_files = []
directory = 'trainingdata/'

# Initializing PDF reader, looping through local directory, then going page-by-page
# through each pdf and scraping relevant dialogue

for file in os.listdir(directory):
    path = os.path.join(directory, file)
    doc = pymupdf.open(path)
    for page in doc:
        nix_pat = r"(?:RN|PRESIDENT):((?:\n(?!\n)|.)+?)(?=\n\n|\Z)"
        page_text = page.get_text()
        nixonspeech = re.findall(nix_pat, page_text)
        dialogue_files.append(nixonspeech)

print(dialogue_files)