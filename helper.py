import xml.etree.ElementTree as ET
import pandas as pd
import json


def XML_to_dataframe(filename):
    df = pd.DataFrame(columns=['description', 'title'])
    tree = ET.parse(filename)
    root = tree.getroot()
    i = 0
    for elem in root:
        if elem.tag.endswith("page"):
            description = ''
            title = ''
            for sub_elem in elem:
                if sub_elem.tag.endswith("title"):
                    title = sub_elem.text
                if sub_elem.tag.endswith("revision"):
                    for sub_sub_elem in sub_elem:
                        if sub_sub_elem.tag.endswith("text"):
                            description = sub_sub_elem.text
            df.loc[i] = [description, title]
            i += 1
    return df


def JSON_to_dataframe(filename):
    df = pd.DataFrame(columns=['title', 'summary', 'link', 'tags'])
    file = open(filename)
    data = json.load(file)
    pos = 0
    for row in data:
        df.loc[pos] = [row['title'] + row['summary'], row['summary'], row['link'], row['tags'][0]]
        pos += 1
    return df
