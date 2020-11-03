import xml.etree.ElementTree as ET
import pandas as pd


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
    print(df)
    return df
