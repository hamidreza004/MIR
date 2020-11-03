import xml.etree.ElementTree as ET
import pandas as pd


def XML_to_dataframe(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    print(root[0].attrib)
