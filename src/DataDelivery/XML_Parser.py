#!/usr/bin/env python
__author__ = 'mackea1'

import operator
from functools import reduce
import xml.etree.ElementTree as ET
import pandas as pd
from collections import OrderedDict
import itertools

class Parser:
    def __init__(self):
        pass

    def removeLink(self, text:str):
        return text.split('}', 1)[-1]

    def findNodesByTag(self, parent: ET.Element, text: list):
        #print("Search Text: %s" % text)
        #print("Text Length: %s" % len(text))
        #print("Parent Type: %s" % type(parent))

        if any([t in self.removeLink(parent.tag).strip() for t in self.tags_to_avoid]):
            return None
        elif isinstance(text, str):
            return [child for child in parent.iter() if text in child.tag]
        elif len(text) == 1:
            return [child for child in parent.iter() if text[0] in child.tag]
        else:
            vals = [self.findNodesByTag(child, text[1:]) for child in [c for c in parent.iter()] if text[0] in child.tag]
            vals = list(filter(lambda a: a is not None, vals))
            if vals:
                return reduce(operator.add, vals)
                # return [findNodesByTag(child, text[1:]) for child in [c for c in parent.iter()] if text[0] in child.tag]
            else:
                return None

    def convertNodeToList(self, parent: ET.Element, vals: OrderedDict, parent_tag: str=None):
        category_nodes = []
        for child in [c for c in parent if all([t not in self.removeLink(c.tag).strip() for t in self.tags_to_skip])]:
            # print("Tag: %s  ||  Text: %s" % (child.tag, child.text))
            tag = self.removeLink(child.tag).strip() or ''
            text = (child.text or '').strip()
            if len(parent.attrib) == 0:
                if (tag != '') and ((text != '') or len(child) == 0) and ((not self.specific_tags) or tag in self.specific_tags):  # If the children have text, add them
                    # print("Value: " + child.text)
                    vals[parent_tag + '_' if parent_tag else '' + tag] = text
                elif len(child) > 0 and not self.skip_all_child_nodes:  # If the children don't have text, dig into them and see if their grandchildren will have text
                    category_nodes.append(child)
            else:
                if child.attrib['type'] == 'Property' and (tag != '') and ((not self.specific_tags) or tag in self.specific_tags):
                    vals[parent_tag + '_' if parent_tag else '' + tag] = text
                elif child.attrib['type'] == 'Category' and not self.skip_all_child_nodes:
                    # print("Found a Category!!! %s" % child)
                    category_nodes.append(child)

        if len(category_nodes) > 0:
            # print("Children: ", category_nodes)
            # print("Vals: ", vals)
            retlist = list()
            parent_tag = self.removeLink(parent.tag).strip() or ''
            [[retlist.append(n.copy()) for n in self.convertNodeToList(c, vals.copy())] for c in category_nodes]
            # print("RetList:\n", retlist)
            return retlist
        else:
            # print("Vals To Return: ", vals)
            return [vals]


    def xmlToCSV(self, file_name: str, root: ET.Element, tags_to_traverse: list, tags_to_avoid: list=list(), tags_to_skip: list=list(),
                specific_tags: list=list(), skip_all_child_nodes: bool=False):
        self.tags_to_avoid = tags_to_avoid
        self.tags_to_skip = tags_to_skip
        self.specific_tags = specific_tags
        self.skip_all_child_nodes = skip_all_child_nodes
        nodes = self.findNodesByTag(root, tags_to_traverse)[0]
        #print("Nodes: ", nodes)
        nodeList = self.convertNodeToList(nodes, vals=OrderedDict([]))
        #print("NodeList: ", nodeList)
        df = pd.DataFrame.from_dict(nodeList)
        #print("DataFrame:\n", df)
        file_name += ('.csv' if '.csv' not in file_name else '')
        df.to_csv(file_name, index=False)
        return file_name

    def xmlToDF(self, root: ET.Element, tags_to_traverse: list, tags_to_avoid: list=list(), tags_to_skip: list=list(),
                specific_tags: list=list(), skip_all_child_nodes: bool=False):
        self.tags_to_avoid = tags_to_avoid
        self.tags_to_skip = tags_to_skip
        self.specific_tags = specific_tags
        self.skip_all_child_nodes = skip_all_child_nodes
        try:
            node_list = list()
            nodes = self.findNodesByTag(root, tags_to_traverse)
            if not nodes is None:
                for node in nodes:
                    node_list.append(self.convertNodeToList(node, vals=OrderedDict([])))
                node_list = list(itertools.chain(*node_list))
                # print('Node Results: ')
                # pprint.pprint(nodeList[0], indent=3)
                #print("NodeList: ", nodeList)
                return pd.DataFrame.from_records(node_list)
            else:
                raise ET.ParseError("Darn! No nodes found matching the search criteria!")
        except IndexError as e:
            return None
