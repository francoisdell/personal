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

        # print("Tag: %s  ||  Text: %s" % (parent.tag, parent.text))
        # if ('FILEMETADATA' in parent.tag.upper() or 'FILEMETADATA' in parent.text.upper()) and self.tags_to_avoid != []:
        #     print("Tag: %s  ||  Text: %s" % (parent.tag, parent.text))
        #     lol = 5

        if any([t in self.removeLink(parent.tag).strip() for t in self.tags_to_avoid]):
            return None
        elif isinstance(text, str):
            # for child in parent.iter():
            #     if text in child.tag:
            #         lol = (text[0], child.tag)
            #         derp=1
            # print("Line 36: %s" % [child.tag for child in parent.iter() if text[0] in child.tag])
            return [child for child in parent.iter() if text in child.tag]
        elif len(text) == 1:
            # for child in parent.iter():
            #     if text[0] in child.tag:
            #         lol = (text[0], child.tag)
            #         derp=1
            # print("One entry. Text: %s" % text)
            # print([child for child in parent.iter()])
            # if self.tags_to_avoid != []:
            #     print("Line 45: %s" % [child.tag for child in parent.iter() if text[0] in child.tag])
            return [child for child in parent.iter() if text[0] in child.tag]
        else:
            # if 'FILEMETADATA' in parent.tag.upper() or 'FILEMETADATA' in parent.text.upper():
            #     print("Tag: %s  ||  Text: %s" % (parent.tag, parent.text))
            #     lol = 5
            # print("Multiple entries. Text: %s" % text)
            vals = [self.findNodesByTag(child, text[1:]) for child in [c for c in parent.iter()] if text[0] in child.tag]
            vals = list(filter(lambda a: a is not None, vals))
            if vals:
                # print("Values:\n%s" % vals)
                # print("\nParent's Children:\n%s" % parent.iter())
                # print("Line 56: %s" % [c.tag for c in vals])
                return reduce(operator.add, vals)
                # return [findNodesByTag(child, text[1:]) for child in [c for c in parent.iter()] if text[0] in child.tag]
            else:
                return None

    def convertNodeToList(self, parent: ET.Element, vals: OrderedDict, parent_tag: str=None):
        category_nodes = []
        # if ((parent.tag and 'FILEMETADATA' in parent.tag.upper())
        #         or (parent.text and 'FILEMETADATA' in parent.text.upper())) \
        #         and self.tags_to_avoid != []:
        #     print("Parent  ||  Tag: %s  ||  Text: %s" % (parent.tag, parent.text))
        #     derp = 5
        # print('Tags to skip: ', self.tags_to_skip)
        for child in [c for c in parent if all([t not in self.removeLink(c.tag).strip() for t in self.tags_to_skip])]:
            # print("Tag: %s  ||  Text: %s" % (child.tag, child.text))
            tag = self.removeLink(child.tag).strip() or ''
            text = (child.text or '').strip()
            # if parent.tag == 'rdf_dir':
            #     print("WHAT THE FUCK IS GOING ON!??!?!")
            # if ((child.tag and 'FILEMETADATA' in child.tag.upper())
            #         or (child.text and 'FILEMETADATA' in child.text.upper())) \
            #         and self.tags_to_avoid != []:
            #     print("Child  ||  Tag: %s  ||  Text: %s" % (child.tag, child.text))
            #     lol = 5
            if len(parent.attrib) == 0:
                if (tag != '') and ((text != '') or len(child) == 0) and ((not self.specific_tags) or tag in self.specific_tags):  # If the children have text, add them
                    # print("Value: " + child.text)
                    vals[parent_tag + '_' if parent_tag else '' + tag] = text
                elif len(child) > 0 and not self.skip_all_child_nodes:  # If the children don't have text, dig into them and see if their grandchildren will have text
                    category_nodes.append(child)
            else:
                # if parent.attrib['type'] == 'Category' and parent.tag == 'FileMetaData':
                #     lol=5
                # if child.attrib['type'] == 'Property':
                #     lol = 5
                # if self.tags_to_avoid != []:
                #     print('Type: %s  ||  tag: %s  ||  Text: %s' % (child.attrib['type'], tag, text))
                if child.attrib['type'] == 'Property' and (tag != '') and ((not self.specific_tags) or tag in self.specific_tags):
                    # if text == 'arrayconfig_CS' and self.tags_to_avoid != []:
                    #     lol = 5
                    vals[parent_tag + '_' if parent_tag else '' + tag] = text
                elif child.attrib['type'] == 'Category' and not self.skip_all_child_nodes:
                    # print("Found a Category!!! %s" % child)
                    category_nodes.append(child)

        # if parent.tag == 'rdf_dir':
        #     print("WHAT THE FUCK IS GOING ON!??!?!")

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


if __name__ == '__main__':

    path = '20160302_034758_CK2EN153800002_EMC-UEM-Telemetry.xml' # for testing

    if not isinstance(path, (list, tuple)):
        path = [path]

    for i in path:
        print(i)
        if i != '.' and i.find('xml') > 0:

            tree = ET.parse(i)
            root = tree.getroot()

            import csv
            with open(i.replace('xml','csv'), 'wt', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['Title', 'Symmetrix_SN', 'Date_time', 'Version', 'idx', 'SName', 'WPData', 'Type',
                    'BusyData', 'P0Data', 'P1Data', 'P0IOPS', 'P1IOPS'])

            print("\nChildren of : root")
            for child in root:
                print(child.tag, child.attrib)

            p = Parser()

            searchstr = ['storage_symmetrix','serial']
            print('\nSearch for %s' % searchstr)
            symmSN = p.findNodesByTag(root, searchstr)
            print("Element: %s\n=======Children=======" % symmSN)
            for child in symmSN:
                print(child.tag, child.attrib, child.text)
            print("=======End Children=======")

            # -------------------------------------------------------------------------------------------
            # COMPLETED FIELDS BELOW THIS POINT
            # -------------------------------------------------------------------------------------------
            print("\nFINAL VALUES")

            FSSizeMB = sum([float(child.text) for child in p.findNodesByTag(root, 'FSSizeMB')])
            print('Total File System Capacity (TB): %s' % (FSSizeMB/1024/1024))

            creationDate = p.findNodesByTag(root, ['UEM_File_Data','FileMetaData','CreationDate'])[0].text
            print('Config File Creation Date: %s' % (creationDate))

            symmSN = p.findNodesByTag(root, ['storage_symmetrix','serial'])[0].text
            print("Symm SN: %s" % symmSN)

            qtyDM = len(p.findNodesByTag(root, ['server_inventory','DataMoverResume']))
            print("Data Movers: %s" % qtyDM)

            qtyFS = len(p.findNodesByTag(root, ['UEM_File_Data','FileSystem_General','FileSystemGeneral']))
            print("File Systems: %s" % qtyFS)

            qtyFSGrp = len(p.findNodesByTag(root, ['UEM_File_Data','FileSystem_Group','FileSystemGroup']))
            print("File System Groups: %s" % qtyFSGrp)

            qtyServer = len(p.findNodesByTag(root, ['ServerData','server']))
            print("Qty Servers: %s" % qtyServer)

            #xmlToCSV('server_data.csv', ['UEM_File_Data','ServerData','server'])
            #xmlToCSV('mount_data.csv', ['UEM_File_Data','MountData','serverMount','mount'])
            #xmlToCSV('filesystem_data', ['UEM_File_Data','FileSystem_General','FileSystemGeneral'])

            p.xmlToCSV('nic_data.csv', ['UEM_File_Data','NICData'])
            p.xmlToCSV('server_data.csv', ['UEM_File_Data','ServerData'])
            p.xmlToCSV('mount_data.csv', ['UEM_File_Data','MountData'])
            p.xmlToCSV('filesystemgeneral_data', ['UEM_File_Data','FileSystem_General'])
            p.xmlToCSV('filesystemcapacity_data', ['UEM_File_Data','FileSystem_Capacity'])
            p.xmlToCSV('storagepools_data', ['UEM_File_Data','StoragePools'])
            p.xmlToCSV('volumes_data', ['UEM_File_Data','Volumes'])
            p.xmlToCSV('filesystemgroup_data', ['UEM_File_Data','FileSystem_Group'])
            p.xmlToCSV('server_inventory_data', ['UEM_File_Data','server_inventory'])
            p.xmlToCSV('servermount_data', ['UEM_File_Data','MountData'])