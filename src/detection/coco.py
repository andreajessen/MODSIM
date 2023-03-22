from collections import defaultdict
import itertools
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class COCO():
    def __init__(self, datasetPath):
        f = open(datasetPath)
        self.dataset = json.load(f)
        self.createIndex()
        self.colors = plt.cm.get_cmap('hsv', len(self.cats.keys()))

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset.keys():
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset.keys():
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset.keys():
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset.keys() and 'categories' in self.dataset.keys():
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
    
    def _isArrayLike(self, obj):
            return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
    
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
    
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
                catIds  (int array)     : get anns for given cats
                areaRng (float array)   : get anns for given area range (e.g. [0 inf])
                iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
    
        imgIds = imgIds if self._isArrayLike(imgIds) else [imgIds]
        catIds = catIds if self._isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids
    
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if self._isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]
    
    def visualizeImage(self, image_id, images_path):
        image_name = str(image_id).zfill(8)+".jpeg" # Image names are 12 characters long
        image = Image.open(images_path+image_name)
        
        fig, ax = plt.subplots()

        anns = self.loadAnns(self.getAnnIds(image_id))
        # Draw boxes and add label to each box
        for ann in anns:
            box = ann['bbox']
            bb = patches.Rectangle((box[0],box[1]), box[2],box[3], linewidth=1, edgecolor=self.colors(ann['category_id']), facecolor="none")
            ax.add_patch(bb)
            ax.text(box[0], box[1], ann['category_id'], color=self.colors(ann['category_id']))
        
        ax.imshow(image)
        plt.show()