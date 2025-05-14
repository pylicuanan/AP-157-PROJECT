import cv2 as cv
import numpy as np

class coin_match():

    def __init__(self, haystack_file, heads_file, tails_file, filename,
                 flag = cv.IMREAD_UNCHANGED, method = cv.TM_CCOEFF_NORMED,
                 heads_threshold = 0.8, tails_threshold = 0.8):
        self.haystack_file = haystack_file
        self.heads_file = heads_file
        self.tails_file = tails_file
        self.filename = filename
        
        self.flag = flag
        self.method = method
        self.heads_threshold = heads_threshold
        self.tails_threshold = tails_threshold
        
        self.rect_heads = []
        self.rect_tails = []
        self.num_tails = 0
        self.num_heads = 0
        
        self._make_rectangles(self.filename)
        
    def match(self, needle_file, threshold):
        full_img = cv.imread(self.haystack_file, self.flag)
        look_img = cv.imread(needle_file, self.flag)

        look_w = look_img.shape[1]
        look_h = look_img.shape[0]

        result = cv.matchTemplate(full_img, look_img, self.method)

        if self.method == cv.TM_SQDIFF_NORMED or self.method == cv.TM_SQDIFF:
            locations = np.where(result<=threshold)
        else:
            locations = np.where(result>=threshold)
        locations = list(zip(*locations[::-1]))

        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), look_w, look_h]
            rectangles.append(rect)
            rectangles.append(rect)

        rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
        return rectangles

    def _make_rectangles(self, filename):
        self.rect_heads = self.match(self.heads_file, self.heads_threshold)
        self.rect_tails = self.match(self.tails_file, self.tails_threshold)
        
        full_img = cv.imread(self.haystack_file, self.flag)

        if len(self.rect_heads):    
            self.num_heads = len(self.rect_heads)
            
            line_color = (0,255,0)
            line_type = cv.LINE_4
            
            for (x, y, w, h) in self.rect_heads:
                top_left = (x, y)
                bottom_right = (x + w, y + h)

                cv.rectangle(full_img, top_left, bottom_right,
                            color = line_color, thickness=2, lineType=line_type)
        else:
            print("Heads not found")
            
        if len(self.rect_tails):    
            self.num_tails = len(self.rect_tails)
            line_color = (255,0,0)
            line_type = cv.LINE_4
            
            for (x, y, w, h) in self.rect_tails:
                top_left = (x, y)
                bottom_right = (x + w, y + h)

                cv.rectangle(full_img, top_left, bottom_right,
                            color = line_color, thickness=2, lineType=line_type)
        else:
            print("Tails not found")
        
        if len(self.rect_heads) or len(self.rect_tails):            
            cv.imwrite(filename, full_img)
            cv.imshow('Result',full_img)