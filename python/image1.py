from PIL import Image, ImageDraw, ImageOps
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import cv2
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

IMAGE_FILE = "hushih.jpg"
IMAGE_SIZE = (300, 300)
POINT_NUMBER = 200
LINE_PIXEL = {}
NOT_ALLOWED = set()
USED_LINES = set()
MAX_ERROR = 1e3
Y, X = np.ogrid[:IMAGE_SIZE[1], :IMAGE_SIZE[0]]
center = (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2)
radius = IMAGE_SIZE[0] // 2
dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
MASK = dist_from_center <= radius
drawed = np.zeros(IMAGE_SIZE)

im = Image.open(IMAGE_FILE)

################### resize, grayscale

new_image = im.convert(mode="L")
width, height = new_image.size

if width>height:
    top = 0
    bottom = height
    left = (width-height)//2
    right = left+height
else:
    left = 0
    right = width
    top = (height-width)//2
    bottom = top+width

new_image = new_image.crop((left,top,right,bottom))
new_image = new_image.resize(IMAGE_SIZE)

################### resize, grayscale


################################################################## specifying boudaries

# img_np = np.array(new_image)
# blurred = cv2.GaussianBlur(img_np, (5, 5), 1.4)

# # Apply Canny edge detection
# edges = cv2.Canny(blurred, threshold1=30, threshold2=30)

# pixel_value = edges.astype(np.float32)

# Y, X = np.ogrid[:IMAGE_SIZE[1], :IMAGE_SIZE[0]]
# dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
# fade_mask = 1 - (dist_from_center / radius)**2
# fade_mask = np.clip(fade_mask, 0, 1)

# Apply to edge map
# pixel_value *= fade_mask

# # Convert back to PIL
# new_image = Image.fromarray(img_clahe)

################################################################## specifying boudaries



############################################################# converting to inverse pixel value

# Apply MASK: set values outside circle to 0
pixel_value = np.array(new_image)
pixel_value = 255-pixel_value
pixel_value[~MASK] = 0

# plt.imshow(pixel_value, cmap='gray')
# plt.colorbar()
# plt.show()

############################################################# converting to inverse pixel value

def merge_pixel(pixel1, pixel2):
    return np.maximum(pixel1,pixel2)

def cal_error(pixel1, target_pixel,drawed):
    # pixel2 is the target
    missing = (target_pixel - pixel1).clip(min=0)
    extra = (pixel1 - target_pixel).clip(min=0)
    if drawed is not None:
        overlap_penalty = (drawed > 200).astype(float)
        extra *= (1 + overlap_penalty * 2)  # boost penalty for redundant drawing
    return np.mean(4*missing+0.5*extra)


def threshold(distance):
    # return 20 * math.exp(-distance**2 / (2 * 0.8**2))  # Smooth falloff
    if distance<=0.1:
        return 30*5
    elif distance<=0.6:
        return (36-60*distance)*5
    else:
        return 0 
    

def threshold_vec(dist):
    # dist is a NumPy array
    result = np.zeros_like(dist)
    mask1 = dist <= 0.1
    mask2 = (dist > 0.1) & (dist <= 0.6)
    result[mask1] = 30 * 5
    result[mask2] = (36 - 60 * dist[mask2]) * 5
    return result

  
def distanceToLine(slope,y_intersect,i,j):
    i=i-(IMAGE_SIZE[0]/2)
    j=j-(IMAGE_SIZE[1]/2)
    if slope=="yee":
        return abs((i - y_intersect))
    return abs(slope*i+y_intersect-j)/math.sqrt(slope**2+1)

def pixelOfLine(start_index,end_index):
    if start_index==end_index:
        print('not allowed')
        return 
    start_x, start_y = IMAGE_SIZE[0]/2*math.cos(2*math.pi*start_index/POINT_NUMBER),IMAGE_SIZE[1]/2*math.sin(2*math.pi*start_index/POINT_NUMBER)
    end_x, end_y = IMAGE_SIZE[0]/2*math.cos(2*math.pi*end_index/POINT_NUMBER),IMAGE_SIZE[1]/2*math.sin(2*math.pi*end_index/POINT_NUMBER)
    if abs(start_x - end_x) < 1e-5:
        slope = "yee"
        y_intersect = start_x
    else:
        slope = (start_y-end_y)/(start_x-end_x)
        y_intersect = (start_x*end_y-start_y*end_x)/(start_x-end_x)
    pixel_of_line = np.zeros(IMAGE_SIZE,dtype=np.float32)
    for i in range(IMAGE_SIZE[0]):
        for j in range(IMAGE_SIZE[1]):
            pixel_of_line[i,j] = threshold(distanceToLine(slope,y_intersect,i,j))
    pixel_of_line[~MASK] = 0
    return pixel_of_line

def bresenham_line(x0, y0, x1, y1):
    """Returns a list of (x, y) pixel coordinates from (x0, y0) to (x1, y1)"""
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))

    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points
def pixelOfLine_bresenham(start_index, end_index):
    if start_index == end_index:
        return np.zeros(IMAGE_SIZE, dtype=np.float32)

    # Convert polar index to Cartesian coordinates
    angle0 = 2 * math.pi * start_index / POINT_NUMBER
    angle1 = 2 * math.pi * end_index / POINT_NUMBER

    x0 = center[0] + radius * math.cos(angle0)
    y0 = center[1] + radius * math.sin(angle0)
    x1 = center[0] + radius * math.cos(angle1)
    y1 = center[1] + radius * math.sin(angle1)

    # Get the integer pixel coordinates
    coords = bresenham_line(x0, y0, x1, y1)

    # Create image mask
    pixel_mask = np.zeros(IMAGE_SIZE, dtype=np.float32)
    for x, y in coords:
        if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1] and MASK[y, x]:
            pixel_mask[y, x] = 150  # or a threshold value, like 150 if you'd rather fade

    return pixel_mask
def length_penalty(p1,p2):
    delta = abs(p1-p2)%POINT_NUMBER
    if delta==POINT_NUMBER//2:
        return 1
    else:
        delta = (delta%(POINT_NUMBER//2))/(POINT_NUMBER//2)        
        return 1+10*((1-delta)**2+delta**2)
def nextPoint(start,present_pixel,target_pixel):
    min_err=MAX_ERROR
    best_point = None
    
    if start==None:
        another_point = None
        for i in range(POINT_NUMBER):
            for j in range(i+1,POINT_NUMBER):
                if ((i,j)) in NOT_ALLOWED or (i,j) in USED_LINES:
                    continue
                line_pixel = LINE_PIXEL[(i,j)]
                line_pixel = merge_pixel(line_pixel,present_pixel)
                err = cal_error(line_pixel,target_pixel,present_pixel)
                if err<min_err:
                    best_point = i
                    another_point = j
                    min_err=err
        return best_point,another_point, min_err
    else:
        for point in range(POINT_NUMBER):
            if point == start or (min(start,point),max(start,point)) in NOT_ALLOWED or (min(start,point),max(start,point)) in USED_LINES:
                continue
            else:
                line_pixel = LINE_PIXEL[(min(start,point),max(start,point))]
                line_pixel = merge_pixel(line_pixel,present_pixel)
                err = cal_error(line_pixel,target_pixel,present_pixel)
                if err<min_err:
                    best_point = point
                    min_err=err

    return best_point, min_err

def process_line(i_j):
    i, j = i_j
    line_pixel = pixelOfLine_bresenham(i,j)
    if np.sum(line_pixel)<20000:
        NOT_ALLOWED.add((i,j))
    return (i, j), line_pixel

with ThreadPoolExecutor() as executor:
    results = executor.map(process_line, combinations(range(POINT_NUMBER), 2))
for (i, j), pixel in results:
    if (i,j) in NOT_ALLOWED:
        continue
    else:
        LINE_PIXEL[(i, j)] = pixel
frames=[]
N=5
fig, ax = plt.subplots()

for i in range(500):
    if i%(N*10)==0:
        start = None
        first, end, pre_err = nextPoint(start,drawed,pixel_value)
        print("reset",first,end)
        key = (min(first,end),max(first,end))
        USED_LINES.add(key)
        line = LINE_PIXEL[key]
        drawed = merge_pixel(drawed,line)
    else:
        new_end,err = nextPoint(end,drawed,pixel_value)
        if new_end==None:
            print("find no point")
            break
        # if pre_err-err<-1:
        #     print("too much error")
        #     break
        key = (min(end,new_end),max(end,new_end))
        print("new end",new_end)
        USED_LINES.add(key)
        line = LINE_PIXEL[key]
        drawed = merge_pixel(drawed,line)
        end = new_end
        pre_err = err
    if i%N==0:
        frame = drawed.copy()
        frame = 255*np.ones_like(frame)-frame
        frames.append([plt.imshow(frame, cmap='gray', animated=True)])

plt.imsave("./lines.png",drawed)
ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=False)
plt.show()

