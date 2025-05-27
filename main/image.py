from PIL import Image, ImageDraw, ImageOps
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import cv2

IMAGE_FILE = "test.png"
IMAGE_SIZE = (300, 300)
POINT_NUMBER = 200
LINE_DICT = {}
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

# circle_mask = Image.new(mode="L", size=IMAGE_SIZE, color=0)
# circle_image = Image.new(mode="L", size=IMAGE_SIZE, color=255)
# circle2 = ImageDraw.Draw(circle_image)
# circle2.ellipse([(0,0),IMAGE_SIZE],fill=0)
# circle1 = ImageDraw.Draw(circle_mask)
# circle1.ellipse([(0,0),IMAGE_SIZE],fill=255)
# new_image = Image.composite(new_image,circle_image,mask=circle_mask)
# #new_image.show()
# Convert PIL image to NumPy and apply CLAHE

img_np = np.array(new_image)
blurred = cv2.GaussianBlur(img_np, (5, 5), 1.4)

# Apply Canny edge detection
edges = cv2.Canny(blurred, threshold1=30, threshold2=30)

# Convert to float for Radon input and apply circular mask
pixel_value = edges.astype(np.float32)
# kernel = np.array([[0, -1, 0],[-1,  5,-1],[0, -1, 0]])
# img_np = cv2.filter2D(img_np, -1, kernel)
# img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# img_clahe = clahe.apply(img_np)
Y, X = np.ogrid[:IMAGE_SIZE[1], :IMAGE_SIZE[0]]
dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
fade_mask = 1 - (dist_from_center / radius)**2
fade_mask = np.clip(fade_mask, 0, 1)

# Apply to edge map
# pixel_value *= fade_mask

# # Convert back to PIL
# new_image = Image.fromarray(img_clahe)
################################################################## specifying boudaries



############################################################# converting to inverse pixel value

# Apply MASK: set values outside circle to 0
pixel_value = 255-pixel_value
pixel_value[~MASK] = 0

plt.imshow(pixel_value, cmap='gray')
plt.title("Edge-detected input")
plt.axis("off")
plt.show()
############################################################# converting to inverse pixel value

########################################################### radon

sinogram = transform.radon(image=pixel_value,circle=True)

########################################################### radon


########################################################### functions
def merge_pixel(pixel1,pixel2):
    pixel3 = np.zeros(IMAGE_SIZE,dtype=np.float32)
    for i in range(IMAGE_SIZE[0]):
        for j in range(IMAGE_SIZE[1]):
            pixel3[i][j] = min(math.sqrt(pixel1[i][j]**2+pixel2[i][j]**2),255)
    return pixel3

def substract_gram(gram1,gram2):
    gram3 = np.zeros_like(gram1)
    for i in range(IMAGE_SIZE[0]):
        for j in range(180):
            gram3[i][j] = max(0,gram1[i][j]-gram2[i][j])
    return gram3
                   
def scaleSinogram(sinogram,parameter):
    for i in range(IMAGE_SIZE[0]):
        scale_val = 2*math.sqrt((IMAGE_SIZE[0]/2)**2-(i-IMAGE_SIZE[0]/2)**2)
        if scale_val==0:
            scale_val = 2
        for j in range(180):
            sinogram[i][j]=sinogram[i][j]/scale_val*parameter*100
    return sinogram 
def distanceToLine(slope,y_intersect,i,j):
    i=i-(IMAGE_SIZE[0]/2)
    j=j-(IMAGE_SIZE[1]/2)
    if slope=="yee":
        return abs((i - y_intersect))
    return abs(slope*i+y_intersect-j)/math.sqrt(slope**2+1)

def threshold(distance):
    # return 20 * math.exp(-distance**2 / (2 * 0.8**2))  # Smooth falloff
    if distance<=0.1:
        return 30*2/3
    elif distance<=0.6:
        return (36-60*distance)*2/3
    else:
        return 0   

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

def sinogramOfLine(pixel_of_line):    
    sinogram_of_line = transform.radon(pixel_of_line,circle=True)
    # sinogram_of_line = scaleSinogram(sinogram_of_line,1)
    return sinogram_of_line

def lineToSinogram(start_index,end_index):
    alpha = int(((start_index+end_index)+POINT_NUMBER/4)%(POINT_NUMBER/2)*180/(POINT_NUMBER/2))
    s = abs(math.cos(math.pi*((start_index-end_index)/POINT_NUMBER))*IMAGE_SIZE[0]/2)
    if abs(start_index-end_index)<=POINT_NUMBER/2:
        mid_arc = (start_index+end_index)/2
    else:
        mid_arc = ((start_index+end_index)/2+POINT_NUMBER/2)%POINT_NUMBER
    if mid_arc>=POINT_NUMBER/4 and mid_arc<=3*POINT_NUMBER/4:
        return math.floor(s+IMAGE_SIZE[0]//2), alpha
    else:
        return round(-s+IMAGE_SIZE[0]//2), alpha
  
def find_max(sinogram,start_index=None):
    if start_index==None:
        scaled_sinogram = np.zeros_like(sinogram)
        for i in range(IMAGE_SIZE[0]):
            scale_val = 2*math.sqrt((IMAGE_SIZE[0]/2)**2-(i-IMAGE_SIZE[0]/2)**2)
            if scale_val==0:
                scale_val = 10000
            scaled_sinogram[i]=sinogram[i]/scale_val
        max_pos, max_angle = np.unravel_index(np.argmax(scaled_sinogram), scaled_sinogram.shape)
        max_val = scaled_sinogram[max_pos][max_angle]
        theta = max_angle + math.asin((max_pos-IMAGE_SIZE[0]/2)/IMAGE_SIZE[0]*2)*180/math.pi
        theta1 = theta+2*math.acos((max_pos-IMAGE_SIZE[0]/2)/IMAGE_SIZE[0]*2)*180/math.pi
        index = round(theta*POINT_NUMBER/360)
        index = index%POINT_NUMBER
        index1 = round(theta1*POINT_NUMBER/360)
        index1 = index1%POINT_NUMBER
        idx_l = math.floor(theta*POINT_NUMBER/360)%POINT_NUMBER
        idx_h = math.ceil(theta*POINT_NUMBER/360)%POINT_NUMBER
        idx1_l = math.floor(theta1*POINT_NUMBER/360)%POINT_NUMBER
        idx1_h = math.ceil(theta1*POINT_NUMBER/360)%POINT_NUMBER
        sinogram[max_pos][max_angle]=sinogram[max_pos][max_angle]*4//5
        print(index,index1,max_val,end=' ')
        if (index==index1):
            index1 = (index1+1)%POINT_NUMBER
        if (min(index,index1),max(index,index1)) in LINE_DICT:
            print("same line",end=' ')
            if (min(idx_h,idx1_h),max(idx_h,idx1_h)) not in LINE_DICT and idx_h!=idx1_h:
                index = idx_h
                index1 = idx1_h
            elif (min(idx_h,idx1_l),max(idx_h,idx1_l)) not in LINE_DICT and idx_h!=idx1_l:
                index = idx_h
                index1 = idx1_l
            elif (min(idx_l,idx1_l),max(idx_l,idx1_l)) not in LINE_DICT and idx_l!=idx1_l:
                index = idx_l
                index1 = idx1_l
            elif (min(idx_l,idx1_h),max(idx_l,idx1_h)) not in LINE_DICT and idx_l!=idx1_h:
                index = idx_h
                index1 = idx1_l
            else:
                print('no more line',end=' ')
        return index ,index1, sinogram, max_val
    else:
        pass

def addLine(sinogram,start_index,drawed,flag):
    if start_index==None:
        a,b,sinogram,maximum = find_max(sinogram,None)
        if maximum<0.01:
           return sinogram, drawed, 1
        if (min(a,b),max(a,b)) not in LINE_DICT:
            LINE_DICT[(min(a,b),max(a,b))]=1
        else:
            LINE_DICT[(min(a,b),max(a,b))]+=1
        if LINE_DICT[(min(a,b),max(a,b))]>=5:
            line = pixelOfLine(a,b)
            #line_radon = sinogramOfLine(line)
            #sinogram = substract_gram(sinogram,line_radon)
            print('ignored',end=' ')
            return sinogram, drawed, flag
        else:
            line = pixelOfLine(a,b)
            line_radon = sinogramOfLine(line)
            sinogram = substract_gram(sinogram,line_radon)
            drawed = merge_pixel(drawed,line)
            return sinogram, drawed, flag
    else:
        pass

#sinogram = scaleSinogram(sinogram,1)
#print(np.max(sinogram))
# plt.imshow(sinogram,cmap='gray')
# plt.colorbar()
# plt.show()
# a = find_max(sinogram,51)
# print(a)
# line = pixelOfLine(0,100)
# line = sinogramOfLine(line)
# print(np.max(line))
# plt.imshow(line,cmap='gray')
# plt.colorbar()
# plt.show()

end=None
flag=0
frames=[]
N=5
fig, ax = plt.subplots()

for i in range(5000):
    sinogram, drawed, flag= addLine(sinogram,None,drawed,flag)
    print(i)
    if i%N==0:
        frame = drawed.copy()
        frame = 255*np.ones_like(frame)-frame
        frames.append([plt.imshow(frame, cmap='gray', animated=True)])
    if(flag):
        break
    # plt.imshow(drawed,cmap='gray')
    # plt.colorbar()
    # plt.show()
plt.imsave("./lines.png",drawed)
print(np.max(drawed))
ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=False)
plt.show()

# drawed = 255*np.ones(IMAGE_SIZE,dtype=np.float32)-drawed
# plt.imshow(drawed,cmap='gray')
# plt.colorbar()
# plt.show()

# plt.imshow(pixel_value,cmap='gray')
# plt.colorbar()
# plt.show()
