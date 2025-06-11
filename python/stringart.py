from PIL import Image, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
import serial
import time
import cv2

def interactive_circle_crop(image_path):
    # Load image
    img = cv2.imread(image_path)
    clone = img.copy()
    height, width = img.shape[:2]

    # Initial circle parameters
    center = [width // 2, height // 2]
    radius = min(width, height) // 4
    dragging = False
    resizing = False

    def draw_circle(img):
        preview = img.copy()
        mask = np.zeros_like(img)
        cv2.circle(mask, tuple(center), radius, (255, 255, 255), -1)
        masked_img = cv2.bitwise_and(preview, mask)
        overlay = cv2.addWeighted(preview, 0.6, masked_img, 0.4, 0)
        cv2.circle(overlay, tuple(center), radius, (0, 255, 0), 2)
        return overlay

    def inside_circle(x, y):
        return np.sqrt((x - center[0])**2 + (y - center[1])**2) < radius

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, resizing, center, radius

        if event == cv2.EVENT_LBUTTONDOWN:
            if inside_circle(x, y):
                dragging = True
            elif abs(np.sqrt((x - center[0])**2 + (y - center[1])**2) - radius) < 10:
                resizing = True

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            resizing = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                center = [x, y]
            elif resizing:
                radius = int(np.sqrt((x - center[0])**2 + (y - center[1])**2))

    cv2.namedWindow("Adjust Circle (Press 'c' to confirm)")
    cv2.setMouseCallback("Adjust Circle (Press 'c' to confirm)", on_mouse)

    while True:
        overlay = draw_circle(clone)
        cv2.imshow("Adjust Circle (Press 'c' to confirm)", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # Press 'c' to confirm
            break

    cv2.destroyAllWindows()

    # Create mask and apply it
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, tuple(center), radius, 255, -1)
    masked_img = cv2.bitwise_and(clone, clone, mask=mask)

    # Crop bounding box around the circle
    x1, y1 = max(center[0]-radius, 0), max(center[1]-radius, 0)
    x2, y2 = min(center[0]+radius, width), min(center[1]+radius, height)
    cropped = masked_img[y1:y2, x1:x2]

    # Resize to square for your algorithm
    square_size = min(cropped.shape[:2])
    square_crop = cv2.resize(cropped, (square_size, square_size))
    return Image.fromarray(cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB)).convert('L')

class StringArtGenerator:
    def __init__(self, image_path, num_nails=200, num_lines=3500, radius=300, weight=30.0):
        self.num_nails = num_nails
        self.num_lines = num_lines
        self.radius = radius
        self.weight = weight

        self.image = self.preprocess_image(image_path)
        self.height, self.width = self.image.shape
        self.data = self.image.copy()
        self.drawed = np.zeros_like(self.data)

        self.nails = self.generate_nails()
        self.paths = self.compute_paths()
        self.line_sequence = []

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        # width, height = img.size
        # if width>height:
        #     top = 0
        #     bottom = height
        #     left = (width-height)//2
        #     right = left+height
        # else:
        #     left = 0
        #     right = width
        #     top = (height-width)//2
        #     bottom = top+width
        # img = img.crop((left,top,right,bottom))
        # # img.show()
        # img = ImageOps.invert(img)
        # img = ImageOps.autocontrast(img)
        # img = img.filter(ImageFilter.EDGE_ENHANCE)
        # img = img.resize((2*self.radius, 2*self.radius))
        # img = ImageOps.fit(img, (2*self.radius, 2*self.radius), centering=(0.5, 0.5))
        img = interactive_circle_crop(image_path)
        np_img = np.array(img)
        np_img = np.flipud(np_img).T
        return np_img

    def generate_nails(self):
        angles = np.linspace(0, 2*np.pi, self.num_nails, endpoint=False)
        cx, cy = self.width // 2, self.height // 2
        return [
            (int(cy + self.radius * np.sin(a)), int(cx + self.radius * np.cos(a)))
            for a in angles
        ]

    def compute_paths(self):
        path_cache = [[None] * self.num_nails for _ in range(self.num_nails)]
        for i in range(self.num_nails):
            y0, x0 = self.nails[i]
            for j in range(self.num_nails):
                if i != j:
                    y1, x1 = self.nails[j]
                    rr, cc = line(y0, x0, y1, x1)
                    path_cache[i][j] = (rr, cc)
        return path_cache

    def draw_line(self, start_idx, end_idx):
        rr, cc = self.paths[start_idx][end_idx]
        rr, cc = np.clip(rr, 0, self.height - 1), np.clip(cc, 0, self.width - 1)


        line_mask = np.zeros_like(self.data, dtype=np.float32)
        line_mask[rr, cc] = 1
        
        self.drawed = np.clip(self.drawed+line_mask*self.weight, 0, 255)
        self.data = np.clip(self.data-line_mask*self.weight, 0, 255)
        self.line_sequence.append((start_idx, end_idx))

    def run(self):
        current = 0
        max_err = 1e10
        count = 10
        self.point_sequence = []
        for _ in range(self.num_lines):
            best_score = -np.inf
            best_target = None

            for target in range(self.num_nails):
                if target == current:
                    continue
                rr, cc = self.paths[current][target]
                rr, cc = np.clip(rr, 0, self.height - 1), np.clip(cc, 0, self.width - 1)

                path_value = np.sum(self.data[rr, cc])
                if path_value > best_score:
                    best_score = path_value
                    best_target = target

            if best_target is None:
                break  # No valid lines left

            self.draw_line(current, best_target)
            self.point_sequence.append(best_target)
            # err = self.cal_error()
            # if err<max_err:
            #     max_err = err
            # else:
            #     if count==0:
            #         print(_)
            #         break
            #     else:
            #         count-=1
            current = best_target

    def plot_result(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = 255*np.ones_like(self.drawed)-self.drawed
        ax.imshow(im.T, cmap='gray', origin='lower')
        plt.show()

    def cal_error(self):
        a = np.sum(np.abs(self.drawed-self.image))
        return a
    
    def send_serial(self):
        point_order = self.point_sequence.copy()
        ser = serial.Serial('COM3', 9600)
        time.sleep(2)
        
        for point in point_order:
            ser.write(f"{point}\n".encode())  # Send nail index with newline
            ack = ser.readline().decode().strip()  # Wait for Arduino's ACK
            print(f"Arduino responded: {ack}")

        ser.close()
                
gen = StringArtGenerator("Yee.png")

gen.run()
gen.plot_result()
# gen.send_serial()

### 釘子順序:gen.point_sequence
