from PIL import Image, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
import serial
import time
import tkinter as tk
from tkinter import filedialog, messagebox
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
        width, height = img.size
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
        img = img.crop((left,top,right,bottom))
        # img.show()
        img = ImageOps.invert(img)
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        img = img.resize((2*self.radius, 2*self.radius))
        # img = ImageOps.fit(img, (2*self.radius, 2*self.radius), centering=(0.5, 0.5))

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
    

user_params = {}
def ask_user_input():
    def browse_image():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            entry_file.delete(0, tk.END)
            entry_file.insert(0, path)
    def submit():
        try:
            image_file = entry_file.get()
            num_nails= int(entry_nail.get())
            num_lines = int(entry_numlines.get())
            user_params['IMAGE_FILE'] = image_file
            user_params['NAIL_NUMBER'] = num_nails
            user_params["NUM_LINES"] = num_lines
            
            root.destroy()
        except Exception as e:
            messagebox.showerror("錯誤", str(e))

    root = tk.Tk()
    root.title("輸入圖片參數")
    root.geometry("400x300")

    tk.Label(root, text="image_patb:").pack()
    entry_file = tk.Entry(root, width=40)
    entry_file.insert(0, "hushih.jpg")
    entry_file.pack()
    tk.Button(root, text="browse...", command=browse_image).pack()

    tk.Label(root, text="nail numbers:").pack()
    entry_nail = tk.Entry(root)
    entry_nail.insert(0, "200")
    entry_nail.pack()

    
    tk.Label(root, text="num_lines:").pack()
    entry_numlines = tk.Entry(root)
    entry_numlines.insert(0, "3500")
    entry_numlines.pack()

    
    tk.Button(root, text="start generating", command=submit).pack(pady=10)
    root.mainloop()  
ask_user_input()
IMAGE_FILE = user_params['IMAGE_FILE']
NAIL_NUMBER = user_params['NAIL_NUMBER']
NUM_LINES = user_params['NUM_LINES']
def line_sequence_to_point_sequence(line_sequence):
    if not line_sequence:
        return []

    result = [line_sequence[0][0], line_sequence[0][1]]  # 起點與第一個終點
    for i in range(1, len(line_sequence)):
        result.append(line_sequence[i][1])  # 加入每條線的終點
    return result

def export_to_arduino(point_sequence, filename="C:/carcar/carcar-final_project/main/nail_list.h", var_name="nail_list"):
    with open(filename, "w") as f:
        f.write(f"int {var_name}[] = {{\n")
        for i in range(0, len(point_sequence), 10):
            line = ", ".join(str(p) for p in point_sequence[i:i+10])
            f.write(f"  {line},\n")
        f.write("};\n")

gen = StringArtGenerator(IMAGE_FILE,NAIL_NUMBER,NUM_LINES)
gen.run()
print(export_to_arduino(gen.line_sequence))
print(line_sequence_to_point_sequence(gen.line_sequence))
export_to_arduino(line_sequence_to_point_sequence(gen.line_sequence))
gen.plot_result()
# gen.semd_serial()


import subprocess
arduino_cli_path = "C:/Users/edwin/Downloads/arduino-cli_1.2.2_Windows_64bit/arduino-cli.exe"
def upload_sketch(sketch_path, com_port="COM18", fqbn="arduino:avr:mega"):
    compile_cmd = [arduino_cli_path, "compile", "--fqbn", fqbn, sketch_path]
    upload_cmd = [arduino_cli_path, "upload", "-p", com_port, "--fqbn", fqbn, sketch_path]
    
    print("Compiling")
    subprocess.run(compile_cmd, check=True)

    print("Uploading")
    subprocess.run(upload_cmd, check=True)

upload_sketch("C:/carcar/carcar-final_project/main")
