import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from PIL import Image, ImageFilter, ExifTags, ImageEnhance
import numpy as np
import cv2
import threading
from datetime import datetime
import random
import noise
import face_recognition
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Képfeldolgozó")
        self.root.geometry("600x920")
        self.root.configure(bg="#ECEFF1")
        
        window_width, window_height = 600, 920
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_left = int(screen_width / 2 - window_width / 2)
        root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#ECEFF1", foreground="#263238", font=("Helvetica", 11))
        style.configure("TButton", font=("Helvetica", 11, "bold"), padding=4, background="#0288D1", foreground="white")
        style.map("TButton", background=[("active", "#0277BD")])
        style.configure("TEntry", fieldbackground="white", foreground="#263238", font=("Helvetica", 11))
        style.configure("Green.TButton", foreground="black", background="green")
        style.configure("Red.TButton", foreground="black", background="red")
        
        main_frame = tk.Frame(root, bg="#ECEFF1", bd=1, relief="solid", padx=10, pady=10)
        main_frame.pack(expand=True, fill="both", padx=5, pady=5)
        tk.Label(main_frame, text="AI Képfeldolgozó", font=("Helvetica", 14, "bold"),
                 bg="#ECEFF1", fg="#263238").pack(pady=(0, 12))
        
        tk.Label(main_frame, text="Válassz input mappát:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.input_folder_entry = ttk.Entry(main_frame, width=40)
        self.input_folder_entry.pack(fill="x", pady=2)
        ttk.Button(main_frame, text="Böngészés...", command=self.browse_input_folder).pack(pady=4)
        
        tk.Label(main_frame, text="Válassz output mappát:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.output_folder_entry = ttk.Entry(main_frame, width=40)
        self.output_folder_entry.pack(fill="x", pady=2)
        ttk.Button(main_frame, text="Böngészés...", command=self.browse_output_folder).pack(pady=4)
        
        tk.Label(main_frame, text="RGB zaj erőssége (alphap):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.alphap_entry = ttk.Entry(main_frame, width=10)
        self.alphap_entry.insert(0, "0.053")  
        self.alphap_entry.pack(fill="x", pady=2)
        tk.Label(main_frame, text="Film szemcse intenzitása:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.intensity_entry = ttk.Entry(main_frame, width=10)
        self.intensity_entry.insert(0, "0.04")  
        self.intensity_entry.pack(fill="x", pady=2)
        tk.Label(main_frame, text="Vignettálás sugara (radius):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.radius_entry = ttk.Entry(main_frame, width=10)
        self.radius_entry.insert(0, "1.2")  
        self.radius_entry.pack(fill="x", pady=2)
        tk.Label(main_frame, text="Vignettálás lágyítása (softness):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.softness_entry = ttk.Entry(main_frame, width=10)
        self.softness_entry.insert(0, "2.5")  
        self.softness_entry.pack(fill="x", pady=2)
        tk.Label(main_frame, text="JPEG minőség (1-100):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.jpeg_quality_entry = ttk.Entry(main_frame, width=10)
        self.jpeg_quality_entry.insert(0, "90")  
        self.jpeg_quality_entry.pack(fill="x", pady=2)
        tk.Label(main_frame, text="Fényképezőgép márkája:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.camera_make_entry = ttk.Entry(main_frame, width=40)
        self.camera_make_entry.insert(0, "Canon")
        self.camera_make_entry.pack(fill="x", pady=2)
        tk.Label(main_frame, text="Fényképezőgép típusa:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 11)).pack(pady=(5, 2))
        self.camera_model_entry = ttk.Entry(main_frame, width=40)
        self.camera_model_entry.insert(0, "Canon EOS 80D")
        self.camera_model_entry.pack(fill="x", pady=2)
        
        self.process_button = ttk.Button(main_frame, text="Start", command=self.start_processing, style="Green.TButton")
        self.process_button.pack(pady=20)
        
        self.status_text = tk.Text(main_frame, height=6, width=50, bg="white", fg="#263238", font=("Helvetica", 11),
                                   relief="flat", bd=1, highlightthickness=1, highlightbackground="#B0BEC5")
        self.status_text.pack(pady=5, fill="x")
        
        button_frame = tk.Frame(main_frame, bg="#ECEFF1")
        button_frame.pack(pady=8)
        ttk.Button(button_frame, text="Szövegmezők törlése", command=self.clear_fields).pack(side="left", padx=3)
        ttk.Button(button_frame, text="Kilépés", command=root.quit, style="Red.TButton").pack(side="left", padx=3)
    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Válassz Input Mappát")
        if folder:
            self.input_folder_entry.delete(0, tk.END)
            self.input_folder_entry.insert(0, folder)
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Válassz Output Mappát")
        if folder:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder)
    def clear_fields(self):
        self.input_folder_entry.delete(0, tk.END)
        self.output_folder_entry.delete(0, tk.END)
        self.alphap_entry.delete(0, tk.END)
        self.alphap_entry.insert(0, "0.03")
        self.intensity_entry.delete(0, tk.END)
        self.intensity_entry.insert(0, "0.007")
        self.radius_entry.delete(0, tk.END)
        self.radius_entry.insert(0, "1.2")
        self.softness_entry.delete(0, tk.END)
        self.softness_entry.insert(0, "2.5")
        self.jpeg_quality_entry.delete(0, tk.END)
        self.jpeg_quality_entry.insert(0, "85")
        self.camera_make_entry.delete(0, tk.END)
        self.camera_make_entry.insert(0, "Canon")
        self.camera_model_entry.delete(0, tk.END)
        self.camera_model_entry.insert(0, "Canon EOS 80D")
        self.status_text.delete(1.0, tk.END)
    def add_chromatic_noise(self, img: Image.Image, intensity=0.005):
        """Add chromatic noise to simulate camera sensor imperfections."""
        arr = np.array(img).astype(np.float32) / 255.0
        for channel in range(3):
            noise = np.random.normal(0, intensity * random.uniform(0.8, 1.2), arr.shape[:2])
            arr[:, :, channel] += noise
        arr = np.clip(arr, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    def apply_lens_distortion(self, img: Image.Image, k1=-0.0001):
        """Apply subtle lens distortion to simulate optical imperfections."""
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([k1, 0, 0, 0], dtype=np.float32)
        img_cv = cv2.undistort(img_cv, camera_matrix, dist_coeffs)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    def apply_frequency_adjustment(self, img: Image.Image, strength=0.1):
        """Apply high-pass filter to reduce low-frequency AI patterns."""
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
        high_pass = img_cv.astype(np.float32) - blurred.astype(np.float32) + 127
        img_cv = cv2.addWeighted(img_cv.astype(np.float32), 1.0, high_pass, strength, 0)
        img_cv = np.clip(img_cv, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    def add_perlin_noise(self, img: Image.Image, scale=50, alphap=0.03, mask=None):
        w, h = img.size
        arr = np.array(img).astype(np.float32) / 255.0
        noise_map_bg = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                noise_map_bg[y][x] = noise.pnoise2(x / scale, y / scale, octaves=3, persistence=0.5, lacunarity=2.0,
                                                    repeatx=w, repeaty=h, base=0)
        noise_map_bg = (noise_map_bg - noise_map_bg.min()) / (noise_map_bg.max() - noise_map_bg.min())
        noise_map_face = np.zeros((h, w), dtype=np.float32)
        face_scale = max(scale / 2, 10)
        for y in range(h):
            for x in range(w):
                noise_map_face[y][x] = noise.pnoise2(x / face_scale, y / face_scale, octaves=3, persistence=0.5, lacunarity=2.0,
                                                      repeatx=w, repeaty=h, base=1)
        noise_map_face = (noise_map_face - noise_map_face.min()) / (noise_map_face.max() - noise_map_face.min())
        noise_map_bg = np.repeat(noise_map_bg[:, :, np.newaxis], 3, axis=2)
        noise_map_face = np.repeat(noise_map_face[:, :, np.newaxis], 3, axis=2)
        if mask is not None:
            alphap_face = alphap * 0.6  
            alphap_bg = alphap
            mask_blurred = cv2.GaussianBlur(mask, (101, 101), 0)
            mask_blurred_norm = mask_blurred.astype(np.float32) / 255.0
            mask_blurred_norm = np.power(mask_blurred_norm, 0.3)  
            noise_map = noise_map_bg * (1 - mask_blurred_norm[:, :, np.newaxis]) + noise_map_face * mask_blurred_norm[:, :, np.newaxis]
            alphap_map = alphap_bg * (1 - mask_blurred_norm[:, :, np.newaxis]) + alphap_face * mask_blurred_norm[:, :, np.newaxis]
            arr = arr * (1 - alphap_map) + noise_map * alphap_map
        else:
            arr = arr * (1 - alphap) + noise_map_bg * alphap
        arr = np.clip(arr, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    def apply_vignette(self, img: Image.Image, radius=1.2, softness=2.5):
        w, h = img.size
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        d = np.sqrt(X ** 2 + Y ** 2)
        vignette_mask = 1 - np.clip((d - radius) / softness, 0, 1)
        vignette_mask = vignette_mask[..., np.newaxis]
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * vignette_mask + (1 - vignette_mask) * 0.95  
        arr = np.clip(arr, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    def add_film_grain(self, img: Image.Image, intensity=0.007):
        arr = np.array(img).astype(np.float32) / 255.0
        noise_grain = np.random.normal(0, intensity, arr.shape)
        arr = arr + noise_grain
        arr = np.clip(arr, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    def apply_motion_blur(self, img: Image.Image, degree=5, angle=0):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        k = np.zeros((degree, degree))
        k[int((degree - 1) / 2), :] = np.ones(degree)
        M = cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1)
        k = cv2.warpAffine(k, M, (degree, degree))
        k = k / degree
        blurred = cv2.filter2D(img_cv, -1, k)
        return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    def detect_face_mask(self, img: Image.Image):
        img_rgb = np.array(img)
        face_locations = face_recognition.face_locations(img_rgb)
        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        for top, right, bottom, left in face_locations:
            margin = int((right - left) * 2.1)
            top = max(0, top - margin)
            bottom = min(img_rgb.shape[0], bottom + margin)
            left = max(0, left - margin)
            right = min(img_rgb.shape[1], right + margin)
            cv2.rectangle(mask, (left, top), (right, bottom), 255, thickness=-1)
        mask = cv2.GaussianBlur(mask, (251, 251), 0)
        return mask
    def add_exif_metadata(self, img: Image.Image, camera_make, camera_model):
        exif_dict = {
            ExifTags.TAGS.get("Make", 271): camera_make,
            ExifTags.TAGS.get("Model", 272): camera_model,
            ExifTags.TAGS.get("DateTime", 306): datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            ExifTags.TAGS.get("FNumber", 33437): (28, 10),  
            ExifTags.TAGS.get("ExposureTime", 33434): (1, 125),  
            ExifTags.TAGS.get("ISOSpeedRatings", 34855): 400,
            ExifTags.TAGS.get("LensModel", 42036): "EF-S18-55mm f/3.5-5.6 IS STM",
            ExifTags.TAGS.get("FocalLength", 41989): (35, 1),  
        }
        img_exif = Image.Exif()
        for tag_id, value in exif_dict.items():
            img_exif[tag_id] = value
        return img_exif
    def log_status(self, text):
        self.status_text.insert(tk.END, text + "\n")
        self.status_text.see(tk.END)
        self.status_text.update_idletasks()
    def get_unique_filename(self, folder, base_name, ext):
        candidate = f"{base_name}_processed{ext}"
        if not os.path.exists(os.path.join(folder, candidate)):
            return candidate
        i = 1
        while True:
            candidate = f"{base_name}_processed{i}{ext}"
            if not os.path.exists(os.path.join(folder, candidate)):
                return candidate
            i += 1
    def process_image(self, img_path, output_folder, alphap, intensity, radius, softness, jpeg_quality,
                      camera_make, camera_model, input_folder):
        try:
            img = Image.open(img_path).convert("RGB")
            
            img = self.add_chromatic_noise(img, intensity=0.005)
            
            face_mask = self.detect_face_mask(img)
            
            img_noise = self.add_perlin_noise(img, alphap=alphap, mask=face_mask)
            
            img_film = self.add_film_grain(img_noise, intensity=intensity)
            
            motion_blur_img = self.apply_motion_blur(img_film, degree=5, angle=random.uniform(-10, 10))
            
            motion_blur_arr = np.array(motion_blur_img)
            face_mask_3c = np.repeat(face_mask[:, :, np.newaxis] / 255.0, 3, axis=2)
            face_mask_3c = np.power(face_mask_3c, 0.3)  
            alpha_mb = 0.15  
            blended_arr = (1 - alpha_mb * face_mask_3c) * motion_blur_arr + (alpha_mb * face_mask_3c) * np.array(img_film)
            blended_arr = np.clip(blended_arr, 0, 255).astype(np.uint8)
            img_blurred = Image.fromarray(blended_arr)
            
            img_distorted = self.apply_lens_distortion(img_blurred)
            
            img_freq = self.apply_frequency_adjustment(img_distorted)
            
            img_vignette = self.apply_vignette(img_freq, radius=radius, softness=softness)
            
            img_np = np.array(img)
            img_final_np = np.array(img_vignette)
            face_mask_3c = np.repeat(face_mask[:, :, np.newaxis] / 255.0, 3, axis=2)
            face_mask_3c = np.power(face_mask_3c, 0.3)  
            alpha_face = 0.7  
            img_final_np = (1 - alpha_face * face_mask_3c) * img_final_np + (alpha_face * face_mask_3c) * img_np
            img_final_np = np.clip(img_final_np, 0, 255).astype(np.uint8)
            img_final = Image.fromarray(img_final_np)
            
            img_final.exif = self.add_exif_metadata(img_final, camera_make, camera_model)
            
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            if os.path.abspath(input_folder) == os.path.abspath(output_folder):
                new_filename = self.get_unique_filename(output_folder, name, ".jpg")
            else:
                new_filename = f"{name}_processed.jpg"
            save_path = os.path.join(output_folder, new_filename)
            img_final.save(save_path, quality=int(jpeg_quality), exif=img_final.exif)
            self.log_status(f"Kész: {new_filename}")
        except Exception as e:
            self.log_status(f"Hiba: {os.path.basename(img_path)} - {str(e)}")
    def start_processing(self):
        input_folder = self.input_folder_entry.get()
        output_folder = self.output_folder_entry.get()
        if not os.path.isdir(input_folder):
            messagebox.showerror("Hiba", "Érvénytelen input mappa!")
            return
        if not os.path.isdir(output_folder):
            messagebox.showerror("Hiba", "Érvénytelen output mappa!")
            return
        try:
            alphap = float(self.alphap_entry.get())
            intensity = float(self.intensity_entry.get())
            radius = float(self.radius_entry.get())
            softness = float(self.softness_entry.get())
            jpeg_quality = int(self.jpeg_quality_entry.get())
            camera_make = self.camera_make_entry.get()
            camera_model = self.camera_model_entry.get()
        except ValueError:
            messagebox.showerror("Hiba", "Az értékek formátuma nem megfelelő!")
            return
        self.status_text.delete(1.0, tk.END)
        self.log_status("Feldolgozás indítása...")
        def worker():
            files = [f for f in os.listdir(input_folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))]
            for f in files:
                img_path = os.path.join(input_folder, f)
                self.process_image(img_path, output_folder, alphap, intensity, radius, softness,
                                   jpeg_quality, camera_make, camera_model, input_folder)
            self.log_status("Feldolgozás befejeződött.")
        threading.Thread(target=worker, daemon=True).start()
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
