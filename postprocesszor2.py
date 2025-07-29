import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from PIL import Image, ImageFilter, ExifTags, ImageEnhance
import numpy as np
import cv2
import threading
from datetime import datetime
import random

# Ha nincs telepítve: pip install noise
import noise

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Képfeldolgozó")
        self.root.geometry("700x900")
        self.root.configure(bg="#ECEFF1")

        # Ablak középre igazítása
        window_width, window_height = 700, 900
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_left = int(screen_width / 2 - window_width / 2)
        root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')

        # Modern stílus konfigurálása
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#ECEFF1", foreground="#263238", font=("Helvetica", 13))
        style.configure("TButton", font=("Helvetica", 13, "bold"), padding=6, background="#0288D1", foreground="white")
        style.map("TButton", background=[("active", "#0277BD")])
        style.configure("TEntry", fieldbackground="white", foreground="#263238", font=("Helvetica", 13))
        style.configure("Green.TButton", foreground="black", background="green")
        style.configure("Red.TButton", foreground="black", background="red")

        # Fő keret
        main_frame = tk.Frame(root, bg="#ECEFF1", bd=1, relief="solid", padx=15, pady=15)
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Cím
        tk.Label(main_frame, text="AI Képfeldolgozó", font=("Helvetica", 18, "bold"),
                bg="#ECEFF1", fg="#263238").pack(pady=(0, 20))

        # Input mappa választás
        tk.Label(main_frame, text="Válassz input mappát:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.input_folder_entry = ttk.Entry(main_frame, width=50)
        self.input_folder_entry.pack(fill="x", pady=2)
        ttk.Button(main_frame, text="Böngészés...", command=self.browse_input_folder).pack(pady=5)

        # Output mappa választás
        tk.Label(main_frame, text="Válassz output mappát:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.output_folder_entry = ttk.Entry(main_frame, width=50)
        self.output_folder_entry.pack(fill="x", pady=2)
        ttk.Button(main_frame, text="Böngészés...", command=self.browse_output_folder).pack(pady=5)

        # Paraméter beviteli mezők
        tk.Label(main_frame, text="RGB zaj erőssége (alphap):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.alphap_entry = ttk.Entry(main_frame, width=10)
        self.alphap_entry.insert(0, "0.05")
        self.alphap_entry.pack(fill="x", pady=2)

        tk.Label(main_frame, text="Film szemcse intenzitása:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.intensity_entry = ttk.Entry(main_frame, width=10)
        self.intensity_entry.insert(0, "0.005")
        self.intensity_entry.pack(fill="x", pady=2)

        tk.Label(main_frame, text="Vignettálás sugara (radius):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.radius_entry = ttk.Entry(main_frame, width=10)
        self.radius_entry.insert(0, "0.9999999")
        self.radius_entry.pack(fill="x", pady=2)

        tk.Label(main_frame, text="Vignettálás lágyítása (softness):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.softness_entry = ttk.Entry(main_frame, width=10)
        self.softness_entry.insert(0, "0.9999999")
        self.softness_entry.pack(fill="x", pady=2)

        tk.Label(main_frame, text="Fényképezőgép márkája:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.camera_make_entry = ttk.Entry(main_frame, width=50)
        self.camera_make_entry.insert(0, "Canon")
        self.camera_make_entry.pack(fill="x", pady=2)

        tk.Label(main_frame, text="Fényképezőgép típusa:", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.camera_model_entry = ttk.Entry(main_frame, width=50)
        self.camera_model_entry.insert(0, "Canon EOS 80D")
        self.camera_model_entry.pack(fill="x", pady=2)

        # Feldolgozás gomb
        self.process_button = ttk.Button(main_frame, text="Start", command=self.start_processing, style="Green.TButton")
        self.process_button.pack(pady=30)

        # Állapot szövegmező
        self.status_text = tk.Text(main_frame, height=8, width=60, bg="white", fg="#263238", font=("Helvetica", 12),
                                  relief="flat", bd=1, highlightthickness=1, highlightbackground="#B0BEC5")
        self.status_text.pack(pady=5, fill="x")

        # Gombok kerete
        button_frame = tk.Frame(main_frame, bg="#ECEFF1")
        button_frame.pack(pady=10)

        # Törlés és Kilépés gombok
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
        self.alphap_entry.insert(0, "0.05")
        self.intensity_entry.delete(0, tk.END)
        self.intensity_entry.insert(0, "0.005")
        self.radius_entry.delete(0, tk.END)
        self.radius_entry.insert(0, "0.999")
        self.softness_entry.delete(0, tk.END)
        self.softness_entry.insert(0, "0.2")
        self.camera_make_entry.delete(0, tk.END)
        self.camera_make_entry.insert(0, "Canon")
        self.camera_model_entry.delete(0, tk.END)
        self.camera_model_entry.insert(0, "Canon EOS 80D")
        self.status_text.delete(1.0, tk.END)

    def add_perlin_noise(self, img: Image.Image, scale=50, alphap=0.05):
        w, h = img.size
        arr = np.array(img).astype(np.float32) / 255.0
        noise_map = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                noise_map[y][x] = noise.pnoise2(x / scale, y / scale, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=w, repeaty=h, base=0)
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        noise_map = np.repeat(noise_map[:, :, np.newaxis], 3, axis=2)
        noisy = np.clip(arr + alphap * noise_map, 0, 1)
        noisy = (noisy * 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def add_rgb_noise_overlay(self, img: Image.Image, alphap=0.05):
        # Csatornánként eltérő zaj erősség és véletlenszerű eltolás
        arr = np.array(img).astype(np.float32) / 255.0
        noisy = arr.copy()
        for c in range(3):
            strength = alphap * random.uniform(0.8, 1.2)
            noise = np.random.normal(0, strength, arr[..., c].shape).astype(np.float32)
            noisy[..., c] = np.clip(arr[..., c] + noise, 0, 1)
        noisy = (noisy * 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def add_film_grain(self, img: Image.Image, intensity=0.005):
        arr = np.array(img)
        h, w = arr.shape[:2]
        # Különböző zaj erősség RGB csatornánként
        grain = np.zeros((h, w, 3), dtype=np.int16)
        for c in range(3):
            grain[..., c] = np.random.normal(0, 255 * intensity * random.uniform(0.7, 1.3), (h, w)).astype(np.int16)
        noisy = np.clip(arr.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def add_vignette(self, img: Image.Image, radius=0.999, softness=0.2):
        w, h = img.size
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        dist = np.sqrt(xv**2 + yv**2)
        # Gaussos lágyítás
        vignette_mask = np.exp(-((dist - radius) ** 2) / (2 * (softness ** 2)))
        vignette_mask = np.clip(vignette_mask, 0, 1)
        vignette_mask = vignette_mask[..., np.newaxis]
        arr = np.array(img).astype(np.float32) / 255.0
        result = (arr * vignette_mask).clip(0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))

    def add_motion_blur(self, img: Image.Image, degree=5, angle=0):
        img_cv = np.array(img)
        kernel = np.zeros((degree, degree))
        kernel[int((degree-1)/2), :] = np.ones(degree)
        M = cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (degree, degree))
        kernel = kernel / degree
        blurred = cv2.filter2D(img_cv, -1, kernel)
        return Image.fromarray(blurred)

    def add_metadata(self, img: Image.Image, metadata: dict):
        exif = img.getexif()
        for tag, value in metadata.items():
            tag_id = None
            for k, v in ExifTags.TAGS.items():
                if v == tag:
                    tag_id = k
                    break
            if tag_id:
                exif[tag_id] = value
        img.info["exif"] = exif.tobytes()
        return img

    def process_single_image(self, img_path, output_folder, alphap, intensity, radius, softness, camera_make, camera_model):
        try:
            img = Image.open(img_path).convert("RGB")

            # Perlin zaj vagy sima RGB zaj választható
            if random.random() < 0.5:
                img = self.add_perlin_noise(img, scale=50, alphap=alphap)
            else:
                img = self.add_rgb_noise_overlay(img, alphap=alphap)

            img = self.add_film_grain(img, intensity=intensity)
            img = self.add_motion_blur(img, degree=random.randint(3, 6), angle=random.uniform(-10, 10))
            img = self.add_vignette(img, radius=radius, softness=softness)

            # EXIF metaadatok beállítása
            now_str = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
            metadata = {
                "Make": camera_make,
                "Model": camera_model,
                "DateTime": now_str,
                "Software": "AI ImageProcessor v1.0",
                "ExposureTime": (1, 60),  # Példa, 1/60 sec
                "FocalLength": (50, 1),   # Példa, 50mm
            }
            img = self.add_metadata(img, metadata)

            # Mentés az output mappába
            base_name = os.path.basename(img_path)
            out_path = os.path.join(output_folder, base_name)
            img.save(out_path, "JPEG", quality=95, exif=img.info.get("exif"))

            return True, base_name
        except Exception as e:
            return False, f"Hiba a feldolgozás során: {str(e)}"

    def update_status(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)

    def process_images_thread(self, input_folder, output_folder, alphap, intensity, radius, softness, camera_make, camera_model):
        file_list = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp"))]
        total = len(file_list)
        if total == 0:
            self.update_status("Nincs feldolgozható kép az input mappában.")
            self.process_button.config(state="normal")
            return

        self.update_status(f"Összesen {total} képet találtam. Feldolgozás elkezdve...")

        for idx, file_name in enumerate(file_list, start=1):
            img_path = os.path.join(input_folder, file_name)
            success, msg = self.process_single_image(img_path, output_folder, alphap, intensity, radius, softness, camera_make, camera_model)
            if success:
                self.update_status(f"[{idx}/{total}] {msg} sikeresen feldolgozva.")
            else:
                self.update_status(f"[{idx}/{total}] Hiba: {msg}")

        self.update_status("Feldolgozás befejezve.")
        self.process_button.config(state="normal")

    def start_processing(self):
        input_folder = self.input_folder_entry.get()
        output_folder = self.output_folder_entry.get()

        try:
            alphap = float(self.alphap_entry.get())
            intensity = float(self.intensity_entry.get())
            radius = float(self.radius_entry.get())
            softness = float(self.softness_entry.get())
            camera_make = self.camera_make_entry.get().strip()
            camera_model = self.camera_model_entry.get().strip()

            if not os.path.isdir(input_folder):
                messagebox.showerror("Hiba", "Az input mappa nem létezik.")
                return
            if not os.path.isdir(output_folder):
                messagebox.showerror("Hiba", "Az output mappa nem létezik.")
                return

            self.status_text.delete(1.0, tk.END)
            self.process_button.config(state="disabled")
            threading.Thread(target=self.process_images_thread, args=(
                input_folder, output_folder, alphap, intensity, radius, softness, camera_make, camera_model
            ), daemon=True).start()
        except ValueError:
            messagebox.showerror("Hiba", "Kérlek, érvényes számokat adj meg az paraméterekhez.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
