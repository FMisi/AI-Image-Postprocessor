import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from PIL import Image, ImageFilter, ImageChops, ImageEnhance, ImageDraw, ExifTags
import numpy as np
import cv2
import threading

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Kép Feldolgozó")
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
        tk.Label(main_frame, text="AI Kép Feldolgozó", font=("Helvetica", 18, "bold"),
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
        tk.Label(main_frame, text="RGB zaj erőssége (alphap, 0.0-1.0):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.alphap_entry = ttk.Entry(main_frame, width=10)
        self.alphap_entry.insert(0, "0.15")
        self.alphap_entry.pack(fill="x", pady=2)

        tk.Label(main_frame, text="Film szemcse intenzitása (0.0-0.1):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.intensity_entry = ttk.Entry(main_frame, width=10)
        self.intensity_entry.insert(0, "0.01")
        self.intensity_entry.pack(fill="x", pady=2)

        tk.Label(main_frame, text="Vignettálás sugara (radius, 0.0-1.0):", bg="#ECEFF1", fg="#263238", font=("Helvetica", 13)).pack(pady=(5, 2))
        self.radius_entry = ttk.Entry(main_frame, width=10)
        self.radius_entry.insert(0, "0.995")
        self.radius_entry.pack(fill="x", pady=2)

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
        self.status_text = tk.Text(main_frame, height=5, width=60, bg="white", fg="#263238", font=("Helvetica", 12),
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
        self.alphap_entry.insert(0, "0.15")
        self.intensity_entry.delete(0, tk.END)
        self.intensity_entry.insert(0, "0.0095")
        self.radius_entry.delete(0, tk.END)
        self.radius_entry.insert(0, "0.995")
        self.camera_make_entry.delete(0, tk.END)
        self.camera_make_entry.insert(0, "Canon")
        self.camera_model_entry.delete(0, tk.END)
        self.camera_model_entry.insert(0, "Canon EOS 80D")
        self.status_text.delete(1.0, tk.END)

    def add_rgb_noise_overlay(self, img: Image.Image, strength=0.03, alphap=0.15):
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, strength, arr.shape).astype(np.float32)
        noisy = np.clip(arr + noise, 0, 1)
        alpha = alphap
        blended = (1 - alpha) * arr + alpha * noisy
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def add_film_grain(self, img: Image.Image, intensity=0.01):
        arr = np.array(img)
        h, w = arr.shape[:2]
        grain = np.random.normal(0, 255 * intensity, (h, w, 1)).astype(np.int16)
        grain = np.repeat(grain, 3, axis=2)
        noisy = np.clip(arr.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def add_vignette(self, img: Image.Image, radius=0.995, softness=0.995):
        w, h = img.size
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        dist = np.sqrt(xv**2 + yv**2)
        vignette_mask = np.clip(1 - ((dist - radius) / softness), 0, 1)
        vignette_mask = vignette_mask[..., np.newaxis]
        arr = np.array(img).astype(np.float32) / 255.0
        result = (arr * vignette_mask).clip(0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))

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

    def process_image(self, input_path, output_path, alphap, intensity, radius, camera_make, camera_model):
        try:
            img = Image.open(input_path).convert("RGB")
            img = self.add_rgb_noise_overlay(img, strength=0.03, alphap=alphap)
            img = self.add_film_grain(img, intensity=intensity)
            img = self.add_vignette(img, radius=radius, softness=radius)
            img = self.add_metadata(img, {
                "Make": camera_make,
                "Model": camera_model
            })
            img.save(output_path, "PNG", quality=95, exif=img.info.get("exif"))
            return f"Feldolgozva: {os.path.basename(input_path)}\n"
        except Exception as e:
            return f"Hiba {os.path.basename(input_path)} feldolgozása közben: {str(e)}\n"

    def update_status(self, message):
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.root.update()

    def start_processing(self):
        input_folder = self.input_folder_entry.get().strip()
        output_folder = self.output_folder_entry.get().strip()

        if not input_folder or not output_folder:
            messagebox.showwarning("Hiányzó Input", "Kérlek válassz input és output mappát is!")
            return

        if not os.path.exists(input_folder):
            messagebox.showerror("Hiba", "Az input mappa nem létezik!")
            return

        try:
            alphap = float(self.alphap_entry.get().strip())
            intensity = float(self.intensity_entry.get().strip())
            radius = float(self.radius_entry.get().strip())
            if not (0 <= alphap <= 1.0):
                raise ValueError("Az alphap értéknek 0 és 1 között kell lennie!")
            if not (0 <= intensity <= 0.1):
                raise ValueError("Az intensity értéknek 0 és 0.1 között kell lennie!")
            if not (0 <= radius <= 1.0):
                raise ValueError("A radius értéknek 0 és 1 között kell lennie!")
        except ValueError as e:
            messagebox.showerror("Hiba", str(e))
            return

        camera_make = self.camera_make_entry.get().strip()
        camera_model = self.camera_model_entry.get().strip()
        if not camera_make or not camera_model:
            messagebox.showwarning("Hiányzó Input", "Kérlek add meg a fényképezőgép márkáját és típusát!")
            return

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.process_button.config(state="disabled")
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "Képek feldolgozása...\n")

        def run_processing():
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
            for filename in os.listdir(input_folder):
                if filename.lower().endswith(image_extensions):
                    input_path = os.path.join(input_folder, filename)
                    output_path = os.path.join(output_folder, f"enhanced_{filename}")
                    status = self.process_image(input_path, output_path, alphap, intensity, radius, camera_make, camera_model)
                    self.root.after(0, self.update_status, status)
            
            self.root.after(0, self.finish_processing)

        thread = threading.Thread(target=run_processing)
        thread.start()

    def finish_processing(self):
        self.process_button.config(state="normal")
        self.status_text.insert(tk.END, "Az összes kép feldolgozása befejeződött!\n")
        self.status_text.see(tk.END)
        messagebox.showinfo("Siker", "Az összes kép fel lett dolgozva!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
