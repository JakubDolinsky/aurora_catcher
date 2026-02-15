import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os

from application.mid_layer.mid_layer import MidLayer
from application.presentation_layer.translator import translate_engine_output

MAX_LONG_SIDE = 6000
MAX_TOTAL_PIXELS = 30_000_000
class AuroraApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Aurora Catcher")
        self.root.geometry("400x250")

        self.mid_layer = MidLayer()
        self.loaded_image_path = None

        self.load_button = tk.Button(root, text="Load picture", command=self.load_picture)
        self.load_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect aurora", command=self.detect)
        self.detect_button.pack(pady=10)

        self.status_label = tk.Label(root, text="No image loaded")
        self.status_label.pack(pady=5)

        # Loader (progress bar)
        self.progress = ttk.Progressbar(
            root,
            orient="horizontal",
            mode="indeterminate",
            length=250
        )
        self.progress.pack(pady=10)

    def load_picture(self):
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        try:
            from PIL import Image

            with Image.open(file_path) as img:
                img.verify()

            with Image.open(file_path) as img:
                width, height = img.size

                if max(width, height) > MAX_LONG_SIDE:
                    raise Exception(
                        f"Image too large. Maximum allowed long side is {MAX_LONG_SIDE}px"
                    )

                if width * height > MAX_TOTAL_PIXELS:
                    raise Exception(
                        "Image too large. Image resolution exceeds 30 megapixels."
                    )
        except Exception as e:
            messagebox.showerror("Invalid image",str(e))
            return

        self.loaded_image_path = file_path
        self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")

    def detect(self):
        if not self.loaded_image_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # disable button during inference
        self.detect_button.config(state="disabled")
        self.progress.start(10)  # speed of animation

        # run inference in separate thread
        thread = threading.Thread(target=self.run_inference)
        thread.start()

    def run_inference(self):
        output = self.mid_layer.decide_if_aurora_or_detect_other_phenomena(
            self.loaded_image_path
        )

        message = translate_engine_output(output)

        # return to main GUI thread
        self.root.after(0, lambda: self.finish_inference(message))

    def finish_inference(self, message):
        self.progress.stop()
        self.detect_button.config(state="normal")
        self.show_result_popup(message)

    def show_result_popup(self, message):
        popup = tk.Toplevel(self.root)
        popup.title("Detection Result")
        popup.resizable(True, True)

        text_widget = tk.Text(popup, wrap="word")
        text_widget.pack(padx=10, pady=10)

        lines = message.split("\n", 1)
        first_line = lines[0]
        rest_text = lines[1] if len(lines) > 1 else ""

        import tkinter.font as tkfont
        default_font = tkfont.nametofont("TkDefaultFont")
        bold_font = default_font.copy()
        bold_font.configure(size=default_font["size"] + 10, weight="bold")

        text_widget.insert("1.0", first_line + "\n", "header")
        text_widget.insert("end", rest_text)
        text_widget.tag_configure("header", font=bold_font)
        text_widget.config(state="disabled")

        save_button = tk.Button(
            popup,
            text="Save as txt",
            command=lambda: self.save_result(message)
        )
        save_button.pack(pady=5)

        popup.update_idletasks()
        popup.geometry("")

    def save_result(self, message):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")]
        )

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(message)

            messagebox.showinfo("Saved", "Result saved successfully.")


def main():
    root = tk.Tk()
    app = AuroraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
