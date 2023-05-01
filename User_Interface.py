import tkinter as tk
from Symptom_Checker import PredictDisease

class BotUI:
    def __init__(self, master):
        self.master = master
        master.title("Symptom Checker Bot")
        master.configure(bg="#00FFFF")
        master.geometry("750x400")

        # Define colors
        self.bg_color = "#4B0082"  # dark purple
        self.text_color = "#FFFFFF"  # white
        self.button_color = "#A380FC"  # blue-green
        self.button_text_color = "#FFFFFF"  # white

        # Create widgets
        self.label = tk.Label(master, text="Enter Symptoms Separated By Commas or Enter 'STOP' To Stop The Bot", font=("Times New Roman", 16, "bold"), fg=self.text_color, bg=self.bg_color)
        self.label.pack(pady=20)

        self.entry = tk.Entry(master, font=("Times New Roman", 14))
        self.entry.pack(padx=50, pady=20)

        self.button = tk.Button(master, text="Submit", font=("Times New Roman", 14), bg=self.button_color, fg=self.button_text_color, command=self.submit)
        self.button.pack(pady=20)

        self.output_label = tk.Label(master, text="", font=("Times New Roman", 14), fg=self.text_color, bg=self.bg_color)
        self.output_label.pack(pady=20)

    def submit(self):
        symptoms = self.entry.get().strip()
        if symptoms.lower() == 'stop':
            self.master.destroy()
        else:
            disease = PredictDisease(symptoms)
            disease = disease.split()
            display = []
            for word in disease:
                display.append(word.capitalize())
            disease = " ".join(display) 
            self.output_label.config(text=disease)

# Create and run the UI
root = tk.Tk()
bot_ui = BotUI(root)
root.mainloop()
