import tkinter as tk
import subprocess
import os

def check_process(root, label, dots=0, retries=0):
    # Límite de seguridad: 30 segundos (30 ciclos de 1000ms)
    if retries > 30:
        root.destroy()
        return
        
    try:
        # Busca el proceso generado por el frontend final de Tauri.
        # "temp-app.exe" es el nombre deducido del Cargo.toml de tu proyecto nativo
        output = subprocess.check_output('tasklist /FI "IMAGENAME eq temp-app.exe"', shell=True).decode(errors='ignore')
        if "temp-app.exe" in output.lower():
            # Si lo detecta, damos 2 segundos extra para que Tauri dibuje la ventana y se cierre
            label.config(text="Interfáz gráfica lista.", fg="#3fb950")
            root.after(2000, root.destroy)
            return
    except Exception:
        pass
        
    # Animación simple de puntitos
    dots = (dots + 1) % 4
    animation = "." * dots
    label.config(text=f"Iniciando Perseo{animation}")
        
    # Programar el siguiente chequeo en 1 segundo
    root.after(1000, check_process, root, label, dots, retries + 1)

def main():
    root = tk.Tk()
    # Elimina los bordes de la ventana para que parezca un overlay moderno
    root.overrideredirect(True)
    # Fondo estilo terminal oscura/Github dark
    root.config(bg="#0d1117")
    # Fuerza a que esté por encima de todas las ventanas
    root.attributes("-topmost", True)
    # Transparencia
    root.attributes("-alpha", 0.9)

    # Centrar la ventana en la pantalla
    width, height = 350, 100
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - width) // 2
    y = (sh - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    # Borde sutil personalizado
    frame = tk.Frame(root, bg="#30363d", bd=1)
    frame.pack(fill=tk.BOTH, expand=True)

    inner_frame = tk.Frame(frame, bg="#0d1117")
    inner_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    # Texto
    label = tk.Label(
        inner_frame, 
        text="Iniciando Perseo...", 
        bg="#0d1117", 
        fg="#58a6ff", 
        font=("Consolas", 14, "bold")
    )
    label.pack(expand=True)

    # Iniciar la detección y animación
    root.after(1000, check_process, root, label, 0, 0)
    root.mainloop()

if __name__ == "__main__":
    main()