import os
import sys
import winreg

def add_to_startup():
    # Ruta del script
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "clap_detector.py"))
    
    # Usamos pythonw para que no se abra la ventana de consola negra
    # Buscamos la ruta del ejecutable de python, pero le agregamos la 'w' al final
    python_exe = sys.executable
    if python_exe.endswith("python.exe"):
        pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
    else:
        pythonw_exe = python_exe

    # Comando completo a ejecutar
    command = f'"{pythonw_exe}" "{script_path}"'
    
    # Clave de registro para inicio automático (Solo para el usuario actual)
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "PerseoClapDetector"

    try:
        # Abrir la clave donde se guarda la configuración de arranque
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, command)
        winreg.CloseKey(key)
        print(f"[+] ¡Éxito! El detector de aplausos ('{app_name}') se ejecutará automáticamente (de forma oculta) cada vez que enciendas el PC.")
    except Exception as e:
        print(f"[-] Ocurrió un error al añadir al registro: {e}")

def remove_from_startup():
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "PerseoClapDetector"
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        winreg.DeleteValue(key, app_name)
        winreg.CloseKey(key)
        print("[+] El detector de aplausos se ha deshabilitado del inicio del sistema.")
    except FileNotFoundError:
        print("[i] El programa no estaba configurado para iniciar con Windows.")
    except Exception as e:
        print(f"[-] Error al eliminar del registro: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "remove":
        remove_from_startup()
    else:
        print("--- Configuración de Auto-Inicio en Background ---")
        add_to_startup()
        print("\nPara detener el proceso si ya está corriendo ahora, tendrás que cerrarlo ejecutando:")
        print("taskkill /F /IM pythonw.exe")
        print("\nPara quitarlo del inicio en el futuro, ejecuta:")
        print("python manage_startup.py remove")