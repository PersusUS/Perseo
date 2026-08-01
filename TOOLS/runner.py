import sys
import json
import logging

# Solo mostrar errores para no contaminar el stdout que devolveremos a Rust
logging.basicConfig(level=logging.ERROR)

def main():
    if len(sys.argv) < 3:
        print("Uso: python runner.py <tool_name> '<json_args>'")
        sys.exit(1)

    tool_name = sys.argv[1]
    args_str = sys.argv[2]
    
    try:
        args = json.loads(args_str)
    except Exception as e:
        print(f"Error parseando argumentos JSON: {e}")
        sys.exit(1)

    try:
        if tool_name == "consultar_base_vectorial":
            from rag_tool import consultar_base_vectorial
            result = consultar_base_vectorial(**args)
            print(result)
        elif tool_name == "guardar_recuerdo":
            from memory_tool import guardar_recuerdo
            result = guardar_recuerdo(**args)
            print(result)
        elif tool_name == "controlar_pc":
            from pc_tool import controlar_pc
            result = controlar_pc(**args)
            print(result)
        else:
            print(f"Herramienta desconocida: {tool_name}")
            sys.exit(1)
    except Exception as e:
        print(f"Error ejecutando la herramienta interna: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()