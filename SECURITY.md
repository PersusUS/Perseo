# Seguridad

Perseo lee correo, mira la pantalla, abre programas y teclea. La superficie
que importa aquí no es la de siempre: es que **texto escrito por un
desconocido acabe moviendo las manos del asistente**.

## Qué cuenta como fallo de seguridad

- Que un correo, una página web, un resultado de herramienta o algo visible en
  pantalla consiga que Perseo ejecute una acción que su dueño no pidió
  (inyección de instrucciones).
- Saltarse la política de niveles: que algo `irreversible` se ejecute sin
  pasar por un sí.
- Salirse de un cerco: que el agente `pc` ejecute fuera de su lista blanca,
  que `web` alcance una dirección privada, que `dev` escriba fuera de su raíz.
- Sacar datos del sistema por un camino que no debería llevarlos: el cuerpo de
  un correo por Telegram, un vector biométrico por la API, el token en un
  registro.
- Cualquier ruta de la API que responda sin token y no debería.

## Qué no lo es

- Que el núcleo deje de funcionar sin Ollama, sin Obsidian o sin Google: es el
  diseño, y lo dice la pantalla de Estado.
- Que quien ya tiene el token pueda encolar trabajos. El token **es** la
  credencial; lo que no puede es conseguir una consola, porque la API no
  expone herramientas.
- Que `PERSEO_DEV_RAIZ` abarque mucho si tú lo has puesto así.

## Cómo avisar

**No abras un issue público.** Usa el aviso privado de GitHub —pestaña
**Security → Report a vulnerability** de este repositorio—, que abre un hilo
que solo vemos tú y yo. Cuenta ahí:

- qué hace falta para reproducirlo,
- qué consigue quien lo explota,
- y, si lo tienes, el texto exacto que lo dispara.

Contesto en cuanto lo vea. Es un proyecto de una persona, así que no hay
programa de recompensas ni compromiso de plazos, pero sí crédito en el arreglo
si lo quieres.

## Lo que ya está puesto

Por si ahorra tiempo, estas son las defensas que hay y que conviene atacar
antes de dar por bueno un hallazgo:

- La regla de que **lo observado no es una instrucción**, en el prompt de los
  tres modelos, con una prueba que falla si se quita.
- La **política de niveles** aplicada en el trabajador, no en el prompt: lo
  que no está clasificado cuenta como irreversible.
- El agente `pc`: lista blanca explícita, sin shell.
- El agente `web`: direcciones privadas cerradas.
- El agente `dev`: raíz de la que no sale.
- El núcleo: nunca escucha en `0.0.0.0`, y exige token en toda petición salvo
  las que sirven la propia web.
- Del correo solo se descargan cabeceras y extracto: el cuerpo no se baja.

Hay un verificador por cada una de esas líneas
(`verificadores/verificar_pc.py`, `verificar_web.py`, `verificar_politica.py`…).
