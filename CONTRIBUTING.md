# Cómo contribuir

Perseo es un asistente personal: nació para **una** persona, en **un**
ordenador con Windows, y eso se nota en muchos sitios. Que lo estés leyendo ya
es más de lo que esperaba, así que gracias.

## Lo que más ayuda

1. **Decir qué no arrancó.** Si lo clonaste y algo falló, eso es el informe
   más valioso que hay: significa que una suposición mía sobre «cómo es un
   ordenador» era falsa.
2. **Linux y macOS.** El núcleo es Python puro y debería correr en los dos,
   pero solo se prueba en CI. El detector de aplausos y el lanzador `perseo`
   son de Windows.
3. **Quitar cosas que sobran** antes que añadir cosas nuevas.

## Antes de abrir un PR

```bash
python commands/perseo.py comprobar
```

Eso es pytest, ruff, el buscador de código muerto, los tipos y las pruebas del
frontend, `knip` y `cargo check`, en orden de coste. Todo tiene que estar en
verde. Corre también en CI, en Linux y en Windows, así que si falla allí y aquí
no, suele ser una ruta con `\` o un final de línea. `--rapido` se salta Rust.

Algunas de esas comprobaciones no miran lo que hace el código sino **dónde vive**:
que el núcleo siga en capas, que ningún fichero pase de novecientas líneas, que
`docs/API.md` describa las rutas que existen y que los recuentos del README sean
los de verdad. Están explicadas en [`AGENTS.md`](AGENTS.md) y las decisiones que
hay detrás, en [`docs/adr/`](docs/adr/). Si una se pone roja, el arreglo no es
tocar la prueba.

Si cambias comportamiento, añade o ajusta la prueba que lo cubre. Si cambias
algo del sistema entero —la cola, la política, un agente— pasa además su
verificador (`verificadores/verificar_*.py`).

## Cómo se escribe aquí

Lee [`AGENTS.md`](AGENTS.md): está el modelo mental, dónde vive cada cosa y las
trampas que cuestan una hora. Lo resumido:

- **En castellano**: nombres, comentarios y mensajes de commit.
- **Los comentarios explican el porqué**, no el qué. Qué se midió, qué se
  rompió antes, por qué no se hizo de la otra manera.
- **Las caras no piensan**: la lógica va en el núcleo.
- **Nada de secretos en el código.**

Mensajes de commit al estilo del repositorio, con el ámbito entre paréntesis:

```
feat(memoria): buscar por título antes que por cuerpo
fix(llamada): la reconexión ya no se pone a cero al abrir el socket
docs(configuracion): las variables del detector
```

## Lo que probablemente no acepte

- Dependencias nuevas en el núcleo. Hoy tiene **una** (`aiohttp`), y esa
  austeridad es el motivo de que quepa en una Raspberry Pi.
- Telemetría, analítica o «informes de uso», en cualquier forma.
- Mover la memoria a una base vectorial o a un servicio en la nube. Ya fue así
  en la v1 y se quitó a propósito.
- Ampliar lo que un agente puede tocar sin una razón muy buena.

## Seguridad

Si encuentras una forma de que un correo, una página o un texto en pantalla
consigan que Perseo haga algo que su dueño no pidió, **no abras un issue
público**: [`SECURITY.md`](SECURITY.md).
