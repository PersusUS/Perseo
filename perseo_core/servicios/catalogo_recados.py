"""Las herramientas que actúan fuera: recados, vigilancias, correo que sale y citas.

Son parte del catálogo de `catalogo.py`, declaradas una vez como todas; viven
aquí porque aquel fichero llegaba a su techo de 900 líneas. `catalogo.CATALOGO`
las junta con las demás.
"""

from __future__ import annotations

from .catalogo_tipos import Herramienta, Parametro

RECADOS: tuple[Herramienta, ...] = (
    Herramienta(
        nombre="encargar_recado",
        voz=(
            "Encarga un recado en la web que Perseo hace solo, en segundo plano, con su propio "
            "navegador: reservar mesa, buscar y comparar, rellenar un formulario, comprar algo "
            "concreto. Tarda minutos; vuelve al momento con el número del trabajo y avisa al "
            "acabar. Lo que sale de casa —pagar, reservar, enviar— se para a esperar su sí, que "
            "él da en la tarjeta del panel o del móvil, NO hablando. Úsala solo con una orden "
            "suya, y con el encargo completo: qué, dónde, cuándo, para cuántos y hasta cuánto."
        ),
        chat=(
            "Recado en la web que Perseo hace solo con su navegador (reservar, comparar, "
            "rellenar formularios, comprar algo concreto). Tarda minutos: devuelve el #N y avisa "
            "al acabar. Pagar, reservar o enviar se para a esperar su sí en la tarjeta del panel "
            "o del móvil. 'texto' es el encargo completo: qué, dónde, cuándo, cuántos, hasta "
            "cuánto."
        ),
        parametros=(
            Parametro(
                nombre="texto",
                tipo="string",
                voz="El encargo entero, con todos los datos que haya dado.",
                chat="El encargo entero y autocontenido.",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="vigilancias",
        voz=(
            "Vigilar una web hasta que pase algo: «avísame cuando haya entradas para…», «dime si "
            "baja el vuelo de 80 €», «resérvalo en cuanto haya mesa». Perseo mira ahora y luego "
            "cada pocas horas, solo, hasta que se cumpla o caduque; al cumplirse te avisa, o lo "
            "hace si al_cumplirse es 'hacer' (y lo que se pague espera su sí en la tarjeta). "
            "que=crear con objetivo y condicion; que=listar para decir qué vigila; que=cancelar "
            "con objetivo para dejar de vigilar algo. Como mucho cinco a la vez."
        ),
        chat=(
            "Vigilar una web hasta que pase algo (entradas, precio, mesa libre): mira ahora y "
            "luego cada pocas horas hasta que se cumpla o caduque; al cumplirse avisa, o lo hace "
            "si al_cumplirse='hacer'. que=crear|listar|cancelar. Como mucho cinco a la vez."
        ),
        parametros=(
            Parametro(
                nombre="que",
                tipo="string",
                voz="crear, listar o cancelar.",
                chat="crear, listar o cancelar.",
                opciones=("crear", "listar", "cancelar"),
                obligatorio=True,
            ),
            Parametro(
                nombre="objetivo",
                tipo="string",
                voz="Qué y dónde se mira, con todos los datos: la web, el evento, las fechas. Al cancelar, su principio.",
                chat="Qué y dónde se mira, completo. Al cancelar, su principio.",
            ),
            Parametro(
                nombre="condicion",
                tipo="string",
                voz="Cuándo avisar o actuar, dicho claro: «hay entradas a la venta», «el precio baja de 80 €».",
                chat="Cuándo se cumple, dicho claro.",
            ),
            Parametro(
                nombre="cada_horas",
                tipo="number",
                voz="Cada cuántas horas mirar. Mínimo una; si no lo dice, tres.",
                chat="Cada cuántas horas (mín. 1, por defecto 3).",
            ),
            Parametro(
                nombre="dias",
                tipo="number",
                voz="Durante cuántos días vigilar. Si no lo dice, siete; como mucho treinta.",
                chat="Durante cuántos días (por defecto 7, máx. 30).",
            ),
            Parametro(
                nombre="al_cumplirse",
                tipo="string",
                voz="avisar (por defecto) o hacer, si ha pedido que lo haga en cuanto se pueda.",
                chat="avisar (por defecto) o hacer.",
                opciones=("avisar", "hacer"),
            ),
        ),
    ),
    Herramienta(
        nombre="enviar_borrador",
        voz=(
            "Envía un borrador que ya dejaste con redactar_borrador, cuando el señor Persus "
            "pida que salga. Pásale el id, el destinatario y el asunto EXACTOS que devolvió "
            "redactar_borrador. No sale al momento: espera su sí, que él da en la tarjeta del "
            "panel o del móvil, NO hablando. Díselo así, y no lo des por enviado."
        ),
        chat=(
            "Envía un borrador ya redactado, con el id, destinatario y asunto EXACTOS que "
            "devolvió redactar_borrador. Espera su sí en la tarjeta: no está enviado hasta "
            "entonces."
        ),
        parametros=(
            Parametro(nombre="borrador", tipo="string", voz="El id del borrador.", chat="El id del borrador.", obligatorio=True),
            Parametro(nombre="para", tipo="string", voz="El destinatario, tal cual.", chat="El destinatario, tal cual.", obligatorio=True),
            Parametro(nombre="asunto", tipo="string", voz="El asunto, tal cual.", chat="El asunto, tal cual.", obligatorio=True),
        ),
    ),
    Herramienta(
        nombre="crear_evento",
        voz=(
            "Apunta una cita en el calendario de Google del señor Persus: «apúntame cena el "
            "viernes a las nueve». Averigua antes la fecha de hoy con la hora si hace falta, "
            "y repite en voz alta el día y la hora que apuntas. Si hay invitados, a ellos les "
            "llega una invitación de Google, así que eso espera su sí en la tarjeta."
        ),
        chat=(
            "Apunta una cita en su calendario de Google. `inicio` en ISO local. Con invitados "
            "les llega la invitación, y eso espera su sí en la tarjeta."
        ),
        parametros=(
            Parametro(nombre="titulo", tipo="string", voz="Qué es, corto.", chat="Qué es, corto.", obligatorio=True),
            Parametro(
                nombre="inicio",
                tipo="string",
                voz="Día y hora de empezar, en ISO local: 2026-09-26T21:00.",
                chat="ISO local: 2026-09-26T21:00.",
                obligatorio=True,
            ),
            Parametro(nombre="duracion_min", tipo="number", voz="Cuánto dura, en minutos. Sin decirlo, una hora.", chat="Minutos (por defecto 60)."),
            Parametro(nombre="lugar", tipo="string", voz="Dónde, si lo dice.", chat="Dónde."),
            Parametro(nombre="descripcion", tipo="string", voz="Una nota, si la hay.", chat="Una nota."),
            Parametro(
                nombre="invitados",
                tipo="string",
                voz="Correos de los invitados separados por comas, solo si lo pide.",
                chat="Correos separados por comas.",
            ),
        ),
    ),
)
