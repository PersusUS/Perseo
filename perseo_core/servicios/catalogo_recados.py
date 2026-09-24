"""Las herramientas de los recados en la web y de las vigilancias.

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
)
