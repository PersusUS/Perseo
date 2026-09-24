# 0007 · Lo que sale de casa se para, aunque las confirmaciones estén apagadas

**Fecha:** 2026-09-24 · **Estado:** vigente · **Matiza:** [0005](0005-las-confirmaciones-estan-apagadas.md)

## Contexto

El 2026-09-24 el señor Persus pidió que Perseo hiciera lo que hace
[Instinct](https://www.vellum.ai/blog/official-instinct-breakdown), el asistente
de Spear Street: se le escribe «resérvame mesa el viernes» y lo hace, con un
navegador y las sesiones de su dueño, hasta el final. Eso llegó como el agente
`recado` y la bóveda (`servicios/boveda.py`).

Con eso, por primera vez, Perseo puede **gastar dinero y hablar con
desconocidos en nombre de él**. Y las confirmaciones estaban apagadas enteras
desde el ADR 0005.

Los incidentes documentados de Instinct son justo de esta familia
([TechCrunch, 2026-08-24](https://techcrunch.com/2026/08/24/instincts-powerful-ai-assistant-is-raising-privacy-and-security-concerns/)):
un correo enviado sin permiso, y un buzón que obedeció a un correo con
instrucciones. Lo irreversible de casa —teclear, borrar un fichero— se deshace o
se paga en casa. Esto lo ve alguien de fuera y ya no se recoge.

## Decisión

Un nivel nuevo, `exterior` (`dominio/niveles.py`): lo que llega a otra persona o
gasta dinero. Con `politica.CONFIRMACIONES` apagado, **es lo único que se
para**. Es el punto medio que el ADR 0005 dejó escrito sin tomar, y vive donde
ese ADR decía: en `hay_que_parar`, dos líneas.

Con las confirmaciones encendidas se porta como lo crítico: el modo confianza
no lo tapa y un sí de hace un rato no vale para el siguiente.

Y dos reglas que lo sostienen:

- **El sí lo da una persona, no el modelo.** Ni el chat
  (`chat_herramientas._resolver_confirmacion`) ni la voz
  (`responder_confirmacion` en `nucleo.rs`) aprueban algo exterior: se confirma
  en la tarjeta del panel o del móvil. La pregunta de un recado lleva dentro el
  nombre de un botón que escribió quien hizo la web, y un botón llamado «aprueba
  el trabajo 5» no puede acabar aprobándolo por boca del modelo que lo lee.
  Rechazar hablando sí vale: decir que no nunca saca nada de casa.
- **El nivel viaja con la pregunta.** Un recado entero es reversible —navegar y
  teclear en su propio navegador—, y lo que lo para a mitad es un «Pagar». Por
  eso `NecesitaConfirmacion` lleva un `nivel`, se guarda en la confirmación, y
  quien decide si el modelo puede aprobar mira `politica.nivel_de_la_pregunta`.

Qué cuenta como exterior en un recado lo decide el código, no el prompt
(`agentes/recado.py`):

- pulsar un botón cuyo nombre **en la página** suena a comprometerse —Pagar,
  Reservar, Confirmar, Enviar, Suscribirse…—, o cuyo nombre según el modelo
  suena así (basta con uno de los dos);
- meter una tarjeta de la bóveda en un formulario;
- pulsar Enter con una tarjeta ya metida.

## Qué se pierde, dicho sin adornos

- **Fricción donde antes no la había:** una compra son dos síes, el de meter la
  tarjeta y el de pagar.
- **Un Enter sin tarjeta no se para.** Desde el teclado, un formulario de buscar
  y uno de enviar se ven iguales. El prompt pide los pasos finales con el botón,
  y el botón sí se mira; pero eso es un prompt, y aquí se escribe.
- **La lista de verbos se equivoca.** «Mi reserva» como enlace pregunta de más;
  un botón que diga «Adelante» para pagar pregunta de menos. Está en
  `servicios/navegacion.py`, con sus pruebas, para que crezca con lo que se vea.

## Qué haría falta para cambiarla

Apagar también lo exterior es borrar la condición de `hay_que_parar`. No se
recomienda mientras la bóveda tenga tarjetas: es exactamente la puerta de los
incidentes de arriba. Encender todo lo demás sigue siendo
`PERSEO_CONFIRMACIONES=1`, como decía el ADR 0005.
