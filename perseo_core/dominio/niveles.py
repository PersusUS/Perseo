"""Los cuatro niveles de riesgo, que son el vocabulario de la política.

La tabla que dice qué nivel tiene cada cosa vive en `infra/politica.py`, con
quien la aplica. Los nombres bajan aquí porque los usa todo el mundo —los
agentes, la API, el chat— y un vocabulario compartido que vive dentro de quien
lo aplica obliga a importar hacia arriba para decir «esto es libre».
"""

from __future__ import annotations


LIBRE = "libre"
REVERSIBLE = "reversible"
IRREVERSIBLE = "irreversible"

#: Como irreversible, pero **el modo confianza no lo tapa**. Es para lo que no
#: se puede deshacer con nada: borrar ficheros, tocar el registro, matar
#: procesos. Nació el 2026-08-27 de un caso concreto: la llamada de voz enciende
#: la confianza al conectar (N-3), así que durante una llamada NADA preguntaba;
#: Perseo dijo de su cosecha «¿confirma que ejecuto el comando?», nadie
#: contestó, y el comando salió igual porque el sistema nunca llegó a
#: preguntarlo. Tener delante a alguien hablando no es su sí a esta orden.
CRITICO = "critico"

#: Lo que **sale de casa**: llega a otra persona o gasta dinero. Enviar un
#: correo, pulsar «Pagar» o «Reservar» en una web, meter una tarjeta en un
#: formulario. Es el único nivel que se para aunque las confirmaciones estén
#: apagadas (ADR 0007), y ni el modo confianza ni un sí de hace un rato lo tapan.
#:
#: Nació el 2026-09-24 mirando a Instinct, el asistente que hace recados por la
#: web: sus incidentes documentados son exactamente de esta familia —un correo
#: enviado sin permiso, un buzón que obedeció a un correo con instrucciones—.
#: Lo irreversible de casa (teclear, borrar) se deshace o se paga en casa; esto
#: lo ve alguien de fuera y ya no se recoge.
EXTERIOR = "exterior"

NIVELES = (LIBRE, REVERSIBLE, IRREVERSIBLE, CRITICO, EXTERIOR)
