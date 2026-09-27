# 0009 · Canales de fuera: Telegram, WhatsApp y el teléfono

**Fecha:** 2026-09-25 · **Estado:** vigente · **Matiza:** la regla de `caras/telegram.py`

## Contexto

Instinct vive en la mensajería: se le escribe por iMessage o WhatsApp y se le
llama por teléfono. Perseo tenía voz en el PC, el panel, la web del móvil por la
VPN de casa, y un Telegram que desde el 2026-08-22 **solo avisa**. Esa decisión
venía con una regla de privacidad: «titular por Telegram, detalle por
Tailscale». Por un tercero solo viajaban recuentos.

Hablarle desde WhatsApp o por teléfono es, por fuerza, mandar la conversación
entera por un tercero: Telegram, Meta o Twilio.

## Decisión

Tres canales nuevos, **los tres apagados de fábrica**, y los tres escriben en el
mismo **hilo principal** (`servicios/hilo.py`) que se ve en el panel y en el
móvil:

- **Telegram de dos sentidos** (`PERSEO_TELEGRAM_CONVERSAR=1`). Solo contesta a
  su chat.
- **WhatsApp y SMS por Twilio.** Hace falta una cuenta y `<datos>/twilio.json`.
- **El teléfono por Twilio.** Se llama al número de Perseo y se habla por
  turnos: Twilio transcribe y dice. Perseo puede llamarle a él, y puede llamar a
  un negocio en su nombre.

Las reglas que los sostienen no dependen del modelo:

1. **Solo el dueño.** Un mensaje o una llamada de otro número o de otro chat no
   llega al hilo ni al modelo.
2. **Lo que llega de Twilio trae su firma** (HMAC con el token de la cuenta), o
   no se atiende.
3. **Twilio entra por un puerto aparte** (`caras/twilio.py`, 8788, solo en
   `127.0.0.1`). Lo que se publica en internet con `tailscale funnel` es ese
   servidor, que solo sabe de `/twilio/…`. El núcleo sigue sin estar expuesto.
4. **Llamar a un tercero es `exterior`** (ADR 0007), y la primera frase de la
   llamada, que pone el código, dice que es una IA que llama en su nombre.
   Obligatorio por el artículo 50 de la ley europea de IA, y aunque no lo fuera.
5. **La ubicación solo llega cuando él la comparte**, y se guarda solo la última.

## Qué se pierde, dicho sin adornos

- Con Telegram de dos sentidos o con WhatsApp encendidos, **la conversación pasa
  por Telegram o por Meta y Twilio**. Correos, citas y lo que haga falta. Por
  eso son interruptores y no lo de fábrica, y por eso está en `docs/PRIVACIDAD.md`.
- El teléfono por turnos es más lento que la voz de la app: medio segundo o un
  segundo entre frases, y «un momento» mientras piensa.
- Publicar el puerto de Twilio es abrir algo a internet. Lo que queda abierto
  comprueba la firma antes de mirar nada, pero sigue siendo un puerto abierto.

## Qué haría falta para cambiarla

Cada canal se apaga solo: sin `PERSEO_TELEGRAM_CONVERSAR`, o sin la cuenta de
Twilio, no arranca. Para quitarlos del todo, basta con borrar sus dos caras
(`caras/telegram_conversa.py` y `caras/twilio.py`); nada más depende de ellas.
