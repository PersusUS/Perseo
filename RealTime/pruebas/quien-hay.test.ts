/**
 * Quién es el dueño y quién es visita.
 *
 * Es el arreglo de la llamada del 2026-08-25: la cámara reconoció al padre del
 * señor Persus como «Desconocido 1», Perseo siguió llamándole «señor Persus» y
 * además se atribuyó a él sus propios gustos. Un fallo aquí no revienta la
 * aplicación — se ve como Perseo tratando a un invitado de dueño de la casa y
 * contándole delante la agenda ajena, que es exactamente lo que no puede pasar.
 */

import { describe, expect, it } from 'vitest';

import {
  avisoCaras,
  avisoHablante,
  bloqueCenso,
  esElSenor,
  esProvisional,
  etiquetaPersona,
  PERFIL_PERSUS_POR_DEFECTO,
  sinAvisoDeIdentidad,
} from '../src/lib/quien-hay';

const SENOR = PERFIL_PERSUS_POR_DEFECTO;

describe('esProvisional', () => {
  it('reconoce las etiquetas que pone el núcleo solo', () => {
    expect(esProvisional('Desconocido 1')).toBe(true);
    expect(esProvisional('desconocido 12')).toBe(true);
  });

  it('no confunde a una persona que se llame parecido', () => {
    expect(esProvisional('Desconocido')).toBe(false);
    expect(esProvisional('Javi')).toBe(false);
  });
});

describe('esElSenor', () => {
  it('acepta el perfil configurado', () => {
    expect(esElSenor('Persus', SENOR)).toBe(true);
    expect(esElSenor('Jesús', SENOR)).toBe(true);
  });

  it('acepta el nombre aunque el ajuste apunte a otro perfil', () => {
    // El perfil puede haberse enrolado como «Jesús» y el ajuste seguir en
    // «Persus»: tratarle de visita por una letra sería peor que suponer.
    expect(esElSenor('jesus perez bazarot', 'Persus')).toBe(true);
  });

  it('niega a cualquier otro, y a los provisionales', () => {
    expect(esElSenor('Antonio', SENOR)).toBe(false);
    expect(esElSenor('Desconocido 1', SENOR)).toBe(false);
    expect(esElSenor(null, SENOR)).toBe(false);
  });
});

describe('etiquetaPersona', () => {
  it('al dueño le llama por su trato', () => {
    expect(etiquetaPersona('Persus', SENOR)).toBe('el señor Persus');
  });

  it('a los demás les marca que NO lo son', () => {
    expect(etiquetaPersona('Antonio', SENOR)).toContain('NO es el señor Persus');
    expect(etiquetaPersona('Desconocido 2', SENOR)).toContain('etiqueta provisional');
  });
});

describe('avisoHablante', () => {
  it('con el dueño no da instrucciones de más', () => {
    const aviso = avisoHablante('Persus', SENOR);
    expect(aviso).toContain('es el señor Persus');
    expect(aviso).not.toContain('nombrar_persona');
  });

  it('con un desconocido manda preguntar el nombre y guardarlo', () => {
    const aviso = avisoHablante('Desconocido 1', SENOR);
    expect(aviso).toContain('NO es el señor Persus');
    expect(aviso).toContain("etiqueta='Desconocido 1'");
    expect(aviso).toContain('nombrar_persona');
  });

  it('con un conocido que no es el dueño manda leer su nota', () => {
    const aviso = avisoHablante('Antonio', SENOR);
    expect(aviso).toContain('Antonio');
    expect(aviso).toContain('Perseo/Personas');
    expect(aviso).toContain('NO es el señor Persus');
  });
});

describe('avisoCaras', () => {
  it('sin nadie identificado no dice nada', () => {
    expect(avisoCaras([], SENOR)).toBeNull();
  });

  it('con el dueño solo, no avisa de visitas', () => {
    const aviso = avisoCaras(['Persus'], SENOR);
    expect(aviso).toContain('el señor Persus');
    expect(aviso).not.toContain('nombrar_persona');
  });

  it('con una visita delante, prohíbe el trato y lo privado', () => {
    const aviso = avisoCaras(['Persus', 'Desconocido 3'], SENOR) ?? '';
    expect(aviso).toContain('Desconocido 3');
    expect(aviso).toContain('no le llames así');
    expect(aviso).toContain('privado');
  });
});

describe('bloqueCenso', () => {
  it('sin perfiles no ocupa sitio en las instrucciones', () => {
    expect(bloqueCenso([], SENOR)).toBeNull();
  });

  it('separa al dueño, a los conocidos y a los que esperan nombre', () => {
    const censo =
      bloqueCenso(
        [
          { nombre: 'Persus', voz: true, caras: 2 },
          { nombre: 'Antonio', voz: false, caras: 1 },
          { nombre: 'Desconocido 4', voz: true, caras: 0 },
        ],
        SENOR,
      ) ?? '';
    expect(censo).toContain('Persus — ES EL SEÑOR PERSUS (voz y cara)');
    expect(censo).toContain('Antonio — NO es el señor Persus (cara)');
    expect(censo).toContain('Desconocido 4 — alguien a quien reconoces');
  });
});

describe('sinAvisoDeIdentidad', () => {
  it('quita el eco recortado sin comerse el saludo', () => {
    // La llamada del 2026-08-26, tal cual salió en pantalla.
    expect(sinAvisoDeIdentidad('[IDENTIDAD] Persus Buenos días, señor Persus.')).toBe(
      'Buenos días, señor Persus.',
    );
  });

  it('quita el aviso leído entero cuando se le pasa el que se mandó', () => {
    const aviso = avisoHablante('Persus', SENOR);
    expect(sinAvisoDeIdentidad(`${aviso} Buenos días.`, [aviso])).toBe('Buenos días.');
  });

  it('quita también el aviso leído sin su marca', () => {
    const aviso = avisoHablante('Antonio', SENOR);
    const leido = `${aviso.replace('[IDENTIDAD] ', '')} Buenas tardes, Antonio.`;
    expect(sinAvisoDeIdentidad(leido, [aviso])).toBe('Buenas tardes, Antonio.');
  });

  it('quita el aviso de caras', () => {
    const leido = '[IDENTIDAD] Delante de la cámara: el señor Persus. Aquí estoy.';
    expect(sinAvisoDeIdentidad(leido)).toBe('Aquí estoy.');
  });

  it('no toca una frase sin marca, ni la relimpia dos veces', () => {
    const dicho = 'Persus Buenos días, señor Persus.';
    expect(sinAvisoDeIdentidad(dicho)).toBe(dicho);
    expect(sinAvisoDeIdentidad(sinAvisoDeIdentidad('[IDENTIDAD] Persus Buenos días.'))).toBe(
      'Buenos días.',
    );
  });

  it('aguanta la marca partida entre fragmentos, limpiando el texto entero', () => {
    // Lo que hace App.tsx: acumular y volver a limpiar.
    let texto = '';
    for (const trozo of ['[IDENTI', 'DAD] Quien habla ahora es el señor Persus. ', 'Dígame.']) {
      texto = sinAvisoDeIdentidad(texto + trozo);
    }
    expect(texto).toBe('Dígame.');
  });
});
