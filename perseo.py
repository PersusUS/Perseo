import unittest
from algoritm_profound import process_question

class TestDiscussionAI(unittest.TestCase):

    def test_process_question_from_terminal(self):
        """Prueba el proceso completo recibiendo la pregunta del usuario desde la terminal."""
        # Solicitar la pregunta desde la terminal
        question = input("Por favor, introduce tu pregunta: ")

        # Procesar la pregunta
        response, improvement_count = process_question(question)

        # Mostrar el resultado y el número de iteraciones
        print(f"Respuesta final: {response}")
        print(f"Iteraciones de mejora: {improvement_count}")

        # Asegurar que se obtuvo una respuesta válida
        self.assertTrue(isinstance(response, str))
        self.assertTrue(improvement_count >= 0)


if __name__ == '__main__':
    unittest.main()
