import unittest
from algoritm_profound import process_question

class TestDiscussionAI(unittest.TestCase):

    def test_process_question_from_terminal(self):
        question = input("Por favor, introduce tu pregunta: ")
        
        response, improvement_count = process_question(question)

        print(f"Respuesta final: {response}")
        print(f"Iteraciones de mejora: {improvement_count}")

        self.assertTrue(isinstance(response, str))
        self.assertTrue(improvement_count >= 0)


if __name__ == '__main__':
    unittest.main()
