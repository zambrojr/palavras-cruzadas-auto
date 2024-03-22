import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

import json
import re


class geminiModel:

    def __init__(self):
        genai.configure(api_key='CHANGEME!')
        self.client = genai.GenerativeModel('gemini-pro')

    def perguntaModelo(self, questao, assistant):
 
        response = self.client.generate_content("Você foi designado para fornecer resolver um jogo de palvras-cruzadas. Forneça somente respostas curtas. Output sempre em formato de objeto JSON e chave com nome de 'resposta'. "+"Qual a resposta para: "+questao+"."+assistant)

        #print(">>>",response.text,">>>")
        #cv2.waitKey(0) 
        json_objects = []
        pattern = r'\{[^{}]*\}'
        matches = re.finditer(pattern, response.text)
        
        for match in matches:
            print("@@@", match.group(0))
            data_json = json.loads(match.group(0)) 
            json_objects.append(data_json["resposta"])

        #print(json_objects)
        if json_objects:
            return json_objects
        else:
            return []

