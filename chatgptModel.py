from openai import OpenAI
import json


class chatgptModel:

    def __init__(self):
        self.tokenGpt = "CHANGEME!"
        self.modelGpt = "gpt-3.5-turbo-16k-0613"
        self.client = OpenAI()


    def perguntaModelo(self, questao, assistant):
 
        messages=[
            {"role": "system", "content": "Você foi designado para fornecer resolver um jogo de palvras-cruzadas. Forneça somente respostas curtas. Output sempre em formato de array JSON e chave com nome de 'respostas'."},
            {"role": "user", "content": "Qual a resposta para: "+questao+"."+assistant+". Forneça três alternativas."}
        ]
  
        response = self.client.chat.completions.create( model=self.modelGpt, messages=messages )

        print(response)
        #cv2.waitKey(0) 
        data_json = json.loads(response.choices[0].message.content) 
        #print(data_json["resposta"])
        if data_json["respostas"]:
            return data_json["respostas"]
        else:
            return []

