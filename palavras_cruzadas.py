import cv2
import imutils
import numpy as np
import pytesseract
import pandas as pd
import random
import string
from geminiModel import geminiModel
from chatgptModel import chatgptModel


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cv2.waitKey(0)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class palavraCruzada:

    thresh = None
    gray = None
    imggray = None
    pdQuestoes = None
    qtdQuadarntesHoriz = 0
    arrayResposta = None

    def __init__(self, imagem):
        self.img = cv2.imread(imagem, cv2.IMREAD_COLOR)
        self.templateInfhoriz = cv2.imread('./images/infhoriz.jpg', cv2.IMREAD_GRAYSCALE)
        self.templateVerthoriz = cv2.imread('./images/verthoriz.jpg', cv2.IMREAD_GRAYSCALE)
        self.templateHoriz = cv2.imread('./images/horiz.jpg', cv2.IMREAD_GRAYSCALE)
        self.templateVert = cv2.imread('./images/vert.jpg', cv2.IMREAD_GRAYSCALE)
        self.templateInfVert = cv2.imread('./images/infvert.jpg', cv2.IMREAD_GRAYSCALE)
        self.templateHorizVert = cv2.imread('./images/horizvert.jpg', cv2.IMREAD_GRAYSCALE)        
        self.templateHorizbackvert = cv2.imread('./images/horizbackvert.jpg', cv2.IMREAD_GRAYSCALE)    
        
        #MODELO GEMINI
        self.modeloIA = geminiModel()    

        #MODELO CHATGPT
        #self.modeloIA = chatgptModel()    

        self.setCountours()

    def grayImage(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.imggray= self.gray.copy()        

    def tresholdImage(self):
        self.thresh = cv2.threshold(self.gray, 160, 255, cv2.THRESH_BINARY_INV)[1]
        minLineLength=100
        lines = cv2.HoughLinesP(image=self.thresh,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=8)
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(self.gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 0), 8, cv2.LINE_AA)
        self.thresh = cv2.threshold(self.gray, 60, 255, cv2.THRESH_BINARY_INV)[1]  

    def getDicaLetra(self, topx, bottomx, topy, bottomy):
        CroppedLt = self.imggray[topx+30:bottomx-20, topy+30:bottomy-20]
        CroppedLt = cv2.threshold(CroppedLt, 160, 255, cv2.THRESH_BINARY_INV)[1]  
        
        letra = ""
        if len(np.unique(CroppedLt)) > 1:
            CroppedLt = self.imggray[topx+10:bottomx-5, topy+10:bottomy-5]
            letra = pytesseract.image_to_string(CroppedLt, config='--psm 7', lang='por')
            letra = letra.replace("\n","").replace("- ","")
        return letra           

    def getQuadrante(self, c):
        rect = cv2.minAreaRect(c)
        corners = cv2.boxPoints(rect)
        x=corners[:,1].astype(int)
        y=corners[:,0].astype(int)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))         
        Cropped = self.thresh[topx+5:bottomx-5, topy+5:bottomy-5]        

        if len(np.unique(Cropped)) > 1:            
            return {"quest":True, "orientacao":None, "resolvido":False, "qtdLetras":0, "encontradas":0, "tentativas":0, "topx":topx, "topy":topy, "bottomx":bottomx, "bottomy":bottomy, "letra":"", "resposta":None, "alternativasGPT":[], "pertencente":[], "contours":c}           
        else:             
            return {"quest":False, "orientacao":None, "resolvido":False, "qtdLetras":0, "encontradas":0, "tentativas":0, "topx":topx, "topy":topy, "bottomx":bottomx, "bottomy":bottomy, "letra":self.getDicaLetra(topx, bottomx, topy, bottomy), "resposta":None, "alternativasGPT":[], "pertencente":[], "contours":c}    

    def setCountours(self):
        self.grayImage()
        self.tresholdImage()        
        contours = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = contours[::-1]
        #print("contours",len(contours))
        arrQuestoes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 9000 and area < 10000:
                arrQuestoes.append( self.getQuadrante(c) )    
        self.pdQuestoes = pd.DataFrame(arrQuestoes)

        mediaTamanhoQuadrante = self.pdQuestoes['bottomy'].sub(self.pdQuestoes['topy']).mean()
        maxHoriz = self.pdQuestoes.loc[self.pdQuestoes['topy'].idxmax()]
        self.qtdQuadarntesHoriz = int(maxHoriz["topy"] / mediaTamanhoQuadrante)     
        
        self.arrayResposta = np.full((9, self.qtdQuadarntesHoriz)," ")   


    def verificaDisposicaoQuadrante(self, quadrante, ind, template):
        CroppedOcr = self.imggray[quadrante["topx"]-3:quadrante["bottomx"], quadrante["topy"]-3:quadrante["bottomy"]]
        res = cv2.matchTemplate(CroppedOcr,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= 0.8)  
        return len(loc[0]) > 0
    
    def drawLetra(self, indice, letra):
        j = int(indice/self.qtdQuadarntesHoriz)
        i = indice % self.qtdQuadarntesHoriz
        self.arrayResposta[j,i] = letra

    def getQtdQuadrantesResposta(self, ind, indiceHorzVert, questaoPai):
        t = ind
        resposta = []
        self.pdQuestoes.at[questaoPai, "pertencente"] = []

        while not self.pdQuestoes.at[t, "quest"]:
            self.pdQuestoes.at[questaoPai, "pertencente"].append(t)
            resposta.append(self.pdQuestoes.at[t, "letra"])
            if self.pdQuestoes.at[t, "letra"]:
                self.drawLetra(t, self.pdQuestoes.at[t, "letra"])

            t = t+indiceHorzVert
            if t >= len(self.pdQuestoes) or (indiceHorzVert == 1 and self.pdQuestoes.at[t, "topx"] > self.pdQuestoes.at[t-indiceHorzVert, "topx"] + 5):
                break        
        return resposta
    
    def analisaPalavraCruzada(self):        
        for ind in self.pdQuestoes.index:
            obj = self.pdQuestoes.loc[ind]    

            if obj["quest"]:        
                cv2.putText(self.img, str(ind), (obj["bottomy"]-5, obj["bottomx"]-5), cv2.FONT_HERSHEY_PLAIN ,  0.84, (255, 0, 0) , 1, cv2.LINE_AA) 

            elif not obj["quest"]:
                
                if self.verificaDisposicaoQuadrante(obj,  ind, self.templateHorizVert):
                    self.pdQuestoes.at[ind-1, "orientacao"] = "HORIZ-VERT"
                    self.pdQuestoes.at[ind-1, "resposta"]  = self.getQtdQuadrantesResposta(ind, self.qtdQuadarntesHoriz, ind-1)
                    self.pdQuestoes.at[ind-1, "qtdLetras"]  = len(self.pdQuestoes.at[ind-1, "resposta"])                    
                    temp = [i for i,x in enumerate(self.pdQuestoes.at[ind-1, "resposta"]  ) if x]       
                    self.pdQuestoes.at[ind-1, "encontradas"]  = len(temp)
                    self.pdQuestoes.at[ind-1, "resolvido"]  = self.pdQuestoes.at[ind-1, "encontradas"] == self.pdQuestoes.at[ind-1, "qtdLetras"]
                elif self.verificaDisposicaoQuadrante(obj,  ind, self.templateHorizbackvert):
                    self.pdQuestoes.at[ind+1, "orientacao"] = "HORIZ-BACK-VERT"
                    self.pdQuestoes.at[ind+1, "resposta"]  = self.getQtdQuadrantesResposta(ind, self.qtdQuadarntesHoriz, ind+1)
                    self.pdQuestoes.at[ind+1, "qtdLetras"]  = len(self.pdQuestoes.at[ind+1, "resposta"])                    
                    temp = [i for i,x in enumerate(self.pdQuestoes.at[ind+1, "resposta"]  ) if x]       
                    self.pdQuestoes.at[ind+1, "encontradas"]  = len(temp)
                    self.pdQuestoes.at[ind+1, "resolvido"]  = self.pdQuestoes.at[ind+1, "encontradas"] == self.pdQuestoes.at[ind+1, "qtdLetras"]                           
                elif self.verificaDisposicaoQuadrante(obj,  ind, self.templateInfVert):
                    self.pdQuestoes.at[ind-15, "orientacao"] = "DIAG-VERT"
                    self.pdQuestoes.at[ind-15, "resposta"]  = self.getQtdQuadrantesResposta(ind, self.qtdQuadarntesHoriz, ind-15)
                    self.pdQuestoes.at[ind-15, "qtdLetras"]  = len(self.pdQuestoes.at[ind-15, "resposta"])                    
                    temp = [i for i,x in enumerate(self.pdQuestoes.at[ind-15, "resposta"]  ) if x]       
                    self.pdQuestoes.at[ind-15, "encontradas"]  = len(temp)
                    self.pdQuestoes.at[ind-15, "resolvido"]  = self.pdQuestoes.at[ind-15, "encontradas"] == self.pdQuestoes.at[ind-15, "qtdLetras"]
                elif self.verificaDisposicaoQuadrante(obj,  ind, self.templateVert):
                    self.pdQuestoes.at[ind-14, "orientacao"] = "VERT"
                    self.pdQuestoes.at[ind-14, "resposta"]  = self.getQtdQuadrantesResposta(ind, self.qtdQuadarntesHoriz, ind-14)
                    self.pdQuestoes.at[ind-14, "qtdLetras"]  = len(self.pdQuestoes.at[ind-14, "resposta"])                    
                    temp = [i for i,x in enumerate(self.pdQuestoes.at[ind-14, "resposta"]  ) if x]       
                    self.pdQuestoes.at[ind-14, "encontradas"]  = len(temp)
                    self.pdQuestoes.at[ind-14, "resolvido"]  = self.pdQuestoes.at[ind-14, "encontradas"] == self.pdQuestoes.at[ind-14, "qtdLetras"]
            

                if self.verificaDisposicaoQuadrante(obj,  ind, self.templateVerthoriz):
                    self.pdQuestoes.at[ind-15, "orientacao"] = "DIAG-HORIZ"
                    self.pdQuestoes.at[ind-15, "resposta"]  = self.getQtdQuadrantesResposta(ind, 1, ind-15)
                    self.pdQuestoes.at[ind-15, "qtdLetras"]  = len(self.pdQuestoes.at[ind-15, "resposta"])                    
                    temp = [i for i,x in enumerate(self.pdQuestoes.at[ind-15, "resposta"]  ) if x]       
                    self.pdQuestoes.at[ind-15, "encontradas"]  = len(temp)
                    self.pdQuestoes.at[ind-15, "resolvido"]  = self.pdQuestoes.at[ind-15, "encontradas"] == self.pdQuestoes.at[ind-15, "qtdLetras"]
                elif self.verificaDisposicaoQuadrante(obj,  ind, self.templateInfhoriz):
                    self.pdQuestoes.at[ind-14, "orientacao"] = "VERT-HORIZ"
                    self.pdQuestoes.at[ind-14, "resposta"]  = self.getQtdQuadrantesResposta(ind, 1, ind-14)
                    self.pdQuestoes.at[ind-14, "qtdLetras"]  = len(self.pdQuestoes.at[ind-14, "resposta"])                    
                    temp = [i for i,x in enumerate(self.pdQuestoes.at[ind-14, "resposta"]  ) if x]       
                    self.pdQuestoes.at[ind-14, "encontradas"]  = len(temp)
                    self.pdQuestoes.at[ind-14, "resolvido"]  = self.pdQuestoes.at[ind-14, "encontradas"] == self.pdQuestoes.at[ind-14, "qtdLetras"]
                elif self.verificaDisposicaoQuadrante(obj,  ind, self.templateHoriz):
                    self.pdQuestoes.at[ind-1, "orientacao"]  = "HORIZ"
                    self.pdQuestoes.at[ind-1, "resposta"]  = self.getQtdQuadrantesResposta(ind, 1, ind-1)
                    self.pdQuestoes.at[ind-1, "qtdLetras"]  = len(self.pdQuestoes.at[ind-1, "resposta"])             
                    temp = [i for i,x in enumerate(self.pdQuestoes.at[ind-1, "resposta"]  ) if x]       
                    self.pdQuestoes.at[ind-1, "encontradas"]  = len(temp)
                    self.pdQuestoes.at[ind-1, "resolvido"]  = self.pdQuestoes.at[ind-1, "encontradas"] == self.pdQuestoes.at[ind-1, "qtdLetras"]

        #questoesOrdenadas = self.pdQuestoes.query('quest == True & resolvido == False')
        #print(questoesOrdenadas) 
        #cv2.imshow('car', cv2.resize(self.img, (937, 607)))      
        #cv2.waitKey(0)


    def get_random_string(self, length, ind):
        letters = string.ascii_lowercase

        if ind == 0:
            return ["oscarwilde", "dwqdwq","321321"]
        elif ind == 4:
            return ["gwen", "dwqdwq","321321"]
        elif ind == 32:
            return ["endossado", "dwqdwq","321321"]
        elif ind == 6:
            return ["infix", "dwqdwq","321321"]
        elif ind == 7:
            return ["ldl", "3dsfds","fsdfsd"]
        elif ind == 8:
            return ["do", "3dsfds","fsdfsd"]
        elif ind == 42:
            return ["arsenal", "3dsfds","fsdfsd"]
        elif ind == 10:
            return ["desejos", "3dsfds","fsdfsd"]
        
        result_str = ''.join(random.choice(letters) for i in range(length))
        return [result_str]
    
    def getPrimeiroQuadranteQuestao(self, orientacao):
        if orientacao == "HORIZ" or orientacao == "HORIZ-VERT":
            return 1
        elif orientacao == "DIAG-HORIZ" or orientacao == "DIAG-VERT":
            return 15
        elif orientacao == "VERT-HORIZ" or orientacao == "VERT":
            return 14
        elif orientacao == "HORIZ-BACK-VERT":
            return -1

    def getPassosOrientacaoQuestao(self, orientacao):
        if orientacao == "HORIZ" or orientacao == "DIAG-HORIZ" or orientacao == "VERT-HORIZ":
            return 1
        else: return 14

    def limpaQuestao(self, indQuestao):
        self.pdQuestoes.at[indQuestao, "resolvido"] = False
        self.pdQuestoes.at[indQuestao, "alternativasGPT"] = []

        passos = self.getPassosOrientacaoQuestao(self.pdQuestoes.at[indQuestao, "orientacao"])
        ind = indQuestao + self.getPrimeiroQuadranteQuestao(self.pdQuestoes.at[indQuestao, "orientacao"])
        
        while ind <= len(self.pdQuestoes) and not self.pdQuestoes.at[ind, "quest"]:
            questaoOrig = self.getQuestaoAdjacente(indQuestao, ind)
            if questaoOrig != None and not self.pdQuestoes.at[questaoOrig, "resolvido"]:
                obj = self.pdQuestoes.loc[ind]
                cv2.putText(self.img, obj["letra"], (obj["topy"]+20, obj["topx"]+45), cv2.FONT_HERSHEY_DUPLEX ,  1.3, (255, 255, 255) , 2, cv2.LINE_AA) 
                self.pdQuestoes.at[ind, "letra"] = ""
                self.drawLetra(ind, "")
            ind += passos


    def getQuestaoAdjacente(self, indQuestao, indQuadranteAtual):
        mask = self.pdQuestoes.pertencente.apply(lambda x: indQuadranteAtual in x)
        df1 = self.pdQuestoes[mask]
        df_filtered = df1[df1.index != indQuestao]
        #print ("adjacente para",indQuadranteAtual)
        if len(df_filtered.index.values)>0:
            #print ("> ",df_filtered.index.values[0])
            return df_filtered.index.values[0]
        return None

    def analisaQuestaoAdjacente(self, indQuestao, indQuadranteAtual):
        
        indQuadranteAtual = self.getQuestaoAdjacente(indQuestao, indQuadranteAtual)

        if indQuadranteAtual != None and not self.pdQuestoes.at[indQuadranteAtual, "resolvido"] and len(self.pdQuestoes.at[indQuadranteAtual, "alternativasGPT"]) > 0:
            print("Adjacente:",indQuadranteAtual," possibi", self.pdQuestoes.at[indQuadranteAtual, "alternativasGPT"])

            passos = self.getPassosOrientacaoQuestao(self.pdQuestoes.at[indQuadranteAtual, "orientacao"])
            tind = indQuadranteAtual + self.getPrimeiroQuadranteQuestao(self.pdQuestoes.at[indQuadranteAtual, "orientacao"])

            for respostaAlternativa in self.pdQuestoes.at[indQuadranteAtual, "alternativasGPT"]:
                print(respostaAlternativa)
            
            cv2.waitKey(0)

            respostaAlternativa = self.pdQuestoes.at[indQuadranteAtual, "alternativasGPT"][0]

            for element in respostaAlternativa.upper():
                obj = self.pdQuestoes.loc[tind]
                if obj["letra"] != "" and obj["letra"] != element:                    
                    qustaoErrada = self.getQuestaoAdjacente(indQuadranteAtual, tind)
                    print(f"{bcolors.FAIL}Resposta não bateu{bcolors.ENDC}", "Questao Errada:", qustaoErrada)
                    self.limpaQuestao(qustaoErrada)
                tind += passos

            print("Renderizando ",respostaAlternativa, " em ", indQuadranteAtual)
            self.renderResposta(indQuadranteAtual, respostaAlternativa)
            cv2.imshow('car', cv2.resize(self.img, (937, 607)))            
            cv2.waitKey(0)


    def renderResposta(self, indOriginal, resposta):

        passos = self.getPassosOrientacaoQuestao(self.pdQuestoes.at[indOriginal, "orientacao"])
        ind = indOriginal + self.getPrimeiroQuadranteQuestao(self.pdQuestoes.at[indOriginal, "orientacao"])
        objOriginal = self.pdQuestoes.loc[indOriginal]
        pos = 0

        tind = ind
        diferente = []
        for element in resposta.upper():
            obj = self.pdQuestoes.loc[tind]
            if obj["letra"] != "" and obj["letra"] != element:
                diferente.append(tind)
            tind += passos

        print("Diferente:",diferente)

        
        if len(diferente) > 0:            
            print(f"{bcolors.FAIL}Resposta não bateu{bcolors.ENDC}")
            return False
            '''for quadranteDiferentes in diferente:
                qustaoErrada = self.getQuestaoAdjacente(indOriginal, quadranteDiferentes)
                if qustaoErrada != None:
                    print(f"{bcolors.FAIL}Limpando a questao ", qustaoErrada, f"para nova tentativa{bcolors.ENDC}")
                    self.limpaQuestao(qustaoErrada)
            return False'''
        
        for element in resposta.upper():
            obj = self.pdQuestoes.loc[ind]
            if obj["letra"] == "":
                self.pdQuestoes.at[ind, "letra"] = element
                cv2.putText(self.img, element, (obj["topy"]+20, obj["topx"]+45), cv2.FONT_HERSHEY_DUPLEX ,  1.3, (128, 128, 0) , 2, cv2.LINE_AA) 
                objOriginal["resposta"][pos] = element
                self.drawLetra(ind, element)
                #self.analisaQuestaoAdjacente(indOriginal, ind)
            pos = pos + 1
            ind += passos
        
        self.pdQuestoes.at[indOriginal, "encontradas"] = self.pdQuestoes.at[indOriginal, "qtdLetras"]
        self.pdQuestoes.at[indOriginal, "resolvido"] = True     

        return True
    
    def perguntaModeloIA(self, obj, ind):
        CroppedOcr = self.imggray[obj["topx"]-2:obj["bottomx"]+2, obj["topy"]-2:obj["bottomy"]+2]
        CroppedOcr = cv2.resize(CroppedOcr, (300, 300))
        CroppedOcr = cv2.threshold(CroppedOcr, 127, 255, cv2.THRESH_BINARY)[1]
        
        text = pytesseract.image_to_string(CroppedOcr, lang='por') 
        pergunta = text.replace("\n"," ").replace("- ","").replace("(...)","?")
        assistant  = " Dica: A resposta correta possui exatamente " + str(obj["qtdLetras"]) + " letras"

        lacunas = ""
        il = 1
        for letra in obj["resposta"]:    
            if letra != "":
                lacunas = lacunas+" letra \""+letra+"\" na posição "+str(il)+","
            il += 1

        if lacunas != "":
            assistant = assistant + " e que possua "+lacunas.rstrip(",")
        else:
            assistant = assistant + "."
            
        print(f"{bcolors.OKBLUE}Qual a resposta para: "+pergunta+"."+assistant+f"{bcolors.ENDC}")

        try:
            return self.modeloIA.perguntaModelo(pergunta, assistant)
        except Exception as e:
            print(f"{bcolors.WARNING}: {e} {bcolors.ENDC}")
            cv2.waitKey(50000)
            return self.modeloIA.perguntaModelo(pergunta, assistant)
        pass           


    def parser(self):
        cv2.imshow('thresh', self.thresh)

        questoesOrdenadas = self.pdQuestoes.query('quest == True & resolvido == False').sort_values(['tentativas', 'encontradas','qtdLetras'], ascending=[True, False, False])
        print(questoesOrdenadas)       
        print(self.arrayResposta)
        
        for ind in questoesOrdenadas.index:    
            obj = self.pdQuestoes.loc[ind]    
            cv2.drawContours(self.img, [obj["contours"]], -1, (128, 128, 0), 2)
            self.pdQuestoes.at[ind, "tentativas"] = obj["tentativas"]+1

            data_json = self.perguntaModeloIA(obj, ind)
            self.pdQuestoes.at[ind, "alternativasGPT"] = []
            
            for alternativa in data_json:
                resposta = alternativa.replace(" ", "")
                if len(resposta) == obj["qtdLetras"]:
                    print(f"{bcolors.OKGREEN}Válido em qtd de letras para a alternativa "+resposta+f"{bcolors.ENDC}")
                    self.pdQuestoes.at[ind, "alternativasGPT"].append(resposta)
                else:
                    print(f"{bcolors.FAIL}Invalido para a alternativa ",alternativa," para ",self.pdQuestoes.at[ind,"qtdLetras"], f"qtd letras{bcolors.ENDC}")
            
            if len(self.pdQuestoes.at[ind, "alternativasGPT"]) > 0:
                alternativa = self.pdQuestoes.at[ind, "alternativasGPT"][0]
                resposta = alternativa.replace(" ", "")
                if self.renderResposta(ind, resposta):
                    self.analisaPalavraCruzada()
                    cv2.imshow('car', cv2.resize(self.img, (937, 607)))            
                    cv2.waitKey(2000)                           
                    return self.parser()

            cv2.imshow('car', cv2.resize(self.img, (937, 607)))            
            cv2.waitKey(2000)        



#https://cruzadasclube.com.br/jogo/i/id/3603/cruzadas-diretas-796/
resolucao = palavraCruzada('./images/grid2.png')
resolucao.analisaPalavraCruzada()
resolucao.parser()


print("FIM")    
cv2.waitKey(0)
cv2.destroyAllWindows()