import json
import os
import nltk
import spacy
import re
import stanfordnlp
import xml.etree.ElementTree as et
from xml.etree import cElementTree as ET
import numpy as np
import warnings
import math
import gensim
from gensim.models import KeyedVectors
import Util
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from LoadModels import LoadModels
from transformers import AutoModel, AutoTokenizer
import bert
import torch

class ExtractFeatures:
    def __init__(self, stanfordModel, embeddings, spacy, corpus = ""):
        self.corpus = corpus
        if stanfordModel != "" and embeddings != "" and spacy != "":
            self.stanfordModel = stanfordModel
            self.sp = spacy
            self.embeddings = embeddings
    
    warnings.filterwarnings("ignore")

    def readXml(self):
        base_path = os.path.dirname(os.path.realpath(__file__))

        xml_file = os.path.join(base_path, self.corpus)

        tree = et.parse(xml_file)

        root = tree.getroot()
        # arq = open('sentencas',"w")
        lfrase = []
        entailment = 0
        none = 0
        paraphrase = 0
        for child in root:
            for  element in child:
                dic = {'id':child.attrib['id'], 'entailment':child.attrib['entailment'],'similarity':child.attrib['similarity'], 'text':element.text}
                #print(element.tag, ";", element.text)
                #print(element.text)
                #print(child.attrib['id'], element.text)
                # arq.write(element.text + ' >>C'+child.attrib['id']+'\n')
                lfrase.append(dic)
                if child.attrib['entailment'] == 'None':
                    none += 1
                elif child.attrib['entailment'] == 'Entailment':
                    entailment += 1
                else:
                    paraphrase += 1
        print(none/2, entailment/2, paraphrase/2)
        return lfrase
    
    def listaSinonimos(self):
        db_tep = open("./assets/databases/base_tep2.txt", "r")
        l_sinonimos = []
        for i in db_tep:
            data = ''.join(i).lower()
            data = re.sub(r'["-,.:@#?!&$\{\}\[\]]', '', data)
            text = data.split(' ')
            size = len(text)
            text[size-1] = text[size-1].split('\n')[0]
            l = []
            for i in range(2, len(text)):
                l.append(text[i])
            text = {"id":text[0],"tipo":text[1],"palavras":l}
            l_sinonimos.append(text)
        db_tep.close()
        
        return l_sinonimos

    def anotacaoDependencia(self, s1, s2):
        i = 0
        ant = -1
        
        dep1 = self.stanfordModel(s1['text'])
        dep2 = self.stanfordModel(s2['text'])
        tks1 = []
        tks2 = []
        dic = {}
        ldic = []
        ll = []
        for sent in dep1.sentences:
            for word in sent.words:
                tks1.append(word.lemma)
                if word.governor > 0:
                    try:
                        dic = {"index":word.index, "text":word.lemma, "governorIndex": word.governor,"governor":dep1.sentences[0].words[word.governor-1].lemma , "rel":word.dependency_relation}
                    except:
                        dic = {"index":word.index, "text":word.lemma, "governorIndex": word.governor,"governor":dep1.sentences[0].words[0].lemma , "rel":word.dependency_relation}
                    
                else:
                    dic = {"index":word.index, "text":word.lemma, "governorIndex": word.governor,"governor":'root' , "rel":word.dependency_relation}
                ll.append(dic)
        ldic.append({"token":tks1, "dependency":ll})
        ll = []
        for sent in dep2.sentences:
            for word in sent.words:
                tks2.append(word.lemma)
                if word.governor > 0:
                    try:
                        dic = {"index":word.index, "text":word.lemma, "governorIndex": word.governor,"governor":dep2.sentences[0].words[word.governor-1].lemma , "rel":word.dependency_relation}
                    except:
                        dic = {"index":word.index, "text":word.lemma, "governorIndex": word.governor,"governor":dep2.sentences[0].words[0].lemma , "rel":word.dependency_relation}
                else:
                    dic = {"index":word.index, "text":word.lemma, "governorIndex": word.governor,"governor":'root' , "rel":word.dependency_relation}
                ll.append(dic)
        ldic.append({"token":tks2, "dependency":ll})
        return {"id":s1['id'], "sent":ldic}
        
    def dependenceParsySpacy(self, frase, nlp):
        doc = nlp(frase)
        ldep = []
        for token in doc:
            ldep.append({"text":token.text, "rel":token.dep_, "children":[child for child in token.children]})
        return ldep
    
    def isSinonimos(self, palavra1, palavra2, sinonimos):
        palavra1.lower()
        palavra2.lower()
        doc = self.stanfordModel(palavra1)
        doc2 = self.stanfordModel(palavra2)
        pal1 = {}
        pal2 = {}
        for sent in doc.sentences:
            for word in sent.words:
                pal1 = {'palavra':word.lemma, 'type':word.upos}

        for sent in doc2.sentences:
            for word in sent.words:
                pal2 = {'palavra':word.lemma, 'type':word.upos}
        if pal1['type'] == pal2['type'] and pal1['palavra'] != pal2['palavra'] and pal1['type'] != "NUM":
            for i in sinonimos:
                if pal1['type'] == 'VERB':
                    if i['tipo'] == 'verbo':
                        cont = 0
                        for pal in i['palavras']:
                            if pal == pal1['palavra'].lower() or pal == pal2['palavra'].lower():
                                cont += 1
                        if cont == 2:
                            return True
                elif pal1['type'] == 'NOUN':
                    if i['tipo'] == 'substantivo':
                        cont = 0
                        for pal in i['palavras']:
                            if pal == pal1['palavra'].lower() or pal == pal2['palavra'].lower():
                                cont += 1
                        if cont == 2:
                            return True
                elif pal1['type'] == 'ADJ':
                    if i['tipo'] == 'adjetivo':
                        cont = 0
                        for pal in i['palavras']:
                            if pal == pal1['palavra'].lower() or pal == pal2['palavra'].lower():
                                cont += 1
                            if cont == 2:
                                return True
                else:
                    cont = 0
                    if i['tipo'] == 'advérbio':
                        for pal in i['palavras']:
                            if pal == pal1['palavra'].lower() or pal == pal2['palavra'].lower():
                                cont += 1
                            if cont == 2:
                                return True
        return False
    
    def isSinonimosPapel(self, p1, p2, sinonimos):
        doc = self.stanfordModel(p1)
        doc1 = self.stanfordModel(p2)
        p1t = ""
        p2t = ""
        for s in doc.sentences:
            for t in s.words:
                p1t = t.upos
                
        for s in doc1.sentences:
            for t in s.words:
                p2t = t.upos
                
        for i in sinonimos:
            if p1t == p2t and p1t == i['tipo']:
                if p1 == i['palavra'] or p2 == i['palavra']:
                    for k in i['sinonimos']:
                        if k == p2 or k == p1:
                            return True
                    return False
            elif p1t != p2t:
                return False
        
        return False
            
    def isAntonimo(self, palavra1, palavra2):
        lantonimos = Util.readJson("./assets/antonimos.json")
        palavra1.lower()
        palavra2.lower()
        doc = self.stanfordModel(palavra1)
        doc2 = self.stanfordModel(palavra2)
        pal1 = {}
        pal2 = {}
        for sent in doc.sentences:
            for word in sent.words:
                pal1 = {'palavra':word.lemma, 'type':word.upos}
        for sent in doc2.sentences:
            for word in sent.words:
                pal2 = {'palavra':word.lemma, 'type':word.upos}
        if pal1['type'] == pal2['type'] and pal1['palavra'] != pal2['palavra'] and pal1['type'] != "NUM":
            for i in lantonimos:
                if pal1['type'] == 'VERB':
                    if i['type'] == '[Verbo]' and (i['palavra'] == pal1['palavra'] or i['palavra'] == pal2['palavra']):
                        for k in i['antonimos']:
                            if (k == pal2['palavra'] and i['palavra'] == pal1['palavra']) or (k == pal1['palavra'] and i['palavra'] == pal2['palavra']):
                                return True
                elif pal1['type'] == 'NOUN' and (i['palavra'] == pal1['palavra'] or i['palavra'] == pal2['palavra']):
                    if i['type'] == '[Substantivo]':
                        for k in i['antonimos']:
                            if (k == pal2['palavra'] and i['palavra'] == pal1['palavra']) or (k == pal1['palavra'] and i['palavra'] == pal2['palavra']):
                                return True
                elif pal1['type'] == 'ADJ' and (i['palavra'] == pal1['palavra'] or i['palavra'] == pal2['palavra']):
                    if i['type'] == '[Adjetivo]' and (i['palavra'] == pal1['palavra'] or i['palavra'] == pal2['palavra']):
                        for k in i['antonimos']:
                            if (k == pal2['palavra'] and i['palavra'] == pal1['palavra']) or (k == pal1['palavra'] and i['palavra'] == pal2['palavra']):
                                return True
                else:
                    if i['type'] == '[Advérbio]' and (i['palavra'] == pal1['palavra'] or i['palavra'] == pal2['palavra']):
                        for k in i['antonimos']:
                            if (k == pal2['palavra'] and i['palavra'] == pal1['palavra']) or (k == pal1['palavra'] and i['palavra'] == pal2['palavra']):
                                return True
        return False
    
    def isAntonimosPapel(self, p1, p2):
        hipe = Util.readJson("./assets/antonimos.json")
        doc = self.stanfordModel(p1)
        doc1 = self.stanfordModel(p2)
        p1t = ""
        p2t = ""
        for s in doc.sentences:
            for t in s.words:
                p1t = t.upos
                
        for s in doc1.sentences:
            for t in s.words:
                p2t = t.upos
        for i in hipe:
            if p1t == p2t and p1t == i['tipo']:
                #print(i['palavra'], p1, p2)
                if (p1 == i['palavra'] or p2 == i['palavra']):
                    for k in i['antonimos']:
                        if k == p1 or k == p2:
                            return True
            elif p1t != p2t:
                return False
      
    def removeStopWords(self, text):
        #Remove stopwords
        stopwords = nltk.corpus.stopwords.words('portuguese')
        flag = False
        for i in text:
            for s in stopwords:
                if i == s:
                    flag = True
                    break
            if flag:
                text.remove(i)
                flag = False
        return text
    
    def preProcess(self, sentenca1, sentenca2):
        token = []
        token2 = []
        doc = self.stanfordModel(sentenca1)
        for sent in doc.sentences:
            for word in sent.words:
                token.append(word.lemma)
        token = self.removeStopWords(token)
        
        doc = self.stanfordModel(sentenca2)
        for sent in doc.sentences:
            for word in sent.words:
                token2.append(word.lemma)
        token2 = self.removeStopWords(token2)
        
        return {"vec1":token, "vec2":token2}

    def lcsr(self, s1, s2):
        maior = max(len(s1), len(s2))
        s1 = list(s1)
        s2 = list(s2)
        ncomum = 0
        aux = s1+s2
        aux.sort()
        for i in range(0,len(aux)):
                try:
                    if aux[i] == aux[i+1]:
                        aux.remove(aux[i+1])
                except:
                    pass
        flag = 0
        for i in aux:
            for k in s1:
                if i == k:
                    flag+= 1
                    break
            for k in s2:
                if i == k:
                    flag+=1
                    break
            if flag > 1:
                ncomum += 1
            
            flag = 0
        
        mlcsr = ncomum / maior
        
        return mlcsr
    
    def sepOcorrencias(self, token, token2):
        ntoken = token + token2
        ntoken.sort()
        for i in range(0,len(ntoken)):
            try:
                if ntoken[i] == ntoken[i+1]:
                    ntoken.remove(ntoken[i+1])
            except:
                pass
        tk1 = []
        tk2 = []
        for i in ntoken:
            flag = 0
            for k in token:
                if i == k:
                    flag = 1
                    break
            if flag == 1:
                tk1.append({"pal":i, "freq":flag})
            else:
                tk1.append({"pal":0, "freq":flag})
            flag = 0
            for k in token2:
                if i == k:
                    flag = 1
                    break
            if flag == 1:
                tk2.append({"pal":i, "freq":flag})
            else:
                tk2.append({"pal":0, "freq":flag})
        return {"vec1":tk1,"vec2":tk2}

    def simCosseno(self, vec1, vec2):
        sim = 0
        somac = 0
        somabe = 0
        somabd = 0
        for i in range(0,len(vec1)):
            somac += vec1[i]*vec2[i]
            somabe += vec1[i]**2
            somabd += vec2[i]**2
        raize = math.sqrt(somabe)
        raizd = math.sqrt(somabd)
        sim = somac/(raize*raizd)
        
        return sim
            
    def cosWordEmbeddings(self, vec1, vec2, model):
        vec1Emb = []
        vec2Emb = []
        vec1.sort()
        vec2.sort()
        freq = self.sepOcorrencias(vec1, vec2)
        vec1.clear()
        vec2.clear()
        for i in range(0,len(freq['vec1'])):
            vec1.append(freq['vec1'][i]['pal'])
            vec2.append(freq['vec2'][i]['pal'])

        for i in range(0, len(vec1)):
            if vec1[i] == 0:
                    vec1Emb.append(0)
            else:
                try:
                    emb = model[vec1[i].lower()]
                    vec1Emb.append((sum(emb)/len(emb)))
                except:
                    vec2Emb.append(0)
                    pass
            if vec2[i] == 0:
                vec2Emb.append(0)
            else:
                try:
                    emb = model[vec2[i].lower()]
                    vec2Emb.append((sum(emb)/len(emb)))
                except:
                    vec2Emb.append(0)
                    pass
        cos = self.simCosseno(vec1Emb, vec2Emb)
        wmd = model.wmdistance(vec1, vec2)
        return {"wmd":wmd, "cos":cos}

    def verboPrinciapalIgual(self, vec, sinonimos):
        isRoot = False
        for i in vec[0]['dependency']:
            if i['rel'] == 'root':
                for j in vec[1]['dependency']:
                    if j['rel'] == 'root' and (i['text'] == j['text'] or self.isSinonimos(i['text'],j['text'], sinonimos)):
                        isRoot = True
                        break
        return isRoot
    
    def getSubj(self, vec):
        nsubj = ""
        root = ""
        lobj = []
        obj = ""
        for i in vec['dependency']:
            if i['rel'] == 'nsubj' or i['rel'] == 'nsubj:pass':
                nsubj += i['text'] + " "
            elif i['rel'] == 'flat:name' and i['governor'] == nsubj:
                nsubj += i['text'] + " "
            elif i['rel'] == 'root':
                root += i['text']
            elif i['rel'] == 'obj':
                lobj.append({"text":i['text'],"governor":i['governor']})
            
        for i in lobj:
            if i['governor'] == root:
                obj = i

        return {"root":root, "subj":nsubj, "obj":obj}
            
    def countNeg(self, frase):
        frase = frase.lower()
        token = frase.split(' ')
        qtd = 0
        negacao = ["jamais", "de modo algum","de jeito nenhum", "de forma alguma", "nunca", "não", "absolutamente", "nem", "tampouco"]
        for i in range(0, len(token)):
            aux = token[i]
            if token[i] == "de" and (token[i+1] == "modo" or token[i+1] == "jeito" or token[i+1] == "forma" ):
                aux = token[i] + " " + token[i+1] + " " + token[i+2]
            for j in negacao:
                if aux == j:
                    qtd += 1
        return qtd
    
    def countAnt(self, s1, s2):
        vecs = self.preProcess(s1, s2)
        count = 0
        for i in vecs['vec1']:
            for k in vecs['vec2']:
                    try:
                        if self.isAntonimosChave(i, k):
                            count+=1
                            vecs['vec2'].remove(k)
                    except:
                        pass
        return count
    
    def subjEquals(self, dep1, dep2):
        subj = False
        root = []
        subj = []
        for i in dep1:
            if i['rel'] == 'ROOT':
                root.append(i)
            elif i['rel'] == 'nsubj' or i['rel'] == 'nsubj:pass':
                subj.append(i)
        
        for i in dep2:
            if i['rel'] == 'ROOT':
                root.append(i)
            elif i['rel'] == 'nsubj' or i['rel'] == 'nsubj:pass':
                subj.append(i) 
        
        try:
            subj1 = subj[0]['text']
            for i in subj[0]['children']:
                subj1 += " "+str(i)

            subj2 = subj[1]['text']
            for i in subj[1]['children']:
                subj2 += " "+str(i)

            lcsrsub = self.lcsr(subj1, subj2)
            
            root1 = ""
            for i in root[0]['children']:
                root1 += " "+str(i)

            root2 = ""
            for i in root[1]['children']:
                root2 += " "+str(i)

            lcsrroot = self.lcsr(root1, root2)

            m = (lcsrsub + lcsrroot)/2

            if m > 0.8:
                return True
        except:
            pass
        return False
    
    def objEquals(self, dep1, dep2):
        subj = False
        root = []
        subj = []
        for i in dep1:
            if i['rel'] == 'ROOT':
                root.append(i)
            elif i['rel'] == 'obj':
                subj.append(i)
        
        for i in dep2:
            if i['rel'] == 'ROOT':
                root.append(i)
            elif i['rel'] == 'obj':
                subj.append(i) 
        
        try:
            subj1 = subj[0]['text']
            for i in subj[0]['children']:
                subj1 += " "+str(i)

            subj2 = subj[1]['text']
            for i in subj[1]['children']:
                subj2 += " "+str(i)

            lcsrsub = self.lcsr(subj1, subj2)
            
            root1 = ""
            for i in root[0]['children']:
                root1 += " "+str(i)

            root2 = ""
            for i in root[1]['children']:
                root2 += " "+str(i)

            lcsrroot = self.lcsr(root1, root2)

            m = (lcsrsub + lcsrroot)/2

            if m > 0.8:
                return True
        except:
            pass
        return False

    def verificarNegacao(self, vec1, vec2):
        
        f1 = self.countNeg(vec1['text'])
        f2 = self.countNeg(vec2['text'])
        
        return max(f1, f2) - min(f1, f2)
        
    def entityEquals(self, vec1, vec2, model):
        s1 = model(vec1['text'])
        s2 = model(vec2['text'])
        count = 0
        sim = 0
        for ents1 in s1.ents:
            for ents2 in s2.ents:
                sim = self.lcsr({"text":ents1.text}, {"text":ents2.text})
                if sim >= 0.5 and ents1.label_ == ents2.label_:
                    count+=1
        return count

    def calcTfIDF(self, s1, s2):
        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform([s1['text'], s2['text']])
        
        return (tfidf * tfidf.T).A[0][1]
    
    def countSinonimos(self, sentenca1, sentenca2, sinonimos):
        doc = self.stanfordModel(sentenca1)
        doc1 = self.stanfordModel(sentenca2)
        count = 0
        tk1 = []
        tk2 = []
        #percorre sentença1
        for sent in doc.sentences:
            for token in sent.words:
                tk1.append(token.lemma)
        for s in doc1.sentences:
            for t in s.words:
                tk2.append(t.lemma)
        
        tk1 = self.removeStopWords(tk1)
        tk2 = self.removeStopWords(tk2)
        for i in tk1:
            for j in tk2:
                try:
                    #if self.isSinonimosPapel(i, j, sinonimos) and i != j:
                    if self.isSinonimosChave(i, j) and i != j:
                        count+=1
                        tk2.remove(j)
                except:
                    pass
        return count
    
    def countTokensEquals(self, s1, s2, sinonimos):
        doc = self.stanfordModel(s1)
        doc1 = self.stanfordModel(s2)
        s1 = []
        s2 = []
        
        for sent in doc.sentences:
            for word in sent.words:
                s1.append(word.lemma)
        for sent in doc1.sentences:
            for word in sent.words:
                s2.append(word.lemma)
        count = 0
        for i in s1:
            for k in s2:
                if i == k:
                    count += 1
                    break
        return count
    
    def countLemmasEquals(self, s1, s2, sinonimos):
        doc1 = self.stanfordModel(s1)
        doc2 = self.stanfordModel(s2)
        count = 0
        tk1 = []
        tk2 = []
        #percorre sentença1
        for sent in doc1.sentences:
            for token in sent.words:
                if token.upos == "VERB" or token.upos == "NOUN" or token.upos == "ADV" or token.upos == "ADJ":
                    tk1.append(token.lemma)
                
        for s in doc2.sentences:
            for token in s.words:
                if token.upos == "VERB" or token.upos == "NOUN" or token.upos == "ADV" or token.upos == "ADJ":
                    tk2.append(token.lemma)
        
        for i in tk1:
            for k in tk2:
                try:
                    if i == k or self.isSinonimos(i, k, sinonimos):
                        count += 1
                        tk2.remove(k)
                except:
                    pass
        return count
    
    def getListSubj(self):
        sentenca = self.readXml()
        sinonimos = self.listaSinonimos()
        sp = self.sp
        i = 0
        l = []
        while i < len(sentenca) - 1:
            subj = self.subjEquals(self.dependenceParsySpacy(sentenca[i]['text'], sp), self.dependenceParsySpacy(sentenca[i+1]['text'], sp))
            #subj = self.subjectEquals([sentenca[i]['text'],sentenca[i+1]['text']], sinonimos)
            obj = self.objEquals(self.dependenceParsySpacy(sentenca[i]['text'], sp), self.dependenceParsySpacy(sentenca[i+1]['text'], sp))

            l.append({
                "subj":subj,
                "obj":obj
            })
            i+=2
        return l
    
    def getListTokenEquals(self):
        sinonimos = self.listaSinonimos()
        sentenca = self.readXml()
        ltk = []
        i = 0
        while i < len(sentenca) - 1:
            ltk.append(
                self.countTokensEquals(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
            )
            i+=2
        
        return ltk
    
    def getListLemmaEquals(self):
        sentenca = self.readXml()
        sinonimos = self.listaSinonimos()
        ltk = []
        i = 0
        while i < len(sentenca) - 1:
            ltk.append(
                self.countLemmasEquals(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
            )
            i+=2
        
        return ltk
    
    def getExpressions(self):
        l = [(r"(DET )?NOUN ADJ","DET? NOUN ADJ", r"(DET)?NOUN (PROPN )?ADP (DET )?(PROPN|NOUN)"),
         (r"(DET )? (NOUN|PROPN", r"(DET )?NOUN (ADP )?(ADJ )?(NOUN|PROPN"),
         (r"(DET )?(NOUN )?(ADP )?(DET )?(PROPN|NOUN{1})", r"(DET )?(PROPN|NOUN) ADP (DET )?(PROPN|NOUN)?"),
         (r"(DET )?(PROPN|NOUN) (ADJ )? (PROPN)",r"(DET )?(PROPN|NOUN) (ADJ )? ADJ? (PROPN)"),
         (r"(DET )?(PROPN )ADP NOUN (NUM)?",r"(DET )?PROPN"),
         (r"(DET )?NOUN ADP (DET |NOUN )?PROPN", r"(DET )?(NOUN )?PROPN"),
         (r"(DET )?(ADJ )?NOUN ADP (DET )? NOUN", r"(DET )?NOUN ADP (DET )? NOUN"),
         (r"ADV NUM (NOUN ) (ADJ|VERB)?",r"(ADV )?NUM (NOUN )?ADJ"),
         (r"(DET )?PROPN PROPN", r"(DET )?(NOUN )?(ADP )?(DET )?PROPN (ADP )?(DET )?PROPN"),
         (r"(DET )?NOUN", r"(DET )?NOUN ADJ"),
         (r"(DET )?NOUN ADJ ADP (DET )?PRON (NOUN )?", r"(DET )?NOUN ADJ ADP (DET )?PRON NOUN"),
         (r"PRON NOUN", r"NOUN ADP (DET )?PROPN"),
         (r"NUM N ADP (DET )?NOUN",r"NUM NOUN ADP (DET )?NOUN ADP (DET )?NOUN"),
         (r"NUM NOUN", r"NUM ADV NOUN"),
         (r"VERB ADP VERB", r"VERB"),
         (r"VERB VERB", r"VERB"),
         (r"ADJ NOUN", r"NOUN ADV ADJ"),
         (r"(ADV )?(ADP )?(NUM )?NOUN (PRON )?VERB NOUN",r"(ADV )?(NUM )? NOUN PRON VERB VERB VERB NOUN"),
         (r"NOUN (ADJ)? PRON VERB VERB VERB ADP DET NOUN",r"NOUN (ADJ )? ADP DET NOUN"),
         (r"DET NOUN DET DET N",r"DET NOUN PRON (ADV )?(PRON )?VERB ADP DET (ADP )?(DET )?(NOUN )?"),
         (r"NOUN ADJ PROPN", r"PROPN NOUN ADP (DET )?(NOUN )?PROPN")
        ]
        return l

    def regularExpression(self, s1, s2):
        expressions = self.getExpressions()
        doc = self.stanfordModel(s1['text'])
        doc2 = self.stanfordModel(s2['text'])
        s1l = []
        s2l = []
        s1p = ''
        s2p = ''
        count = 0
        #sim = self.calcTfIDF(s1, s2)
        
        for sent in doc.sentences:
            for token in sent.words:
                s1l.append({"palavra":token.lemma, "upos":token.upos})
        for sent in doc2.sentences:
            for token in sent.words:
                s2l.append({"palavra":token.lemma, "upos":token.upos})
            
        for i in s1l:
            s1p+=i['upos']+' '
        for i in s2l:
            s2p+=i['upos']+' ' 
            #test = re.compile(r'(DET)?NOUN{1} (PROPN )?ADP (DET )?(PROPN|NOUN)')
        #print("S2: ",s2p,"\n","S1: ", s1p)
        for i in expressions:
            try:
                if (re.search(i[0], s1p) and re.search(i[1], s2p)) or (re.search(i[1], s1p) and re.search(i[0], s2p)):
                    count+=1
            except:
                pass
        return count
    
    def getListRegex(self):
        sentenca = self.readXml()
        ltk = []
        i = 0
        while i < len(sentenca) - 1:
            ltk.append(
                self.regularExpression(sentenca[i], sentenca[i+1])
            )
            i+=2
        
        return ltk
    
    def isHiperonimo(self, p1, p2):
        hipe = Util.readJson("./assets/hiperonimos.json")
    
        for i in hipe:
            if p1 == i['palavra'] or p2 == i['palavra']:
                for k in i['hiperonimos']:
                    if p2 == k or p1 == k:
                        return True
                return False
        
        return False

    def countHiperonimo(self, s1, s2):
        s1 = s1.split(' ')
        s2 = s2.split(' ')
        count = 0
        for i in s1:
            for k in s2:
                if self.isHiperonimo(i, k):
                    count += 1
                    s2.remove(k)
                    break
        return count
    
    def getListHipe(self):
        sentenca = self.readXml()
        lhipe = []
        i = 0
        while i < len(sentenca)-1:
            lhipe.append(
                self.countHiperonimo(sentenca[i]['text'], sentenca[i+1]['text'])
            )
            i+=2
        
        return lhipe
    
    def bertTest(self, sentenca):
        tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        
        input_ids = tokenizer.encode(sentenca, return_tensors='pt')

        with torch.no_grad():
            outs = model(input_ids)
            encoded = outs[0][0, 1:-1]  # Ignore [CLS] and [SEP] special tokens
        
        #sim = simCosseno(encoded[0], encoded[1])
        
        return encoded
    
    def simBert(self, s1, s2):
        s1Emb = self.bertTest(s1)
        s2Emb = self.bertTest(s2)
        sim = self.simCosseno(s1Emb[0], s2Emb[0])
        return sim.item()
    
    def listSimBert(self, index):
        sentenca = self.readXml()
        i = index
        lhipe = []
        temp = 0
        while i < len(sentenca)-1 and temp < 10600:
            print(sentenca[i]['id'])
            start = time.time()
            lhipe.append(
                self.simBert(sentenca[i]['text'], sentenca[i+1]['text'])
            )
            i+=2
            end = time.time()
            temp += (end-start)
        
        return lhipe
    
    def listCountSinonimos(self):
        sentenca = self.readXml()
        sinonimos = Util.readJson("./assets/sinonimos.json")
        lhipe = []
        i = 0
        while i < len(sentenca)-1:
            print(sentenca[i]['id'])
            lhipe.append(
                self.countSinonimos(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
            )
            i+=2
        
        return lhipe
    
    def joinFeatures(self):
        filename = self.corpus.split('/')
        filename = filename[4].split('.xml')
        arq = Util.readJson("./assets/test_features_"+filename[0]+"10.0.json")
        nlist = []
        print(filename)
        #sub = self.getListSubj()
        has = []
        i = 0
        try:
            has = Util.readJson("./test_features_"+filename[0]+"_bert.json")
            i = len(has)
        except:
            i = 0
            pass
        count = self.listSimBert(i*2)
        aux = 0
        while i < len(arq) and aux < len(count): 
            print(i+1)
            nlist.append({
                "id":arq[i]['id'],
                "sentencas":arq[i]['sentencas'],
                "wmd":arq[i]['wmd'],
                "cos_embeddings":arq[i]['cos_embeddings'],
                "hiperonimo":arq[i]['hiperonimo'],
                "tf-idf":arq[i]['sim'],
                "subjEquals":arq[i]['subjEquals'],
                "objEquals":arq[i]['objEquals'],
                "parafraseamento":arq[i]['parafraseamento'],
                "neg":arq[i]['neg'],
                "root":arq[i]['root'],
                "bert":count[aux],
                "qtdTokensEquals":arq[i]['qtdTokensEquals'],
                "qtdLemmasEquals":arq[i]['qtdLemmasEquals'],
                "entity":arq[i]['entity'],
                "antonimos":arq[i]['antonimos'],
                "sinonimos":arq[i]['sinonimos'],
                "entailment":arq[i]['entailment'],
                "similarity":arq[i]['similarity']
            })
            i+= 1
            aux+=1
        has+=nlist
        Util.writeJson(has, "test_features_"+filename[0]+"_bert")
        time.sleep(10)
        os.system('shutdown -s -f')
        return nlist
        
    def getFeatures(self):
        sentenca = self.readXml()
        model = self.embeddings
        sinonimos = self.listaSinonimos()
        filename = self.corpus.split('/')
        filename = "./assets/test_features-"+filename[5]+"1.0"
        
        sp = spacy.load('pt_core_news_sm')
        total = 0
        i = 0
        features = []
        try:
            features = Util.readJson(filename+'.json')  
             
            i = len(features) * 2
        except :
            features = []

        while i < len(sentenca) - 1 and total < 15000:
            print(sentenca[i]['id'])
            start = time.time()
            try:
                
                sin = self.countSinonimos(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
                dep = self.anotacaoDependencia(sentenca[i], sentenca[i+1])
                cos_wmd = self.cosWordEmbeddings(dep['sent'][0]['token'], dep['sent'][1]['token'], model)
                sim = self.calcTfIDF(sentenca[i], sentenca[i+1])
                neg = self.verificarNegacao(sentenca[i], sentenca[i+1])
                root = self.verboPrinciapalIgual(dep['sent'], sinonimos)
                entity = self.entityEquals(sentenca[i], sentenca[i+1], sp)
                antonimos = self.countAnt(sentenca[i]['text'],sentenca[i+1]['text'])
                subj = self.subjEquals(self.dependenceParsySpacy(sentenca[i]['text'], sp), self.dependenceParsySpacy(sentenca[i+1]['text'], sp))
                obj = self.objEquals(self.dependenceParsySpacy(sentenca[i]['text'], sp), self.dependenceParsySpacy(sentenca[i+1]['text'], sp))
                hiperonimo = self.countHiperonimo(sentenca[i]['text'], sentenca[i+1]['text'])
                parafrase = self.regularExpression(sentenca[i], sentenca[i+1])
                ltoken = self.countTokensEquals(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
                llemmas = self.countLemmasEquals(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
                features.append({
                    "id":sentenca[i]['id'],
                    "objEquals":obj,
                    "hiperonimo":hiperonimo,
                    "parafraseamento":parafrase,
                    "qtdTokensEquals":ltoken,
                    "qtdLemmasEquals":llemmas,
                    "sentencas":[sentenca[i]['text'], sentenca[i+1]['text']],
                    "wmd":cos_wmd['wmd'],
                    "cos_embeddings":cos_wmd['cos'],
                    "sim":sim,
                    "subjEquals":subj,
                    "neg":neg,
                    "root":root,
                    "entity":entity,
                    "antonimos":antonimos,
                    "sinonimos":sin,
                    "entailment":sentenca[i]['entailment'],
                    "similarity":sentenca[i]['similarity']
                })
            except:
                pass
            
            i += 2
            end = time.time()
            total += (end - start)

        Util.writeJson(features, filename)
        time.sleep(10)
        #os.system('shutdown -s -f')
        
        return features
    
    def getFeaturesIndividual(self, text1, text2):
        print("Carregando as word embeddings")
        sentenca = [{"text":text1, "id":0, "entailment":"", "similarity":""}, {"text":text2, "id":0, "entailment":"", "similarity":""}]
        model = self.embeddings
        sinonimos = self.listaSinonimos()
        print("Carregando model spacy")
        sp = spacy.load('pt_core_news_sm')
        i = 0
        features = []
        sin = self.countSinonimos(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
        dep = self.anotacaoDependencia(sentenca[i], sentenca[i+1])
        cos_wmd = self.cosWordEmbeddings(dep['sent'][0]['token'], dep['sent'][1]['token'], model)
        sim = self.calcTfIDF(sentenca[i], sentenca[i+1])
        neg = self.verificarNegacao(sentenca[i], sentenca[i+1])
        root = self.verboPrinciapalIgual(dep['sent'], sinonimos)
        entity = self.entityEquals(sentenca[i], sentenca[i+1], sp)
        antonimos = self.countAnt(sentenca[i]['text'],sentenca[i+1]['text'])
        subj = self.subjEquals(self.dependenceParsySpacy(sentenca[i]['text'], sp), self.dependenceParsySpacy(sentenca[i+1]['text'], sp))
        obj = self.objEquals(self.dependenceParsySpacy(sentenca[i]['text'], sp), self.dependenceParsySpacy(sentenca[i+1]['text'], sp))
        hiperonimo = self.countHiperonimo(sentenca[i]['text'], sentenca[i+1]['text'])
        parafrase = self.regularExpression(sentenca[i], sentenca[i+1])
        ltoken = self.countTokensEquals(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
        llemmas = self.countLemmasEquals(sentenca[i]['text'], sentenca[i+1]['text'], sinonimos)
        features.append({
            "id":sentenca[i]['id'],
            "objEquals":obj,
            "hiperonimo":hiperonimo,
            "parafraseamento":parafrase,
            "qtdTokensEquals":ltoken,
            "qtdLemmasEquals":llemmas,
            "sentencas":[sentenca[i]['text'], sentenca[i+1]['text']],
            "wmd":cos_wmd['wmd'],
            "cos_embeddings":cos_wmd['cos'],
            "sim":sim,
            "subjEquals":subj,
            "neg":neg,
            "root":root,
            "entity":entity,
            "antonimos":antonimos,
            "sinonimos":sin,
            "entailment":sentenca[i]['entailment'],
            "similarity":sentenca[i]['similarity']
        })
        return features
        

if __name__ == '__main__':
    filename = "./assets/corpus/2019/assin2-train-only"
    models = LoadModels()
    #teste = ExtractFeatures(models.loadModelStanfordNLP(), models.loadWorldEmbeddings(), models.loadSpacy, filename+".xml")
    teste = ExtractFeatures("", "", "", filename+".xml")
    teste.joinFeatures()