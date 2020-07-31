import json
import os
import nltk
import spacy
import re
import stanfordnlp
import jellyfish as jf
import xml.etree.ElementTree as et
from xml.etree import cElementTree as ET
import numpy as np
import warnings
import math
import gensim
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings("ignore")

config = {
        'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separ-+ated list of processors to use
        'lang': 'pt', # Language code for the language to build the Pipeline in
        'tokenize_model_path': 'C:/Users/LUIZ ANTONIO/stanfordnlp_resources/pt_bosque_models/pt_bosque_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        'mwt_model_path': 'C:/Users/LUIZ ANTONIO/stanfordnlp_resources/pt_bosque_models/pt_bosque_mwt_expander.pt',
        'pos_model_path': 'C:/Users/LUIZ ANTONIO/stanfordnlp_resources/pt_bosque_models/pt_bosque_tagger.pt',
        'pos_pretrain_path': 'C:/Users/LUIZ ANTONIO/stanfordnlp_resources/pt_bosque_models/pt_bosque.pretrain.pt',
        'lemma_model_path': 'C:/Users/LUIZ ANTONIO/stanfordnlp_resources/pt_bosque_models/pt_bosque_lemmatizer.pt',
        'depparse_model_path': 'C:/Users/LUIZ ANTONIO/stanfordnlp_resources/pt_bosque_models/pt_bosque_parser.pt',
        'depparse_pretrain_path': 'C:/Users/LUIZ ANTONIO/stanfordnlp_resources/pt_bosque_models/pt_bosque.pretrain.pt'
    }
nlp = stanfordnlp.Pipeline(**config)

fileName = 'corpus/2019/assin2-train-only.xml'

def removeStopWords(text):
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

def readXml(fileName):
    base_path = os.path.dirname(os.path.realpath(__file__))

    xml_file = os.path.join(base_path, fileName)

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

def listaSinonimos():
    db_tep = open("databases/base_tep2.txt", "r")
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

def writeJson(data, name):
    with open(name+".json", 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4, separators=(',',':'))

def readJson(name):
    with open(name, 'r', encoding='utf8') as f:
        return json.load(f)

def isSinonimos(palavra1, palavra2, sinonimos):
    palavra1.lower()
    palavra2.lower()
    doc = nlp(palavra1)
    doc2 = nlp(palavra2)
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
    # sin = []
    # for i in sinonimos:
    #     if i['palavra'] == palavra1:
    #        sin.append(i)
    # for i in sin:
    #     if palavra2 == i['sinonimo_de']:
    #         return True
    # return False 

def countSinonimos(sentenca1, sentenca2, sinonimos):
    doc = nlp(sentenca1)
    doc1 = nlp(sentenca2)
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
    
    tk1 = removeStopWords(tk1)
    tk2 = removeStopWords(tk2)
    
    for i in tk1:
        for j in tk2:
            if isSinonimos(i, j, sinonimos)  and i != j:
                count+=1
    # if isSinonimos(p1, p2, sinonimos) and p1 != p2:
    #                print(p1, p2)
    #                     count+=1
    return count

def frequencia(text, sinonimos):
    freq = []
    x = len(text)
    k = 1
    i = 0
    while i < x:
        c = 1
        for j in range(k,x):
            if text[i] == text[j] or isSinonimos(text[i],text[j], sinonimos):
                c+=1
        freq.append({'palavra':text[i],'freq':c})
        k+=c
        i+=c
    return freq

def SepararFreq(v1,v2, db_tep):
    f1 = []
    f2 = []

    if len(v1) >= len(v2):
        freq1 = frequencia(sorted(v1), db_tep)
        freq2 = frequencia(sorted(v2), db_tep)
    else:
        freq1 = frequencia(sorted(v2), db_tep)
        freq2 = frequencia(sorted(v1), db_tep)
    x = max(len(freq1), len(freq2))
    y = min(len(freq1), len(freq2))
    if y < x:
        i = 0
        while i < y:
            if i >= len(freq2):
                for j in range(0,len(freq2)):
                    if isSinonimos(freq1[i]['palavra'], freq2[j]['palavra'], db_tep):
                        freq2[j] = freq1[i]
                        break
            if freq1[i] in freq2:
                f1.insert(i,freq1[i]['freq'])
                f2.insert(i,freq2[freq2.index(freq1[i])]['freq'])
            else:
                f1.insert(i,freq1[i]['freq'])
                f2.insert(i,0)
            if i < y and freq2[i] not in freq1:
                f1.insert(i,0)
                f2.insert(i,freq2[i]['freq'])
                
            i+=1
    elif y == x:
        i = 0
        while i < y:
            for j in range(0,len(freq2)):
                if isSinonimos(freq1[i]['palavra'], freq2[j]['palavra'], db_tep):
                    freq2[j] = freq1[i]
                    break
            if freq1[i] in freq2:
                f1.insert(i,freq1[i]['freq'])
                f2.insert(i,freq2[freq2.index(freq1[i])]['freq'])
            else:
                f1.insert(i,freq1[i]['freq'])
                f2.insert(i,0)
            if i < y and freq2[i] not in freq1:
                f1.insert(i,0)
                f2.insert(i,freq2[i]['freq'])
                
            i+=1
    return {"frase1":f1, "frase2":f2}

def levenshtein(s1, s2):
    sim = []
    if len(s1) < len(s2):
        aux = s1
        s1 = s2
        s2 = aux
    for i in range(0, len(s1)):
        sim.append([])
        for j in range(0, len(s2)):
            sim[i].append(round(1 - (jf.levenshtein_distance(s1[i], s2[j]) / max(len(s1[i]), len(s2[j]))), 2))

    total = 0
    interacao = 0
    for i in sim:
        maior = 0
        for j in i:
            if j > maior:
                maior = j
        total += maior
        interacao += 1
    sim = total/interacao
    t1 = len(s1)
    t2 = len(s2)
    if t1 > t2:
        return round(1 - sim, 2)
    return round(sim,2)

def espacoVetorial(s1, s2, db_tep):
    freq = SepararFreq(s1, s2, db_tep)
    sim = 0
    somac = 0
    somabe = 0
    somabd = 0
    
    for i in range(0,len(freq['frase1'])):
        somac += freq['frase1'][i]*freq['frase2'][i]
        somabe += freq['frase1'][i]**2
        somabd += freq['frase2'][i]**2
    sim = somac/(float((somabe*somabd)**0.5))
    return sim

def anotacaoDependencia(s1, dbName):
    i = 0
    ldic = []
    ant = -1
    aux = []
    sinonimos = listaSinonimos()
    taux = []
    while i < len(s1):
        doc = nlp(s1[i]['text'])
        dic = {}
        l = []
        token = []
        for sent in doc.sentences:
            for word in sent.words:
                token.append(word.lemma)
                if word.governor > 0:
                    dic = {"index":word.index, "text":word.text, "governorIndex": word.governor,"governor":doc.sentences[0].words[word.governor-1].lemma , "rel":word.dependency_relation}
                else:
                    dic = {"index":word.index, "text":word.text, "governorIndex": word.governor,"governor":'root' , "rel":word.dependency_relation}
                l.append(dic)        
        
        token = removeStopWords(token)
        dic = {"id":s1[i]["id"],"text":s1[i]["text"],"dependency":l}
        
        
        if ant == s1[i]['id']:
            aux.append(dic)
            #mev = espacoVetorial(taux, token)
            #qtdSin = countSinonimos(taux, token)
            dic = {"id":s1[i]['id'], "sent":aux}# "mev":mev, "qtdSin":qtdSin}
            ldic.append(dic)
            aux = []
            taux = []
        else:
            aux.append(dic)
            taux = token
            ant = s1[i]['id']
        i+=1
    writeJson(ldic, dbName)

def calcMev(sentencas, arqName):
    tant = []
    lmev = []
    db_tep = listaSinonimos()
    for i in sentencas:
        token = []
        doc = nlp(i["text"])
        for sent in doc.sentences:
            for word in sent.words:
                token.append(word.lemma)
        token = removeStopWords(token)
        
        if len(tant) == 0:
            tant = token
        else:
            mev = {"id":i['id'], "mev":espacoVetorial(tant, token, db_tep)}
            lmev.append(mev)
            tant = []
    writeJson(lmev, "mev-dev")

def loadWorldEmbeddings():
    return KeyedVectors.load_word2vec_format('skip_s300.txt')

def preProcess(sentenca1, sentenca2):
    token = []
    token2 = []
    doc = nlp(sentenca1["text"])
    for sent in doc.sentences:
        for word in sent.words:
            token.append(word.lemma)
    token = removeStopWords(token)
    
    doc = nlp(sentenca2["text"])
    for sent in doc.sentences:
        for word in sent.words:
            token2.append(word.lemma)
    token2 = removeStopWords(token2)
    
    return {"vec1":token, "vec2":token2}

def calcularSimilaridade(sentenca1, sentenca2):
    t = preProcess(sentenca1, sentenca2)    
    kd = sepOcorrencias(t['vec1'], t['vec2'])
    vec1 = []
    vec2 = []
    for i in range(0, len(kd['vec1'])):
      vec1.append(kd['vec1'][i]['freq'])
      vec2.append(kd['vec2'][i]['freq'])
    sim = simCosseno(vec1, vec2)
    
    return sim
    

def sepOcorrencias(token, token2):
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

def simCosseno(vec1, vec2):
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
        
def cosWordEmbeddings(vec1, vec2, model):
    vec1Emb = []
    vec2Emb = []
    vec1.sort()
    vec2.sort()
    freq = sepOcorrencias(vec1, vec2)
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
                vec1Emb.append(sum(emb))#(sum(emb)/len(emb)))
            except:
                vec2Emb.append(0)
                pass
        if vec2[i] == 0:
            vec2Emb.append(0)
        else:
            try:
                emb = model[vec2[i].lower()]
                vec2Emb.append(sum(emb))#(sum(emb))/len(emb)))
            except:
                vec2Emb.append(0)
                pass
    cos = simCosseno(vec1Emb, vec2Emb)
    wmd = model.wmdistance(vec1, vec2)
    return {"wmd":wmd, "cos":cos}

def anotarWMDeCos(sentencas, name):
    model = loadWorldEmbeddings()
    l = []
    i = 0
    while i < len(sentencas):
        print(sentencas[i]['id'])
        sentenca1 = sentencas[i]
        sentenca2 = sentencas[i+1]
        t = preProcess(sentenca1, sentenca2)
        sim = cosWordEmbeddings(t['vec1'], t['vec2'],  model)
        l.append({"id":sentencas[i]['id'],"entailment":sentencas[i]['entailment'], "cos":sim['cos'], "wmd":sim['wmd']})
        i += 2
    writeJson(l, name)

def verboPrinciapalIgual(vec, sinonimos):
    isSubj = False
    isRoot = False

    for i in vec[0]['dependency']:
        if i['rel'] == 'root':
            for j in vec[1]['dependency']:
                if j['rel'] == 'root' and (i['text'] == j['text'] or isSinonimos(i['text'],j['text'], sinonimos)):
                    isRoot = True
                    break
                elif j['rel'] == 'root' and i['text'] != j['text']:
                    isRoot = False
                    break
        if i['rel'] == 'nsubj' or i['rel'] == 'nsubj:pass':
            for j in vec[1]['dependency']:
                if (j['rel'] == 'nsubj' or i['rel'] == 'nsubj:pass') and (i['text'] == j['text'] or isSinonimos(i['text'],j['text'], sinonimos)):
                    isSubj = True
                    break
                elif (j['rel'] == 'nsubj' or i['rel'] == 'nsubj:pass') and i['text'] != j['text']:
                    isSubj = False
                    break
                
    return {'id':vec[0]['id'] ,'subjEquals': isSubj, 'rootEquals':isRoot, 'texto':[vec[0]['text'], vec[1]['text']]}

def gerarArff():
    sim_wordEmb = readJson('sim-word-embeddings-train-600.json')
    verbo_sujeito = readJson('verbo_principal_train.json')
    cos = readJson('sim-cos-train.json')
    #sin = readJson('lista_sinonimos_train.json')
    neg = readJson('negacao-train.json')
    ent = readJson('entity-train.json')
    arff = open('similarity-sentences-train-600-test.arff','w')
    head = '@relation similarity-sentences\n'
    #head += '@attribute subjEquals       {sim, nao}\n'
    head += '@attribute rootEquals       {sim, nao}\n'
    head += '@attribute hasNeg           {sim, nao}\n'
    head += '@attribute entityQtd        numeric\n'
    head += '@attribute cosEmbleddings   real\n'
    head += '@attribute wmd              real\n'
    head += '@attribute cos              real\n'
    #head += '@attribute qtdSin           numeric\n'
    head += '@attribute class            {none, entailment}\n'
    head += '@data \n'
    arff.write(head)
    for i in range(0, len(sim_wordEmb)):
        string = ''
        if verbo_sujeito[i]['subjEquals'] and verbo_sujeito[i]['rootEquals']:
            string+='sim,'
        elif verbo_sujeito[i]['subjEquals'] and not verbo_sujeito[i]['rootEquals']:
            string+='nao,'
        elif not verbo_sujeito[i]['subjEquals'] and verbo_sujeito[i]['rootEquals']:
            string+='sim,'
        else:
            string+='nao,'
            
        if neg[i]['vec1'] != neg[i]['vec2']:
            string += "sim,"
        else:
            string += "nao,"
        string+=str(ent[i]['entity'])+','
        string+=str(sim_wordEmb[i]['cos'])+','
        string+=str(sim_wordEmb[i]['wmd'])+','    
        string+=str(cos[i]['sin'])+','
        #string+=str(sin[i]['sinonimos'])+','
        string+=sim_wordEmb[i]['entailment'].lower()+'\n'
        arff.write(string)
    arff.close()
        
def datasetScikit(dataType):
    sim_wordEmb = readJson('sim-word-embeddings-'+dataType+'.json')
    verbo_sujeito = readJson('verbo_principal_'+dataType+'.json')
    cos = readJson('sim-cos-'+dataType+'.json')
    sin = readJson('lista_sinonimos_'+dataType+'.json')
    neg = readJson('negacao-'+dataType+'.json')
    ent = readJson('entity-'+dataType+'.json')
    labels_names = ['none','entailment']
    labels = []
    feature_names = ['hasNeg','entityQtd','cosEmbleddings','wmd','cos','qtdSin']
    features = []
    for i in range(0, len(sin)):
        aux = []
        # if verbo_sujeito[i]['subjEquals'] and verbo_sujeito[i]['rootEquals']:
        #    aux += [1]
        # elif verbo_sujeito[i]['subjEquals'] and not verbo_sujeito[i]['rootEquals']:
        #    aux += [0]
        # elif not verbo_sujeito[i]['subjEquals'] and verbo_sujeito[i]['rootEquals']:
        #    aux += [1]
        # else:
        #     aux += [0]
        
        if neg[i]['vec1'] != neg[i]['vec2']:
            aux += [1]
        else:
            aux += [0] 
        aux.append(ent[i]['entity'])
        aux.append(sim_wordEmb[i]['cos'])
        aux.append(sim_wordEmb[i]['wmd'])    
        aux.append(cos[i]['sin'])
        #aux.append(sin[i]['sinonimos'])
        if(sim_wordEmb[i]['entailment']=='None'):
            labels.append(0)
        else:
            labels.append(1)
        features.append(aux)
        
    return {'label_names':labels_names, 'labels':labels, 'feature_names': feature_names, 'features': features}
        
def naiveBayes(data):
    features = data['features']
    labels = data['labels']
    label_names = data['label_names']
    feature_names = data['feature_names']
    
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
    # Inicializar classificador
    gnb = GaussianNB()
    scores = cross_validate(gnb, features, labels, return_train_score=False, scoring=
                            ['accuracy',
                            'average_precision',
                            'f1',
                            'precision',
                            'recall',
                            'roc_auc'])
    # Treinar classificador
    model = gnb.fit(train, train_labels)
    # fazer predições
    preds = gnb.predict(test)
    
    print("Naive Bayes")
    
    print(preds)
    #avaliar precisão    
    print(accuracy_score(test_labels, preds))

    t = 0
    for i in scores['test_accuracy']:
        t += i
    print(t/len(scores['test_accuracy']))

def supportVM(data):
    features = data['features']
    labels = data['labels']
    label_names = data['label_names']
    feature_names = data['feature_names']
    
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
    # Inicializar classificador
    clf = SVC(gamma='auto')
    
    scores = cross_validate(clf, features, labels, return_train_score=False, scoring=
                            ['accuracy',
                            'average_precision',
                            'f1',
                            'precision',
                            'recall',
                            'roc_auc'])
    # Treinar classificador
    model = clf.fit(train, train_labels)
    # fazer predições
    preds = clf.predict(test)
    
    print("Support Vector Machine")
    
    print(preds)
    #avaliar precisão    
    print(accuracy_score(test_labels, preds))
    
    t = 0
    for i in scores['test_accuracy']:
        t += i
    print(t/len(scores['test_accuracy']))
    
def decisionTreeModel(data):
    features = data['features']
    labels = data['labels']
    label_names = data['label_names']
    feature_names = data['feature_names']
    
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

    clf = DecisionTreeClassifier()
    scores = cross_validate(clf, features, labels, return_train_score=False, scoring=
                            ['accuracy',
                            'average_precision',
                            'f1',
                            'precision',
                            'recall',
                            'roc_auc'])
    clf = clf.fit(train, train_labels)
    
    preds = clf.predict(test)
    
    print("Arvore de decisão")
    
    print(preds)
    
    print(accuracy_score(test_labels, preds))
    
    t = 0
    for i in scores['test_accuracy']:
        t += i
    print(t/len(scores['test_accuracy']))

def randomForest(data, testXml, output):
    features = data['features']
    labels = data['labels']
    label_names = data['label_names']
    feature_names = data['feature_names']
    
    train = features
    test = datasetScikit('dev')

    clf = RandomForestClassifier()
    
    clf = clf.fit(train, labels)
    
    entailment_preds = clf.predict(test['features'])
     
    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))
    tree = ET.parse('src/corpus/2019/assin2-dev.xml')
    root = tree.getroot()
    entailment_to_str = ['None', 'Entailment']
    for i in range(len(entailment_preds)):
        pair = root[i]
        entailment_str = entailment_to_str[entailment_preds[i]]
        pair.set('entailment', entailment_str)
        #pair.set('similarity', str(predicted_similarity[i]))

    tree.write(output, 'utf-8')
    

    
    
    
    
def redesNeurais(data):
    features = data['features']
    labels = data['labels']
    label_names = data['label_names']
    feature_names = data['feature_names']
    
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    scores = cross_validate(clf, features, labels, return_train_score=False, scoring=
                            ['accuracy',
                            'average_precision',
                            'f1',
                            'precision',
                            'recall',
                            'roc_auc'])
    clf = clf.fit(train, train_labels)
    
    preds = clf.predict(test)
    
    print("Redes Neurais")
    
    print(preds)
    
    print(accuracy_score(test_labels, preds))
    
    t = 0
    for i in scores['test_accuracy']:
        t += i
    print(t/len(scores['test_accuracy']))

def identficarNegacao(vec1, vec2):
    nVec1 = False
    nVec2 = False
    
    for i in vec1:
        if i == "não":
            nVec1 = True
    for i in vec2:
        if i == "não":
            nVec2 = True
    
    return {"vec1":nVec1, "vec2":nVec2}

def verificarNegacao(sentencas):
    i = 0
    lNeg = []
    while i < len(sentencas) - 1:
        tk1 = sentencas[i]['text'].lower().split(' ')
        tk2 = sentencas[i+1]['text'].lower().split(' ')
        dicNeg = identficarNegacao(tk1, tk2 )
        lNeg.append({'id':sentencas[i]['id'], 'vec1':dicNeg['vec1'], 'vec2':dicNeg['vec2']})
        i+=2
    
    writeJson(lNeg, "negacao-train.json")

def entityEquals(vec1, vec2, model):
    s1 = model(vec1['text'])
    s2 = model(vec2['text'])
    count = 0
    sim = 0
    for ents1 in s1.ents:
        for ents2 in s2.ents:
            sim = calcularSimilaridade({"text":ents1.text}, {"text":ents2.text})
            if sim >= 0.5 and ents1.label_ == ents2.label_:
                count+=1
    return count
    
def anotaEntidades(sentencas, name):
    model = spacy.load('pt_core_news_sm')
    i = 0
    lEnt = []
    while i < len(sentencas) - 1:
        print(sentencas[i]['id'])
        lEnt.append({"id":sentencas[i]['id'],"entity":entityEquals(sentencas[i], sentencas[i+1], model)})
        i+=2
    writeJson(lEnt, name)


sentencas = readXml(fileName)


#text2 = preProcess("é melhor recuar daquele delinquente")
#dependenceParsy("O garoto comeu toda a banana")
#print(espacoVetorial(text1, text2))
# anotacaoDependencia(sentencas[2]['text'])
# anotacaoDependencia(sentencas[3]['text'])
# anotacaoDependencia(sentencas[8]['text'])
#anotacaoDependencia(sentencas)
#print(isSinonimos("afastar-se", "recuar"))
#print(isSinonimos("agredir", "atacar"))
#print(readXml(fileName)[2]['text'])
# i = 0
# lsim = []
# sinonimos = listaSinonimos()
# while i < len(sentencas):
#     print(sentencas[i]['id'])
#     dic = {"id":sentencas[i]["id"], "sinonimos":countSinonimos(sentencas[i]['text'],sentencas[i+1]['text'], sinonimos)}
#     lsim.append(dic)
#     i+=2
# writeJson(lsim, 'lista_sinonimos_test')
#anotacaoDependencia(sentencas, "anotacao-dev")

#print(calcMev(sentencas, "teste"))
#print(countSinonimos(sentencas[11030]['text'],sentencas[11031]['text']))
#test = readJson('parser.json')
gerarArff()
anotarWMDeCos(sentencas, "sim-word-embeddings-train-300n")

#anotarWMDeCos(sentencas, "sim-word-embeddings-dev")

# parser = readJson('anotacao-dev.json')
# sinonimos = listaSinonimos()

# l = []
# for i in parser:
#     l.append(verboPrinciapalIgual(i['sent'], sinonimos))
# writeJson(l, 'verbo_sujeito.json')

#erarArff()

#anotacaoDependencia(sentencas, "anotacao-test")

# lsin = []
# i = 0
# while i < len(sentencas)-1:
#   print(sentencas[i]['id'])
#   lsin.append({"id":sentencas[i]['id'], "sin":calcularSimilaridade(sentencas[i], sentencas[i+1])})
#   i+=2
# writeJson(lsin, 'sim-cos-train')
# sent = readXml('corpus/2019/assin2-dev.xml')
# lsin = []
# i = 0
# while i < len(sent)-1:
#     print(sent[i]['id'])
#     lsin.append({"id":sent[i]['id'], "sin":calcularSimilaridade(sent[i], sent[i+1])})
#     i += 2
# writeJson(lsin, 'sim-cos-dev')
#anotaEntidades(sent, 'entity-dev')

# data = datasetScikit('train')

# #anotarWMDeCos(sentencas, "sim-emb-sum")
# gerarArff()
# naiveBayes(data)
# supportVM(data)
# decisionTreeModel(data)
# redesNeurais(data)
# randomForest(data, 'assin2-dev.xml', 'teste.xml')
# anotaEntidades(sentencas, "entity-train")
#anotarWMDeCos(sentencas, "sim-word-embeddings-train-300")

#anotarWMDeCos(sentencas, "sim-word-embeddings-train-100")
# anotacao = readJson('anotacao-train.json')
# ldic = []
# sinonimos = listaSinonimos()
# for i in anotacao:
#     print(i['id'])
#     ldic.append(verboPrinciapalIgual(i['sent'], sinonimos))

# writeJson(ldic, 'verbo_principal_train')

# verificarNegacao(sentencas)