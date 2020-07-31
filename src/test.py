import spacy
import stanfordnlp
import warnings
import re
import Util
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
stanfordModel = stanfordnlp.Pipeline(**config)  
# print("--------")

# dep1 = stanfordModel("A capital da russia")

# for sent in dep1.sentences:
#     for word in sent.words:
#         s2 += word.upos+" "

# print(s1,"\n", s2)

# t1 = re.search("DET? NOUN ADJ", s1).group()

# t2  = re.search("DET? NOUN PROP? ADP DET? (PROP|NOUN)", s2).group()
# print(s1)
# print(t1, t2)
def hasNeg(frase):
    frase = frase.lower()
    token = frase.split(' ')
    negacao = ["jamais", "de modo algum","de jeito nenhum", "de forma alguma", "nunca", "não", "absolutamente", "nem", "tampouco"]
    for i in range(0, len(token)):
        aux = token[i]
        if token[i] == "de" and (token[i+1] == "modo" or token[i+1] == "jeito" or token[i+1] == "forma" ):
            aux = token[i] + " " + token[i+1] + " " + token[i+2]
        for j in negacao:
            if aux == j:
                return True
    return False

def dependenceParsySpacy(frase):
    nlp = spacy.load("pt_core_news_sm")
    doc = nlp(frase)
    ldep = []
    for token in doc:
        ldep.append({"text":token.text, "rel":token.dep_, "children":[child for child in token.children]})
    return ldep
def lcsr(s1, s2):
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
def subjEquals(dep1, dep2):
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
    
    subj1 = subj[0]['text']
    for i in subj[0]['children']:
        subj1 += " "+str(i)

    subj2 = subj[1]['text']
    for i in subj[1]['children']:
        subj2 += " "+str(i)

    lcsrsub = lcsr(subj1, subj2)
    
    root1 = root[0]['text']
    for i in root[0]['children']:
        root1 += " "+str(i)

    root2 = root[1]['text']
    for i in root[1]['children']:
        root2 += " "+str(i)

    print(root1,"\n", root2)
    lcsrroot = lcsr(root1, root2)
    
    print(root, subj, lcsrsub, lcsrroot)


def isAntonimo(palavra1, palavra2):
        lantonimos = Util.readJson("lista_antonimos.json")
        palavra1.lower()
        palavra2.lower()
        doc = stanfordModel(palavra1)
        doc2 = stanfordModel(palavra2)
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

def getExpressions():
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

def regularExpression(s1, s2):
    expressions = getExpressions()
    doc = stanfordModel(s1)
    doc2 = stanfordModel(s2)
    s1l = []
    s2l = []
    s1p = ''
    s2p = ''
    count = 0
    for sent in doc.sentences:
        for token in sent.words:
            s1l.append({"palavra":token.lemma, "upos":token.upos})
    for sent in doc.sentences:
        for token in sent.words:
            s2l.append({"palavra":token.lemma, "upos":token.upos})
    
    for i in s1l:
        s1p+=i['upos']+' '
    for i in s2l:
        s2p+=i['upos']+' ' 
    #test = re.compile(r'(DET)?NOUN{1} (PROPN )?ADP (DET )?(PROPN|NOUN)')
    print("s1: ", s1p)
    print("s2: ", s2p)
    for i in expressions:
        try:
            if (re.search(i[0], s1p) and re.search(i[1], s2p)) or (re.search(i[1], s1p) and re.search(i[0], s2p)) or (re.search(i[0], s1p) and re.search(i[0], s2p)) or (re.search(i[1], s1p) and re.search(i[1], s2p)):
                count+=1
        except:
            pass
    
    print(count)

def getHiperonimo():
    arq = open('./databases/triplos_todos_10recs_n.txt', 'r', encoding="utf8")
    lhipe = []
    for i in arq:
        aux = i.split(' ')
        if aux[1] == "HIPERONIMO_DE":
            lhipe.append({'palavra':aux[0], 'hiperonimo_de':aux[2].split('\t')[0]})
        
    Util.writeJson(lhipe, "hiperonimos")
    
def getAntonimo():
    arq = open('../databases/triplos_todos_10recs_n.txt', 'r', encoding="utf8")
    lhipe = []
    for i in arq:
        aux = i.split(' ')
        tipo = ""
        if aux[1] == "ANTONIMO_N_DE":
            tipo = "NOUN"
            lhipe.append({'tipo':tipo,'palavra':aux[0], 'ANTONIMO_de':aux[2].split('\t')[0]})
        elif aux[1] == "ANTONIMO_V_DE":
            tipo = "VERB"
            lhipe.append({'tipo':tipo,'palavra':aux[0], 'ANTONIMO_de':aux[2].split('\t')[0]})
        elif aux[1] == "ANTONIMO_ADV_DE":
            tipo = "ADV"
            lhipe.append({'tipo':tipo,'palavra':aux[0], 'ANTONIMO_de':aux[2].split('\t')[0]})
        elif aux[1] == "ANTONIMO_ADJ_DE":
            tipo = "ADJ"
            lhipe.append({'tipo':tipo,'palavra':aux[0], 'ANTONIMO_de':aux[2].split('\t')[0]})
        
    Util.writeJson(lhipe, "antonimos")

def getSinPapel():
    arq = open('./databases/triplos_todos_10recs_n.txt', 'r', encoding="utf8")
    lhipe = []
    for i in arq:
        aux = i.split(' ')
        tipo = ""
        if aux[1] == "SINONIMO_N_DE":
            tipo = "NOUN"
            lhipe.append({'tipo':tipo,'palavra':aux[0], 'sinonimo_de':aux[2].split('\t')[0]})
        elif aux[1] == "SINONIMO_V_DE":
            tipo = "VERB"
            lhipe.append({'tipo':tipo,'palavra':aux[0], 'sinonimo_de':aux[2].split('\t')[0]})
        elif aux[1] == "SINONIMO_ADV_DE":
            tipo = "ADV"
            lhipe.append({'tipo':tipo,'palavra':aux[0], 'sinonimo_de':aux[2].split('\t')[0]})
        elif aux[1] == "SINONIMO_ADJ_DE":
            tipo = "ADJ"
            lhipe.append({'tipo':tipo,'palavra':aux[0], 'sinonimo_de':aux[2].split('\t')[0]})
        
    Util.writeJson(lhipe, "sinonimos")

def isHiperonimo(p1, p2):
    hipe = Util.readJson("hiperonimo.json")
    
    for i in hipe:
        if (p1 == i['palavra'] and p2 == i['hiperonimo_de']) or (p2 == i['palavra'] and p1 == i['hiperonimo_de']):
            return True
    
    return False

def isSinonimosPapel(p1, p2):
    hipe = Util.readJson("sinonimos.json")
    doc = stanfordModel(p1)
    doc1 = stanfordModel(p2)
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
            if (p1 == i['palavra'] and p2 == i['sinonimo_de']) or (p2 == i['palavra'] and p1 == i['sinonimo_de']):
                return True
        elif p1t != p2t:
            return False
    
    return False
def isAntonimosPapel(p1, p2):
    hipe = Util.readJson("antonimos.json")
    doc = stanfordModel(p1)
    doc1 = stanfordModel(p2)
    p1t = ""
    p2t = ""
    for s in doc.sentences:
        for t in s.words:
            p1t = t.upos
            
    for s in doc1.sentences:
        for t in s.words:
            p2t = t.upos
    print(p1t, p2t)             
    for i in hipe:
        if p1t == p2t and p1t == i['tipo']:
            #print(i['palavra'], p1, p2)
            if (p1 == i['palavra'] or p2 == i['palavra']):
                for k in i['antonimos']:
                    print(k, p1)
                    print(k, p2)
                    if k == p1 or k == p2:
                        return True
        elif p1t != p2t:
            return False
    
    return False
def countHiperonimo(s1, s2):
    count = 0
    for i in s1:
        for k in s2:
            if isHiperonimo(i, k):
                count += 1
                s2.remove(k)
                break
        print(s2)
    return count
        
def arrumarSinonimos():
    sin = Util.readJson('sinonimos.json')
    i = 0
    total = len(sin)
    nlist = []
    while i < total:
        aux = []
        palavra = sin[i]['palavra']
        tipo = sin[i]['tipo']
        print(i)
        for k in sin:
            if k['tipo'] == tipo and palavra == k['palavra']:
                aux.append(k['sinonimo_de'])
                sin.remove(k)
                total-=1        
        nlist.append({"palavra":palavra, "sinonimos":aux, "tipo":tipo})
        i+=1

    Util.writeJson(nlist, 'sinonimos')
    
def arrumarAntonimos():
    sin = Util.readJson('./antonimos.json')
    i = 0
    total = len(sin)
    nlist = []
    while i < total:
        aux = []
        palavra = sin[i]['palavra']
        tipo = sin[i]['tipo']
        print(i)
        for k in sin:
            if k['tipo'] == tipo and palavra == k['palavra']:
                aux.append(k['ANTONIMO_de'])
                sin.remove(k)
                total-=1        
        nlist.append({"palavra":palavra, "antonimos":aux, "tipo":tipo})
        i+=1

    Util.writeJson(nlist, 'antonimos')

def arrumarHiperonimos():
    sin = Util.readJson('hiperonimos.json')
    i = 0
    total = len(sin)
    nlist = []
    while i < total:
        aux = []
        palavra = sin[i]['palavra']
        print(i)
        for k in sin:
            if palavra == k['palavra']:
                aux.append(k['hiperonimo_de'])
                sin.remove(k)
                total-=1        
        nlist.append({"palavra":palavra, "hiperonimos":aux})
        i+=1

    Util.writeJson(nlist, 'hiperonimos')
# s1 = "O veículo estava parado"
# s2 = "O carro estava parado"
# print(countHiperonimo(s1.split(' '),s2.split(' ') ))

# teste = getSinPapel()
# print(teste[4])

#print(isSinonimosPapel("estudar", "meditar"))
print(isAntonimosPapel("improvável", "provável"))
#Util.writeJson(teste , "sinonimosPapel")  
#regularExpression("A direção da Câmara", "A mesa Diretora da Câmara")
# dep1 = dependenceParsySpacy("Batatas estão sendo fatiadas por um homem")
# dep2 = dependenceParsySpacy("O homem está fatiando a batata")
# print(dep1, dep2)
# subjEquals(dep1, dep2)

#lcsr("Lula da Silva", "Luiz Inacio Lula da Silva")
# print(hasNeg(frase))
# frase = "De jeito nenhum irei para a escola amanhã de noite"
# print(hasNeg(frase))
# frase = "De forma alguma irei para a escola amanhã de noite"
# print(hasNeg(frase))
# frase = "De modo algum irei para a escola amanhã de noite"
# print(hasNeg(frase))
# frase = "Jamais iria fazer isso"
# print(hasNeg(frase))
# frase = "Se não vou amanhã, tampouco irei depois"
# print(hasNeg(frase))
# frase = "Não irei amanhã pra escola"
# print(hasNeg(frase))
# frase = "Nem vou pra escola amanhã"
# print(hasNeg(frase))
# frase = "Nunca vi essa flor antes"
# print(hasNeg(frase))
# frase = "Amanhã concerteza irei para a escola"
# print(hasNeg(frase))