import stanfordnlp
import spacy
import gensim
from gensim.models import KeyedVectors

#Esta classe carrega os modelos treinados necessário para rodar o código, sendo eles spacy, stanfordnlp e word embeddings
class LoadModels:
    def __init__(self):
        pass
    def loadModelStanfordNLP(self):
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
        model = stanfordnlp.Pipeline(**config)
        return model
    def loadWorldEmbeddings(self):
        model = KeyedVectors.load_word2vec_format('../skip_s300.txt')
        return model
    def loadSpacy(self):
        model = spacy.load('pt_core_news_sm')
        return model