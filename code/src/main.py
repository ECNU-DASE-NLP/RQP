from utils import *
print(device)

from model.structural_decoder import DecoderGRU

idf, unigram_prob, output_lang, tag_lang, dep_lang, train_simple, valid_simple, test_simple, train_complex, valid_complex, test_complex, output_embedding_weights, tag_embedding_weights, dep_embedding_weights = prepareData(config['embedding_dim'], 
config['freq'], config['ver'], config['dataset'], config['operation'])

lm_forward = DecoderGRU(config['hidden_size'], output_lang.n_words, tag_lang.n_words, dep_lang.n_words, config['num_layers'], 
	output_embedding_weights, tag_embedding_weights, dep_embedding_weights, config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['dropout'], config['use_structural_as_standard']).to(device)
lm_backward = DecoderGRU(config['hidden_size'], output_lang.n_words, tag_lang.n_words, dep_lang.n_words, config['num_layers'], 
	output_embedding_weights, tag_embedding_weights, dep_embedding_weights, config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['dropout'], config['use_structural_as_standard']).to(device)

from tree_edits_beam import *
if config['set'] == 'valid':
	sample(valid_complex, valid_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward, output_embedding_weights, idf, unigram_prob)
elif config['set'] == 'test':
	sample(test_complex, test_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward, output_embedding_weights, idf, unigram_prob)

'''
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
    -preload tokenize,ssplit,pos,lemma,parse,depparse \
    -status_port 9000 -port 9000 -timeout 15000

python src/main.py
'''
