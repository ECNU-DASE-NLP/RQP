import torch
from utils import *
import math
from model.SARI import calculate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import re
sf = SmoothingFunction()

def sample(complex_sentences, simple_sentences, input_lang, tag_lang, dep_lang, lm_forward, lm_backward, embedding_weights, idf, unigram_prob):
    if os.path.isfile(config['file_name'] + '_' + config['set'] + '.txt'): 
        os.remove(config['file_name'] + '_' + config['set'] + '.txt')
    count = 0
    sari_scorel = 0
    keepl = 0
    deletel = 0
    addl = 0
    b_scorel = 0
    p_scorel = 0
    fkgl_scorel = 0
    fre_scorel = 0
    stats = {'ls':0, 'dl':0, 'las':0, 'rl':0}
    lm_forward.load_state_dict(torch.load(config['lm_name']+'.pt'))
    if config['double_LM']:
        lm_backward.load_state_dict(torch.load('structured_lm_forward_300_150_0_4_freq5.pt'))
    lm_forward.eval()
    lm_backward.eval()
    for i in range(len(complex_sentences)):
        #if i > 100 and ('val' in config['file_name']): continue
        if i % 100 == 0: print('process %s ...' %(i*1./len(complex_sentences)))
        if len(complex_sentences[i].split(' ')) <= config['min_length']:

            sl, kl, dl, al, bl, pl, fkl, frl = mcmc(complex_sentences[i], simple_sentences[i], input_lang, tag_lang, dep_lang, lm_forward, lm_backward, embedding_weights, idf, unigram_prob, stats)
            
            # print('\n')
            # print("Average sentence level SARI till now for sentences")
            sari_scorel += sl
            keepl += kl
            deletel += dl
            addl += al
            p_scorel += pl
            # print(sari_scorel/(count+1))
            # print(keepl/(count+1))
            # print(deletel/(count+1))
            # print(addl/(count+1))
            #print("Average sentence level BLEU till now for sentences")
            b_scorel += bl
            # print(b_scorel/(count+1))
            # print("Average perplexity of sentences")
            # print(p_scorel/(count+1))
            fkgl_scorel += fkl
            fre_scorel += frl
            # print('Average sentence level FKGL and FRE till now for sentences')
            # print(fkgl_scorel/(count+1))
            # print(fre_scorel/(count+1))
            # print('\n')
            # print(i+1)

            with open(config['file_name'], "a") as file:
                file.write("Average Sentence Level Perplexity, Bleu, SARI \n")
                file.write(str(p_scorel/(count+1)) + " " + str(b_scorel/(count+1)) + " " + str(sari_scorel/(count+1)) +  "\n\n")
            count += 1
    print(stats)

def mcmc(input_sent, reference, input_lang, tag_lang, dep_lang, lm_forward, lm_backward, embedding_weights, idf, unigram_prob, stats):
    # reference = 'A toothbrush in a toothbrush and toothpaste holder.'
    # input_sent = 'What part of the face is the toothbrush in?'
    reference = reference.lower()
    given_complex_sentence = input_sent.lower()
    # print(reference, given_complex_sentence); 

    orig_sent = input_sent
    beam = {}
    entities = get_entities(input_sent)
    perplexity = -10000
    perpf = -10000
    synonym_dict = {}
    sent_list = []
    spl = input_sent.lower().split(' ')

    doc=nlp(input_sent)
    elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor = tokenize_sent_special(input_sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for 
        tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
    prob_old = calculate_score(lm_forward, elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor, input_lang, input_sent, orig_sent, embedding_weights, idf, unigram_prob, False)
    if config['double_LM']:
        elmo_tensor_b, input_sent_tensor_b, tag_tensor_b, dep_tensor_b = tokenize_sent_special(reverse_sent(input_sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for 
            tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
        prob_old += calculate_score(lm_backward, elmo_tensor_b, input_sent_tensor_b, tag_tensor_b, dep_tensor_b, input_lang, reverse_sent(input_sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, False)
        prob_old /= 2.0
    # for the first time step the beam size is 1, just the original complex sentence
    beam[input_sent] = [prob_old, 'original']

    new_beam = {}
    beam2scores = {}
    # intialize the candidate beam
    for key in beam:
        # get candidate sentence through different edit operations
        cap_phrase = generate_caption_phrases(reference)
        candidates = get_subphrase_mod(input_sent, cap_phrase, idf)
        candidates.append({input_sent: 'las'})

        for i in range(len(candidates)):
            #print(candidate)
            sent = list(candidates[i].keys())[0]
            operation = candidates[i][sent]
            doc = nlp(list(candidates[i].keys())[0])
            elmo_tensor, candidate_tensor, candidate_tag_tensor, candidate_dep_tensor = tokenize_sent_special(sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for 
                tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
            # calculate score for each candidate sentence using the scoring function
            perpf, scores = calculate_score(lm_forward, elmo_tensor, candidate_tensor, candidate_tag_tensor, candidate_dep_tensor, input_lang, sent, orig_sent, embedding_weights, idf, unigram_prob, True)
            #print('scores', scores)
            p = perpf
            if config['double_LM']:
                elmo_tensor_b, candidate_tensor_b, candidate_tag_tensor_b, candidate_dep_tensor_b = tokenize_sent_special(reverse_sent(sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for 
                    tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
                perpf, scores = calculate_score(lm_backward, elmo_tensor_b, candidate_tensor_b, candidate_tag_tensor_b, candidate_dep_tensor_b, input_lang, reverse_sent(sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, True)
                #print('double LM', scores)
                p += perpf
                p /= 2.0

            # if the candidate sentence is able to increase the score by a threshold value, add it to the beam
            if True: #p > prob_old*config['threshold'][operation]:
                new_beam[sent] = [p, operation]
                beam2scores[sent] = scores
                # record the edit operation by which the candidate sentence was created
                stats[operation] += 1
            else:
                # if the threshold is not crossed, add it to a list so that the sentence is not considered in the future
                sent_list.append(sent)

        # exit()
        new_beam_sorted_list = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)
        #print(new_beam_sorted_list, new_beam); exit()
        new_beam_sorted_list = new_beam_sorted_list[:config['beam_size']]
        do_add_input_sent = False if len([1 for q in new_beam_sorted_list if q[0] == input_sent]) > 0 else True
        #print('do_add_input_sent', do_add_input_sent)
        if do_add_input_sent:
            new_beam_sorted_list += [(input_sent, new_beam[input_sent])]
        # sort the created beam on the basis of scores from the scoring function
        # print(new_beam_sorted_list)
        new_beam = {}
        # top k top scoring sentences selected. In our experiments the beam size is 1
        # copying the new_beam_sorted_list into new_beam
        for key in new_beam_sorted_list:
            new_beam[key[0]] = key[1]
        #new_beam = new_beam_sorted_list.copy()
        #print(new_beam)
        # we'll get top beam_size (or <= beam size) candidates

        # get the top scoring sentence. This will act as the source sentence for the next iteartion                
        maxvalue_sent = max(new_beam.items(), key=lambda x: x[1])[0]
        # perpf = new_beam[maxvalue_sent][0]
        # input_sent = maxvalue_sent
        # for the next iteration
        beam = new_beam.copy()

    for input_id, raw_input_sent in enumerate(new_beam):
        input_sent = raw_input_sent.lower()
        #print(given_complex_sentence)
        #print(reference)
        if input_id == 0:
            print("\n====="*10)
            print("Input complex sentence")
            print(given_complex_sentence)
            print("Reference sentence")
            print(reference)
        print("----"*10)
        print("Modified sentence")
        print(input_sent); 
        #exit()

        scorel, keepl, deletel, addl = calculate(given_complex_sentence, input_sent.lower(), [reference])
        #print(scorel)
        #print(keepl)
        #print(deletel)
        #print(addl)
        bleul = sentence_bleu([convert_to_blue(reference)], convert_to_blue(input_sent.lower()), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3)
        #print("Blue score")
        #print(bleul)
        #print("Perplexity")
        if (perpf == -10000):
            #print('sentence remain unchanged therefore calculating perp score for last generated sentence')
            doc = nlp(input_sent)
            elmo_tensor, best_input_tensor, best_tag_tensor, best_dep_tensor = tokenize_sent_special(input_sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for 
                            tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
            perpf, scores = calculate_score(lm_forward, elmo_tensor, best_input_tensor, best_tag_tensor, best_dep_tensor, input_lang, input_sent, orig_sent, embedding_weights, idf, unigram_prob, False)
            #print('scores', scores)
            if config['double_LM']:
                elmo_tensor_b, best_input_tensor_b, best_tag_tensor_b, best_dep_tensor_b = tokenize_sent_special(reverse_sent(input_sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for 
                    tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
                raw_perpf, scores= calculate_score(lm_backward, elmo_tensor_b, best_input_tensor_b, best_tag_tensor_b, best_dep_tensor_b, input_lang, reverse_sent(input_sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, False)
                perpf += raw_perpf
        #print(perpf)
        #print('fkgl and fre')
        fkgl_scorel = sentence_fkgl(input_sent)
        fre_scorel = sentence_fre(input_sent)
        #print(fkgl_scorel)
        #print(fre_scorel)
        with open(config['file_name'] + '_' + config['set'] + '.txt', "a") as file:
            if input_id == 0:
                file.write("\n")
                file.write('given_complex_sentence\t' + given_complex_sentence + "\n")
                file.write('reference\t' + reference + "\n")
                #file.write(final_sent.lower() + "\n")
                #file.write(str(perplexity) + " " + str(bleu) + " " + str(score) + " " + str(keep) + " " + str(delete) + " " + str(add) + " " + str(fkgl_score) + " " + str(fre_score) + "\n")
            print(beam2scores[raw_input_sent])
            file.write('input_sent %s\t' %input_id + input_sent.lower() + "\n")
            file.write(str(beam2scores[raw_input_sent]) + '\n')
            file.write(str(perpf) + " " + str(bleul) + " " + str(scorel) + " " + str(keepl) + " " + str(deletel) + " " + str(addl) + " " + str(fkgl_scorel) + " " + str(fre_scorel) + "\n")
    return scorel, keepl, deletel, addl, bleul, perpf, fkgl_scorel, fre_scorel

def mcmc_backup(input_sent, reference, input_lang, tag_lang, dep_lang, lm_forward, lm_backward, embedding_weights, idf, unigram_prob, stats):
    print(stats)
    #input_sent = "highlights 2009 from the 2009 version of 52 seconds setup for passmark 5 32 5 2nd scan time , and 7 mb memory- 7 mb memory ."
    reference = reference.lower()
    given_complex_sentence = input_sent.lower()
    #final_sent = input_sent
    orig_sent = input_sent
    #print(given_complex_sentence)
    beam = {}
    entities = get_entities(input_sent)
    perplexity = -10000
    perpf = -10000
    synonym_dict = {}
    sent_list = []
    spl = input_sent.lower().split(' ')
    #print(spl)
    # the for loop below is just in case if the edit operations go for a very long time
    # in almost all the cases this will not be required
    for iter in range(2*len(spl)):

        doc=nlp(input_sent)
        elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor = tokenize_sent_special(input_sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for 
            tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
        prob_old = 0. #calculate_score(lm_forward, elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor, input_lang, input_sent, orig_sent, embedding_weights, idf, unigram_prob, False)
        if config['double_LM']:
            elmo_tensor_b, input_sent_tensor_b, tag_tensor_b, dep_tensor_b = tokenize_sent_special(reverse_sent(input_sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for 
                tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
            prob_old += calculate_score(lm_backward, elmo_tensor_b, input_sent_tensor_b, tag_tensor_b, dep_tensor_b, input_lang, reverse_sent(input_sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, False)
            prob_old /= 2.0
        # for the first time step the beam size is 1, just the original complex sentence
        if iter == 0:
            beam[input_sent] = [prob_old, 'original']
        print('Getting candidates for iteration: ', iter)
        #print(input_sent)
        new_beam = {}
        # intialize the candidate beam
        for key in beam:
            # get candidate sentence through different edit operations
            candidates = get_subphrase_mod(key, sent_list, input_lang, idf, spl, entities, synonym_dict)
            '''if len(candidates) == 0:
                break'''

            for i in range(len(candidates)):
                print(candidate)
                sent = list(candidates[i].keys())[0]
                operation = candidates[i][sent]
                doc=nlp(list(candidates[i].keys())[0])
                elmo_tensor, candidate_tensor, candidate_tag_tensor, candidate_dep_tensor = tokenize_sent_special(sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for 
                    tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
                # calculate score for each candidate sentence using the scoring function
                p = 0. #calculate_score(lm_forward, elmo_tensor, candidate_tensor, candidate_tag_tensor, candidate_dep_tensor, input_lang, sent, orig_sent, embedding_weights, idf, unigram_prob, True)
                if config['double_LM']:
                    elmo_tensor_b, candidate_tensor_b, candidate_tag_tensor_b, candidate_dep_tensor_b = tokenize_sent_special(reverse_sent(sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for 
                        tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
                    p += 0. #calculate_score(lm_backward, elmo_tensor_b, candidate_tensor_b, candidate_tag_tensor_b, candidate_dep_tensor_b, input_lang, reverse_sent(sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, True)
                    p /= 2.0

                # if the candidate sentence is able to increase the score by a threshold value, add it to the beam
                if p > prob_old*config['threshold'][operation]:
                    new_beam[sent] = [p, operation]
                    # record the edit operation by which the candidate sentence was created
                    stats[operation] += 1
                else:
                    # if the threshold is not crossed, add it to a list so that the sentence is not considered in the future
                    sent_list.append(sent)
            exit()

        if new_beam == {}:
            # if there are no candidate sentences, exit
            break
        #print(new_beam)
        new_beam_sorted_list = sorted(new_beam.items(), key=lambda x: x[1])[-config['beam_size']:]
        # sort the created beam on the basis of scores from the scoring function
        #print(new_beam_sorted_list)
        new_beam = {}
        # top k top scoring sentences selected. In our experiments the beam size is 1
        # copying the new_beam_sorted_list into new_beam
        for key in new_beam_sorted_list:
            new_beam[key[0]] = key[1]
        #new_beam = new_beam_sorted_list.copy()
        #print(new_beam)
        # we'll get top beam_size (or <= beam size) candidates

        # get the top scoring sentence. This will act as the source sentence for the next iteartion                
        maxvalue_sent = max(new_beam.items(), key=lambda x: x[1])[0]
        perpf = new_beam[maxvalue_sent][0]
        input_sent = maxvalue_sent
        # for the next iteration
        beam = new_beam.copy()


    input_sent = input_sent.lower()
    #print(given_complex_sentence)
    #print(reference)
    print("====="*10)
    print("Input complex sentence")
    print(given_complex_sentence)
    print("Reference sentence")
    print(reference)
    print("Simplified sentence")
    print(input_sent)
    
    scorel, keepl, deletel, addl = calculate(given_complex_sentence, input_sent.lower(), [reference])
    #print(scorel)
    #print(keepl)
    #print(deletel)
    #print(addl)
    bleul = sentence_bleu([convert_to_blue(reference)], convert_to_blue(input_sent.lower()), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3)
    #print("Blue score")
    #print(bleul)
    #print("Perplexity")
    if (perpf == -10000):
        #print('sentence remain unchanged therefore calculating perp score for last generated sentence')
        doc = nlp(input_sent)
        elmo_tensor, best_input_tensor, best_tag_tensor, best_dep_tensor = tokenize_sent_special(input_sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for 
                        tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
        perpf = 0. #calculate_score(lm_forward, elmo_tensor, best_input_tensor, best_tag_tensor, best_dep_tensor, input_lang, input_sent, orig_sent, embedding_weights, idf, unigram_prob, False)
        if config['double_LM']:
            elmo_tensor_b, best_input_tensor_b, best_tag_tensor_b, best_dep_tensor_b = tokenize_sent_special(reverse_sent(input_sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for 
                tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
            perpf += calculate_score(lm_backward, elmo_tensor_b, best_input_tensor_b, best_tag_tensor_b, best_dep_tensor_b, input_lang, reverse_sent(input_sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, False)
    #print(perpf)
    #print('fkgl and fre')
    fkgl_scorel = sentence_fkgl(input_sent)
    fre_scorel = sentence_fre(input_sent)
    #print(fkgl_scorel)
    #print(fre_scorel)
    with open(config['file_name'] + '_' + config['set'] + '.txt', "a") as file:
        file.write(given_complex_sentence + "\n")
        file.write(reference + "\n")
        #file.write(final_sent.lower() + "\n")
        #file.write(str(perplexity) + " " + str(bleu) + " " + str(score) + " " + str(keep) + " " + str(delete) + " " + str(add) + " " + str(fkgl_score) + " " + str(fre_score) + "\n")
        file.write(input_sent.lower() + "\n")
        file.write(str(perpf) + " " + str(bleul) + " " + str(scorel) + " " + str(keepl) + " " + str(deletel) + " " + str(addl) + " " + str(fkgl_scorel) + " " + str(fre_scorel) + "\n")
        file.write("\n")
    return scorel, keepl, deletel, addl, bleul, perpf, fkgl_scorel, fre_scorel
