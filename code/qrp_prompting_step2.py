import os
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
import random
import copy
import re
import torch
import openai
import ast
from tqdm import tqdm
import transformers
from transformers import GPT2Config, GPT2Model
from transformers import GPT2Tokenizer
from collections import Counter, defaultdict
from transformers import BertTokenizer, BertModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_selection, do_auto_device = False):
    if do_auto_device:
        model = AutoModelForCausalLM.from_pretrained(model_selection, cache_dir = './data/cache/', #model_selection,
                         local_files_only=False, device_map='auto') #~/autodl-tmp/huggingface/hub/models--facebook--opt-6.7b
    else:
        model = AutoModelForCausalLM.from_pretrained(model_selection, cache_dir='./data/cache/',  # model_selection,
                        local_files_only=False) #, device_map='auto'

    tokenizer = AutoTokenizer.from_pretrained(model_selection, cache_dir = './data/cache/',
                                              use_fast=False)
    return model,tokenizer

def load_gpt_cache(llm):
    cache = {}
    if 'opt' in llm:
        if os.path.isfile('./data/cache/opt_cache.json'):
            cache = {} 
    else:
        if os.path.isfile('./data/cache/gpt3_cache.json'):
            cache = json.load(open('./data/cache/gpt3_cache.json'))

    return cache


def process_opt_answer(answer):
    answer = re.sub('^[^a-zA-Z]+|[^a-zA-Z]+$', '', answer)
    answer = re.split('[^a-zA-Z ]', answer)[0].lower()
    answer = re.sub('^ | $', '', answer)
    return answer


def load_gpt_cache_for_inference():
    new_cache = defaultdict(list)
    cache = json.load(open('./data/cache/gpt3_cache.json'))

    for single_prompt in cache:
        if len(re.findall('\n===\nQ:', single_prompt)) == 2:
            question = single_prompt.split('\n===\nQ: ')[-1].split('\nA:')[0]
            new_cache[question] += [(single_prompt, cache[single_prompt])]
    return new_cache


def save_gpt3_cache(cache, edit_dict, llm='gpt3'):
    for line_idx, line in enumerate(edit_dict):
        for single_prompt in line:
            cache[single_prompt['prompt']] = single_prompt['pred_answer']

    if 'opt' in llm:
        g_file = open('./data/cache/opt_cache.json', 'w')
    else:
        g_file = open('./data/cache/gpt3_cache.json', 'w')

    json.dump(cache, g_file)
    g_file.close()


def process_answer(answer):
    answer = answer.replace('.', '').replace(',', '').lower()
    to_be_removed = {'a', 'an', 'the', 'to', ''}
    answer_list = answer.split(' ')
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return ' '.join(answer_list)


def load_anno(coco_caption_file, answer_anno_file, edit_file, question_anno_file, similarity_metric):
    if coco_caption_file is not None:
        coco_caption = json.load(open(coco_caption_file, 'r'))
        if type(coco_caption) == type({}): coco_caption = coco_caption['annotations']
    if answer_anno_file is not None:
        answer_anno = json.load(open(answer_anno_file, 'r'))
    question_anno = json.load(open(question_anno_file, 'r'))
    edit_anno = json.load(open(edit_file, 'r'))

    caption_dict = {}
    if coco_caption_file is not None:
        for sample in coco_caption:
            if sample['image_id'] not in caption_dict:
                caption_dict[sample['image_id']] = [sample['caption']]
            else:
                caption_dict[sample['image_id']].append(sample['caption'])

    answer_dict = {}
    if answer_anno_file is not None:
        for sample in answer_anno['annotations']:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in answer_dict:
                answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = [x['answer'] for x in
                                                                                             sample['answers']]

    edit_dict = {}
    # print(question_anno.keys()); exit()
    for sample in edit_anno:
        if str(sample['image']) + '<->' + str(sample['qid']) not in edit_dict and len(sample['edit']):
            edit_dict[str(sample['image']) + '<->' + str(sample['qid'])] = sample['edit']
        else:
            edit_dict[str(sample['image']) + '<->' + str(sample['qid'])] = sample['question']

    question_dict, choice_dict = {}, {}
    for sample in question_anno['questions']:
        # print('sample', sample); exit()
        if str(sample['image_id']) + '<->' + str(sample['question_id']) not in question_dict:
            question_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample['question']
            if 'choices' in sample:
                choice_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = '; '.join(sample['choices'])

    return caption_dict, answer_dict, edit_dict, question_dict


def load_edit(args_edit_path, edit_path, additional_information=''):
    g_file = open('./results/%s.json' % args_edit_path, 'r')
    # print('edit_path', edit_path)
    edit_dict = json.load(open(edit_path, 'r'))

    q2edit_prob = defaultdict(int)
    for qs in edit_dict:
        for qid, q in enumerate(qs['edit']):
            prob = 1. if ('prob' not in qs) else float(qs['prob'][qid])
            q2edit_prob[(str(qs['qid']), q)] = prob
        #print(q2edit_prob); exit()

    edit_dict, lines = [], ''
    for line in g_file:
        lines += line.strip()
        if ']' in line:
            edit_dict += [ast.literal_eval(re.sub('\[$', '', lines))]
            lines = '['

    qid2edit = defaultdict(set)
    for q in edit_dict:
        qid2prob = defaultdict(int)
        pred = set()
        for a_id, a in enumerate(q):
            key = a['key']
            imgid, qid = key.split('<->')
            pred.add(a['pred_answer'])
            #print(np.exp(a['prob']), np.exp(q2edit_prob[(qid, a['question'])]))
            qid2prob[a['pred_answer']] += (np.exp(a['prob'])*np.exp(q2edit_prob[(qid, a['question'].lower())]))
        q_prob = np.array([qid2prob[a] for a in qid2prob])
        q_prob = q_prob*1./(np.sum(q_prob)+0.1e-5)
        #print('qid2prob', qid2prob); exit()
        if True: #additional_information == '':
            for a_id, a in enumerate(qid2prob):
                qid2edit[key].add(a+ '(' +str(np.round(q_prob[a_id], 2)) + ')')# #
        elif additional_information == 'prob':
            a_id = q_prob.argmax()
            qid2edit[key].add(list(qid2prob.keys())[a_id]+'(1.0)')
    #print('qid2edit', qid2edit)

    qid2tem = {}
    for q in qid2edit:
        qid2tem[q] = '; '.join(list(qid2edit[q]))
    #print(qid2tem); exit()
    return qid2tem

def load_choice(choice_path):
    g_file = open('./data/%s.json' %choice_path, 'r')

    edit_dict, lines = [], ''
    for line in g_file:
        lines += line.strip()
        if ']' in line:
            edit_dict += [ast.literal_eval(re.sub('\[$', '', lines))]
            lines = '['

    qid2edit = {}
    for q in edit_dict:
        for a in q:
            qid2edit[a['key']] = sorted(a['pred_answer'])

    return qid2edit

def load_exemplar(data_file):
    examplars = json.load(open(data_file))
    key2context = {}

    prompt = ''
    for examplar in examplars['questions']:
        #print(examplar); exit()
        key = str(examplar['image_id']) + '<->' + str(examplar['question_id'])

        prompt = "Please reason the answer of the questions according to the given contexts.\n===\n"

        for cap_id, caption in enumerate(examplar['captions'][:5]):
            prompt += 'Context: %s \n===\n' %caption
            qa = examplar['qa_Synthetic'][cap_id]
            prompt += 'Question: %s\nAnswer: %s\n\n===\n'%(qa['question'], qa['answer'])


        key2context[key] = prompt
    return key2context

def read_apikey(key_path):
    apikeys = []
    with open(key_path, 'r')  as f:
        for line in f.readlines():
            line = line.strip()
            _, _, apikey = line.split('----')
            apikeys += [apikey]
    #print(apikeys)
    return apikeys


class Answer_Choosing:
    def __init__(self, args):
        self.args = args
        self.args.dataset, self.args.mode, self.args.sample, self.args.llm, self.args.n_shot, self.args.edit_num, self.args.baseline = \
            self.args.edit_path.split('_')[:7]
        self.args.n_shot = int(self.args.n_shot[5:])
        self.args.edit_num = int(self.args.edit_num[7:])
        #self.args.sample = '2097-10000'

        self.similarity_metric = self.args.similarity_metric
        ## loading input questions (and answer for reference accuracy computing)
        similarity_metric = 'imagequestion'

        if self.args.dataset == 'okvqa':
            question_path = '%s/OpenEnded_mscoco_val2014_questions.json'%args.coco_path
            answer_path = '%s/mscoco_val2014_annotations.json'%args.coco_path
            edit_path = './data/okvqa/edits_val.json'
        elif self.args.dataset == 'aokvqa-val':
            question_path = './data/aokvqa/aokvqa_val_qusetions.json'
            answer_path = './data/aokvqa/aokvqa_val_annotations.json'
            edit_path = './data/aokvqa/edits_val.json'
        elif self.args.dataset == 'aokvqa-test':
            question_path = './data/aokvqa/aokvqa_test_qusetions.json'
            answer_path = None
            edit_path = './data/aokvqa/edits_test.json'

        _, self.answer_dict, self.edit_dict, self.question_dict = \
            load_anno(None, \
                answer_path, edit_path, question_path, \
                similarity_metric) #self.args.similarity_metric

        # print(self.question_dict)
        self.val_keys = list(self.question_dict.keys())

        ## load cached image representation (Coco caption & Tags)
        self.inputtext_dict = self.load_cachetext()

        self.traincontext_caption_dict, self.traincontext_answer_dict, self.traincontext_edit_dict, self.traincontext_question_dict = \
            load_anno('%s/captions_train2014.json'%args.coco_path, \
                '%s/mscoco_train2014_annotations.json'%args.coco_path, \
                './data/okvqa/edits_train.json', \
                '%s/OpenEnded_mscoco_train2014_questions.json'%args.coco_path , \
                self.similarity_metric)
        self.train_keys = list(self.traincontext_answer_dict.keys())

        #print(args.edit_path); exit()
        self.img2edit = load_edit(args.edit_path, edit_path, self.args.additional_information)
        #print(self.img2edit)
        #print('self.edit_dict', self.edit_dict)
        self.load_similarity()
        if torch.cuda.is_available() and self.args.gpu_id != -1:
            print("cuda {} is available".format(self.args.gpu_id))
            self.device = torch.device("cuda", self.args.gpu_id)  #
            args.gpu_id = 1
        else:
            self.device = None
            print("cuda is unavailable")

        if 'opt' in self.args.llm:
            self.gpt3_cache = load_gpt_cache(self.args.llm)
            self.llm_model, self.tokenizer = load_model('facebook/%s' %self.args.llm)
            #self.llm_model = self.llm_model.half()
            if self.device:
                self.llm_model = self.llm_model.half().to(self.device)
            #exit()
        elif self.args.llm in ['gpt-neo-2.7b', 'gpt-j-6B']:
            self.gpt3_cache = load_gpt_cache(self.args.llm)
            self.llm_model, self.tokenizer = load_model('EleutherAI/%s' %self.args.llm)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.device:
                self.llm_model = self.llm_model.half().to(self.device) #.half()
        elif self.args.llm in ['bloom-7.1b']:
            self.gpt3_cache = load_gpt_cache(self.args.llm)
            self.llm_model, self.tokenizer = load_model('bigscience/bloomz-7b1', True)
        elif self.args.llm == 'gpt3':
            self.gpt3_cache = load_gpt_cache(self.args.llm)
            #self.gpt3_cache_inference = load_gpt_cache_for_inference()
            self.apikeys = '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'
        elif self.args.llm == 'gpt3.5':
            self.gpt3_cache = {}
            self.apikeys = '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'
            openai.api_key = self.apikeys

    def inference(self):
        out_path = '%s_choose_%s' %(self.args.edit_path, self.args.additional_information)
        if os.path.isfile('./results/%s.json' %out_path):
            os.remove('./results/%s.json' %out_path)
        out_file = open('./results/%s.json' %out_path, 'a')

        answers = []
        all_answers, cache_answers = [], []
        for key_id, key in enumerate(tqdm(self.val_keys)):
            # if int(key.split('<->')[1]) not in self.img2edit:
            #     continue
            # if key != '301753<->3017535':
            #     continue
            start_id, end_id = self.args.sample.split('-')
            if not (int(end_id) > key_id >= int(start_id)):  # 2 or key_id < 1:
                continue
            answer, pred_acc = self.sample_inference(key)
            # print('answer', answer); #exit()
            if answer[0]['pred_answer'] == 'NA':
                continue

            # pred_ans = [a['pred_answer'] for a in answer]
            # pred_acc = Counter(pred_ans)
            # print('answer', answer); #exit()

            # print('pred_acc', pred_acc); exit()
            max_acc = sorted(pred_acc.items(), key=lambda x: x[1], reverse=True)
            all_answers.append(answer)
            cache_answers.append(answer)
            answers.append([ans for ans in answer if ans['question'] == max_acc[0][0]])
            # print('answers', answers); exit()

            # if np.random.random() < 0.2:
            #     print('now stop for 10s ...')
            #     time.sleep(10)
            #     print('continue ...')

            json.dump(answer, out_file, indent=6)

            if key_id % self.args.save_cache == 0:
                # print('self.args.llm', self.args.llm)
                save_gpt3_cache(self.gpt3_cache, cache_answers, self.args.llm)
                cache_answers = []
        out_file.close()
        return answers

    def sample_inference(self, key):
        if self.args.dataset in ['okvqa']:
            img_key, qid= int(key.split('<->')[0]), int(key.split('<->')[1])
        else:
            img_key, qid = int(key.split('<->')[0]), key.split('<->')[1]

        question, caption = self.question_dict[key], self.inputtext_dict[img_key]
        answer = 'NA'
        if len(self.answer_dict) > 0:
            answer = self.answer_dict[key]

        caption_i = caption[random.randint(0, len(caption) - 1)]
        ## select one caption if exists multiple, not true except COCO GT (5)

        #print(self.img2edit)
        questions = ['%s\nCandidates: %s' %(question, self.img2edit[key])]
        #print(questions); exit()
        #questions = ['%s' % (question)]

        #print('questions', questions); # exit()
        pred_answer_list, pred_prob_list = [], []
        context_key_list = self.get_context_keys(key, self.args.similarity_metric, self.args.n_shot * self.args.n_ensemble)

        pred_ans, pred_acc = [], {}
        for question in questions:
            for repeat in range(self.args.n_ensemble):
                if self.args.baseline in ['image2prompt']:
                    #prompt = 'Please answer the question according to the above context.\n'
                    prompt = self.valkey2prompt[key]
                else:
                    ## prompt format following GPT-3 QA API
                    prompt = 'Please choose the answer from choices according to the above context.\n===\n'
                    for ni in range(self.args.n_shot):
                        if context_key_list is None:
                            context_key = self.train_keys[random.randint(0,len(self.train_keys)-1)]
                        else:
                            context_key = context_key_list[ni+self.args.n_shot*repeat]
                        img_context_key = int(context_key.split('<->')[0])
                        while True: ## make sure get context with valid question and answer
                            if len(self.traincontext_question_dict[context_key])!=0 and len(self.traincontext_answer_dict[context_key][0])!=0:
                                break
                            context_key = self.train_keys[random.randint(0,len(self.train_keys)-1)]
                        if self.similarity_metric == 'imagequestion':
                            context = self.traincontext_caption_dict[img_context_key][random.randint(0,len(self.traincontext_caption_dict[img_context_key])-1)]
                            prompt += 'Context: %s\n===\n'%self.traincontext_caption_dict[img_context_key][random.randint(0,len(self.traincontext_caption_dict[img_context_key])-1)]
                        # print(self.traincontext_question_dict[context_key])

                        question_dict = self.traincontext_question_dict[context_key]
                        prompt += 'Question: %s\nAnswer: %s\n\n===\n'%(question_dict, self.traincontext_answer_dict[context_key][0])

                #prompt += '''Please choose the answer from candidates.\n'''

                if True:
                    prompt += '''Context: a glass of wine sitting next to a laptop.. computer, laptop, indoor, table, candle, drink, electronic device, personal computer, computer hardware, electronics\n'''
                    question_dict = 'Why is this plugged in?'
                    answer_dict = 'charge'
                    #prompt += 'Question: %s\nAnswer: %s\n\n===\n' % (question_dict, answer_dict)

                    if np.random.random() < 0.5:
                        prompt += 'Question: %s\nCandidates: %s\nAnswer: %s\n\n===\n'%(question_dict, 'drink; charge', answer_dict) #
                    else:
                        prompt += 'Question: %s\nCandidates: %s\nAnswer: %s\n\n===\n'%(question_dict, 'charge; drink', answer_dict)


                if self.similarity_metric == 'imagequestion':
                    prompt += 'Context: %s\n' % caption_i
                prompt += 'Question: %s\nAnswer:' % question

                if ';' not in self.img2edit[key]:
                    counter = 0
                    for ii in range(len(answer)):
                        ans = self.img2edit[key]
                        if True: #'opt' in self.args.llm:
                            ans = process_answer(process_opt_answer(ans))
                        if ans == answer[ii]: counter += 1
                    pred_ans.append(
                        {'key': key, 'pred_answer': ans, 'prompt': prompt, 'question': question,
                         'acc': min(1., float(counter) * 0.3)});
                    pred_acc[question] = pred_ans[-1]['acc']
                    # print(pred_acc[question])
                    return pred_ans, pred_acc

                #print(prompt)
                if ('opt' in self.args.llm) or (self.args.llm in ['gpt-neo-2.7b', 'gpt-j-6B', 'bloom-7.1b']):
                    pred_answer, pred_prob = self.generate_via_opt(self.llm_model, self.tokenizer, prompt)
                elif self.args.llm == 'gpt3':
                    pred_answer, pred_prob = self.generate_via_gpt3(self.args.engine, prompt, self.apikeys)
                    # print('pred_answer, pred_prob', pred_answer, pred_prob)

                pred_answer_list.append(pred_answer)
                pred_prob_list.append(pred_prob)
                # print(pred_answer_list)
                # exit()

            if len(pred_prob_list) == 0:
                pred_ans.append({'key': key, 'pred_answer': 'NA', 'prompt': prompt, 'question': question, 'acc': 0.});
                continue

            maxval = -999.
            for ii in range(len(pred_prob_list)):
                if pred_prob_list[ii] >= maxval:
                    maxval, pred_answer = pred_prob_list[ii], pred_answer_list[ii]
            ## a rough accuracy estimator for fast results check
            # print('answer', answer, pred_answer)
            counter = 0
            for ii in range(len(answer)):
                if pred_answer == answer[ii]: counter += 1

            # print('pred_answer_list', pred_answer_list, answer)
            pred_ans.append(
                {'key': key,
                 'pred_answer': pred_answer,
                 'prompt': prompt,
                 'question': question,
                 'prob': maxval,
                 'acc': min(1., float(counter) * 0.3)})
            pred_acc[question] = pred_ans[-1]['acc']
        # print('pred_ans', pred_ans); exit()
        return pred_ans, pred_acc

    def get_context_keys(self, key, metric, n):
        if len(self.valkey2idx) == 0:
            return None
        if metric in ['question', 'edit']:
            lineid = self.valkey2idx[key]
            # print(self.train_feature.shape, self.val_feature.shape)
            similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            index = similarity.argsort()[-n:][::-1]
            return [self.train_idx[str(x)] for x in index]
        elif metric == 'imagequestion':
            ## combined with Q-similairty (image+question)
            lineid = self.valkey2idx[key]
            question_similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            ## end of Q-similairty
            similarity = question_similarity + np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
            index = similarity.argsort()[-n:][::-1]
            return [self.train_idx[str(x)] for x in index]
        else:
            return None

    def load_similarity(self):
        self.valkey2idx = {}
        # if self.args.baseline == 'image2prompt' and self.args.n_shot == 0:
        if True: #re.search('(?<=nshot)0', self.args.edit_path):
            if self.args.dataset == 'okvqa':
                self.valkey2prompt = load_exemplar('./data/okvqa/okvqa_val_qusetions_qa_prompts.json' )
            elif self.args.dataset == 'aokvqa-val':
                self.valkey2prompt = load_exemplar('./data/aokvqa/aokvqa_val_qusetions_qa_prompts.json' )
            elif self.args.dataset == 'aokvqa-test':
                self.valkey2prompt = load_exemplar('./data/aokvqa/aokvqa_test_qusetions_qa_prompts.json' )
        else:
            if self.args.n_shot > 0:
                for ii in val_idx:
                    self.valkey2idx[val_idx[ii]] = int(ii)
                if self.args.similarity_metric=='question':
                    self.train_feature = np.load('%s/coco_clip_vitb16_train2014_okvqa_question.npy'%self.args.similarity_path)
                    self.val_feature = np.load('%s/coco_clip_vitb16_val2014_okvqa_question.npy'%self.args.similarity_path)
                    self.train_idx = json.load(open('%s/okvqa_qa_line2sample_idx_train2014.json'%self.args.similarity_path,'r'))
                elif self.args.similarity_metric=='imagequestion':
                    self.train_feature = np.load('%s/coco_clip_vitb16_train2014_okvqa_question.npy'%self.args.similarity_path)
                    self.val_feature = np.load('%s/coco_clip_vitb16_val2014_okvqa_question.npy'%self.args.similarity_path)
                    self.train_idx = json.load(open('%s/okvqa_qa_line2sample_idx_train2014.json'%self.args.similarity_path,'r'))
                    self.image_train_feature = np.load('%s/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy'%self.args.similarity_path)
                    self.image_val_feature = np.load('%s/coco_clip_vitb16_val2014_okvqa_convertedidx_image.npy'%self.args.similarity_path)
                    # print(self.val_feature.shape)
                    # print(len(self.image_val_feature), len(self.image_val_feature[0]));
                    # exit()
                elif self.args.similarity_metric=='edit':
                    self.train_idx = json.load(open('%s/okvqa_qa_line2sample_idx_train2014.json'%self.args.similarity_path,'r'))
                    self.train_feature = np.load('./edit_feature/okvqa_edit_train.npy')
                    self.val_feature = np.load('./edit_feature/okvqa_edit_val.npy' )
    

    def load_tags(self):
        tags_dict = {}
        tagging_pred_file = '%s/test.score.json.tsv' % self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]), json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        tagging_pred_file = '%s/val.score.json.tsv' % self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]), json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        tagging_pred_file = '%s/train.score.json.tsv' % self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]), json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        return tags_dict

    def load_cachetext(self):
        if self.args.dataset in ['aokvqa-val', 'okvqa']:
            valcaption_file = './data/caption/vinvl_caption/VinVL_base_val2014.tsv'
        else:
            valcaption_file = './data/caption/vinvl_caption/aokvqa_test_caption_vinl.tsv'

        read_tsv = csv.reader(open(valcaption_file, 'r'), delimiter="\t")
        caption_dict = {}
        if 'tag' in self.args.caption_type:
            tags_dict = self.load_tags()

        if self.args.caption_type=='vinvl_tag':
            for row in read_tsv:
                row_name = row[0][-6:]
                #print(row[0], row_name)
                if (int(row_name) not in caption_dict) and (int(row_name) in tags_dict):
                    caption_dict[int(row_name)] = [row[1].split('caption": "')[1].split('", "conf"')[0]+'. '+tags_dict[int(row_name)]]
                elif int(row_name) not in caption_dict:
                    caption_dict[int(row_name)] = [row[1].split('caption": "')[1].split('", "conf"')[0]]
                elif (int(row_name) in caption_dict) and (int(row_name) in tags_dict):
                    caption_dict[int(row_name)].append(row[1].split('caption": "')[1].split('", "conf"')[0]+'. '+tags_dict[int(row_name)])
                else:
                    caption_dict[int(row_name)].append(row[1].split('caption": "')[1].split('", "conf"')[0])
        return caption_dict

    def generate_via_gpt3(self, engine, prompt, apikeys):
        if False: #prompt in self.gpt3_cache:
            # print([prompt]); exit()
            # print(self.gpt3_cache[prompt])
            pred_answer = self.gpt3_cache[prompt]
            pred_prob = 1.0
            return pred_answer, pred_prob
        else:
            print('Request from LLM ...');

        # print(prompt); exit()
        response = None
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=5,
                logprobs=1,
                temperature=0.,
                stream=False,
                stop=["\n", "<|endoftext|>"]
            )
        except Exception as e:
            print('\nnow stop for 10s ...')
            time.sleep(60)
            print('continue ...')

            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=5,
                logprobs=1,
                temperature=0.,
                stream=False,
                stop=["\n", "<|endoftext|>"]
            )

            '''
            print(e)
            api_key_index = apikeys.index(openai.api_key)
            openai.api_key = apikeys[api_key_index+1]
            print('change api key to %s ...' %openai.api_key)
            exit()
            '''
        # print('response', response)

        plist = []
        for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
            if response['choices'][0]['logprobs']['tokens'][ii] == '\n':
                break
            plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])

        pred_answer = process_answer(response['choices'][0]["text"])
        pred_prob = sum(plist)
        # print(pred_answer, pred_prob)
        return pred_answer, pred_prob

    def generate_via_opt(self, llm_model, tokenizer, Img2Prompt):
        #print('Img2Prompt', Img2Prompt); exit()
        if self.args.llm in ['gpt-neo-2.7b', 'gpt-j-6B', 'bloom-7.1b']:
            prompt = tokenizer(Img2Prompt,
                               padding='longest',
                               truncation=True,
                               return_tensors="pt")
        else:
            prompt = tokenizer(Img2Prompt,
                               padding='longest',
                               truncation=True,
                               return_tensors="pt")

        if self.device:
            prompt.input_ids = prompt.input_ids.to(self.device)
            prompt.attention_mask = prompt.attention_mask.to(self.device)
        #print('prompt', prompt.input_ids); exit()
        if self.args.llm in ['gpt-neo-2.7b', 'gpt-j-6B', 'bloom-7.1b']:
            outputs = llm_model.generate(input_ids=prompt.input_ids,
                             attention_mask=prompt.attention_mask,
                             max_length=20+len(prompt.input_ids[0]),
                             return_dict_in_generate=True,
                             pad_token_id=tokenizer.eos_token_id,
                             output_scores=True
                             )
        else:
            outputs = llm_model.generate(input_ids=prompt.input_ids,
                             attention_mask=prompt.attention_mask,
                             max_length=20+len(prompt.input_ids[0]),
                             return_dict_in_generate=True,
                             output_scores=True
                             )

        if self.device:
            outputs.sequences = outputs.sequences.cpu()
        # print('__dict__', outputs.__dict__.keys())
        # print(outputs.scores[0].size())
        pred_answer = tokenizer.batch_decode(outputs.sequences[:, len(prompt.input_ids[0]):])[0]  #
        # print('pred_answer', pred_answer)
        pred_answer = process_answer(process_opt_answer(pred_answer))
        # print('after pred_answer', pred_answer); #exit()
        pred_prob = 1.0
        # print('final_answer', pred_answer); exit()
        return pred_answer, pred_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, \
                        default='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',
                        help='api key; https://openai.com/api/')  
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--edit_path', type=str, default='okvqa_merge_0-2000_opt-6.7b_nshot4_editnum3_pica_')
    parser.add_argument('--caption_type', type=str, default='vinvl_tag', help='vinvl_tag, vinvl')
    parser.add_argument('--n_shot', type=int, default=0, help="number of shots")
    parser.add_argument('--n_ensemble', type=int, default=1, help="number of ensemble")
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--valcaption_file', type=str, default='input_text/vinvl_caption/VinVL_base_val2014.tsv')
    parser.add_argument('--tag_path', type=str, default='./data/caption/coco_caption_pred_tags')
    parser.add_argument('--coco_path', type=str, default='./data/caption/coco_annotations')
    parser.add_argument('--similarity_path', type=str, default='./coco_clip_new')
    parser.add_argument('--dataset', type=str, default='aokvqa-val', help = 'okvqa, aokvqa-val, aokvqa-test')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--mode', type=str, default='merge')
    parser.add_argument('--sample', type=str, default='0-100000')
    parser.add_argument('--llm', type=str, default='gpt3', help='gpt3, opt-2.7b, opt-30b, \
                            gpt-neo-2.7b, bloom-7.1b, gpt-j-6B, gpt3.5')
    parser.add_argument('--save_cache', type=int, default=5)
    parser.add_argument('--edit_num', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--additional_information', type=str, default='')
    args = parser.parse_args()
    print('args.edit_path', args.edit_path)

    openai.api_key = args.apikey

    okvqa = Answer_Choosing(args)

    ## main inference
    answers = okvqa.inference()

    prediction = {}
    acc = 0.
    for answer in answers:
        # print('answer', answer)
        answer = answer[0]
        prediction[answer['key']] = [answer['pred_answer'], answer['prompt']]
        acc += float(answer['acc'])

    format_prediction = []
    for answer in answers:
        # print(answer)
        answer = answer[0]
        format_prediction.append({"answer": answer['acc'], "question_id": answer['key'].split('<->')[1]})

    print(acc * 100. / len(answers), len(answers))
    acc = acc * 100. / len(answers)

    ## if save final predictions
    os.system("mkdir -p %s" % args.output_path)
    os.system("mkdir -p %s/prompt_answer" % args.output_path)
    os.system("mkdir -p %s/format_answer" % args.output_path)
    output_name = 'PICa_%s_n%d_repeat%d_%s_%f.json' % (
    args.caption_type, args.n_shot, args.n_ensemble, args.similarity_metric, acc)
    json.dump(prediction, open("%s/prompt_answer/%s" % (args.output_path, output_name), 'w'))
    json.dump(format_prediction, open("%s/format_answer/%s" % (args.output_path, output_name), 'w'))


if __name__ == '__main__':
    main()


'''
python code/qrp_prompting_step2.py \
--edit_path aokvqa-val_merge_0-10000_gpt3_nshot0_editnum5_image2prompt \
--additional_information prob
'''