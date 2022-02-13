# Adapted from SQLova script for interaction.
# @author: Ziyu Yao
# Oct 7th, 2020
#
import os, sys, argparse, re, json, pickle, math

from copy import deepcopy
from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets
import numpy as np
import time, datetime, pytimeparse

import SQLova_model.bert.tokenization as tokenization
from SQLova_model.bert.modeling import BertConfig, BertModel
from SQLova_model.sqlova.utils.utils_wikisql import *
from SQLova_model.sqlova.model.nl2sql.wikisql_models import *
from SQLova_model.sqlnet.dbengine import DBEngine
from SQLova_model.agent import Agent
from SQLova_model.world_model import WorldModel
from SQLova_model.error_detector import *
from MISP_SQL.question_gen import QuestionGenerator
from SQLova_model.environment import UserSim, RealUser, ErrorEvaluator, GoldUserSim
from user_study_utils import *
from MISP_SQL.utils import semantic_unit_segment

np.set_printoptions(precision=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EARLY_STOP_EPOCH_STAGE1=10
EARLY_STOP_EPOCH_STAGE2=5
EARLY_THRESHOLD=30000

class WebInteractiveParser:
    def __init__(self) -> None:
        pass
    
    def construct_hyper_param(self,parser):
        parser.add_argument("--bS", default=1, type=int, help="Batch size")
        parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                            help="Type of model.")
        parser.add_argument('--seed', type=int, default=0, help='Random seed.')
        parser.add_argument('--model_dir', type=str, required=False, help='Which folder to save the model checkpoints.')

        # 1.2 BERT Parameters
        parser.add_argument("--vocab_file",
                            default='vocab.txt', type=str,
                            help="The vocabulary file that the BERT model was trained on.")
        parser.add_argument("--max_seq_length",
                            default=222, type=int,  # Set based on maximum length of input tokens.
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--num_target_layers",
                            default=2, type=int,
                            help="The Number of final layers of BERT to be used in downstream task.")
        parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
        parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
        parser.add_argument("--bert_type_abb", default='uS', type=str,
                            help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

        # 1.3 Seq-to-SQL module parameters
        parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
        parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
        parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
        parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

        # 1.4 Execution-guided decoding beam-size. It is used only in test.py
        # parser.add_argument('--EG',
        #                     default=False,
        #                     action='store_true',
        #                     help="If present, Execution guided decoding is used in test.")
        # parser.add_argument('--beam_size', # used for non-interactive decoding only
        #                     type=int,
        #                     default=4,
        #                     help="The size of beam for smart decoding")

        # Job setting
        parser.add_argument('--job', default='test_w_interaction', choices=['test_w_interaction', 'online_learning'],
                            help='Set the job. For parser pretraining, see other scripts.')

        # Data setting
        parser.add_argument('--data', default='dev', choices=['dev', 'test', 'user_study', 'online'],
                            help='which dataset to test.')
        parser.add_argument('--data_seed', type=int, default=0, choices=[0, 10, 100],
                            help='Seed for simulated online data order.')

        # Model (initialization/testing) setting
        parser.add_argument('--setting', default='full_train',
                            choices=['full_train', 'online_pretrain_1p', 'online_pretrain_5p', 'online_pretrain_10p'],
                            help='Model setting; checkpoints will be loaded accordingly.')

        # for interaction
        parser.add_argument('--num_options', type=str, default='3',
                            help='[INTERACTION] Number of options (inf or an int number).')
        parser.add_argument('--user', type=str, default='sim', choices=['sim', 'gold_sim', 'real'],
                            help='[INTERACTION] The user setting.')
        parser.add_argument('--err_detector', type=str, default='any',
                            help='[INTERACTION] The error detector: '
                                '(1) prob=x for using policy probability threshold;'
                                '(2) stddev=x for using Bayesian dropout threshold (need to set --dropout and --passes);'
                                '(3) any for querying about every policy action;'
                                '(4) perfect for using a simulated perfect detector.')
        parser.add_argument('--friendly_agent', type=int, default=0, choices=[0, 1],
                            help='[INTERACTION] If 1, the agent will not trigger further interactions '
                                'if any wrong decision is not resolved during parsing.')
        parser.add_argument('--output_path', type=str, default='temp', help='[INTERACTION] Where to save outputs.')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='[INTERACTION] Dropout rate for Bayesian dropout-based uncertainty analysis.')
        parser.add_argument('--passes', type=int, default=1,
                            help='[INTERACTION] Number of decoding passes for Bayesian dropout-based uncertainty analysis.')
        parser.add_argument('--ask_structure', type=int, default=0, choices=[0, 1],
                            help='[INTERACTION] Set to True to allow questions about query structure (WHERE clause).')

        # for online learning
        parser.add_argument('--update_iter', default=1000, type=int,
                            help="[LEARNING] Number of iterations per update.")
        parser.add_argument('--supervision', default='misp_neil',
                            choices=['full_expert', 'misp_neil', 'misp_neil_pos', 'misp_neil_perfect',
                                    'bin_feedback', 'bin_feedback_expert',
                                    'self_train', 'self_train_0.5'],
                            help='[LEARNING] Online learning supervision based on different algorithms.')
        parser.add_argument('--start_iter', default=0, type=int, help='[LEARNING] Iteration to start.')
        parser.add_argument('--end_iter', default=-1, type=int, help='[LEARNING] Iteration to end.')
        parser.add_argument('--auto_iter', default=0, type=int, choices=[0, 1],
                            help='[LEARNING] If 1, unless args.start_iter > 0 is specified, the system will automatically '
                                'search for `start_iter` given the aggregated training data. '
                                'Only applies to args.supervision = misp_neil/bin_feedback(_expert).')

        args = parser.parse_args()

        map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                            'uL': 'uncased_L-24_H-1024_A-16',
                            'cS': 'cased_L-12_H-768_A-12',
                            'cL': 'cased_L-24_H-1024_A-16',
                            'mcS': 'multi_cased_L-12_H-768_A-12'}
        args.bert_type = map_bert_type_abb[args.bert_type_abb]
        print(f"BERT-type: {args.bert_type}")

        # Decide whether to use lower_case.
        if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
            args.do_lower_case = False
        else:
            args.do_lower_case = True

        # Seeds for random number generation
        if args.data == "online":
            print("## online data seed: %d" % args.data_seed)
        print("## random seed: %d" % args.seed)
        python_random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # args.toy_model = not torch.cuda.is_available()
        args.toy_model = False
        args.toy_size = 12

        print("Testing data: {}".format(args.data))
        return args

    def get_bert(self,BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
        bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
        vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
        init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')

        bert_config = BertConfig.from_json_file(bert_config_file)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        bert_config.print_status()

        model_bert = BertModel(bert_config)
        if no_pretraining:
            pass
        else:
            model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
            print("Load pre-trained parameters.")
        model_bert.to(device)

        return model_bert, tokenizer, bert_config

    def get_models(self,args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
        # some constants
        agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

        print(f"Batch_size = {args.bS}")
        # print(f"Batch_size = {args.bS * args.accumulate_gradients}")
        print(f"BERT parameters:")
        print(f"learning rate: {args.lr_bert}")
        # print(f"Fine-tune BERT: {args.fine_tune}")

        # Get BERT
        model_bert, tokenizer, bert_config = self.get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                    args.no_pretraining)
        args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

        # Get Seq-to-SQL

        n_cond_ops = len(cond_ops)
        n_agg_ops = len(agg_ops)
        print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
        print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
        print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
        print(f"Seq-to-SQL: dropout rate = {args.dr}")
        print(f"Seq-to-SQL: learning rate = {args.lr}")
        model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
        model = model.to(device)

        if trained:
            assert path_model_bert != None
            assert path_model != None

            if torch.cuda.is_available():
                res = torch.load(path_model_bert)
            else:
                res = torch.load(path_model_bert, map_location='cpu')
            model_bert.load_state_dict(res['model_bert'])
            model_bert.to(device)

            if torch.cuda.is_available():
                res = torch.load(path_model)
            else:
                res = torch.load(path_model, map_location='cpu')

            model.load_state_dict(res['model'])

        return model, model_bert, tokenizer, bert_config

    def get_opt(self,model, model_bert, fine_tune):
        if fine_tune:
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=0)

            opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                        lr=args.lr_bert, weight_decay=0)
        else:
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=0)
            opt_bert = None

        return opt, opt_bert

    def report_detail(self,hds, nlu,
                    g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                    pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                    cnt_list, current_cnt):
        cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

        print(f'cnt = {cnt} / {cnt_tot} ===============================')

        print(f'headers: {hds}')
        print(f'nlu: {nlu}')

        # print(f's_sc: {s_sc[0]}')
        # print(f's_sa: {s_sa[0]}')
        # print(f's_wn: {s_wn[0]}')
        # print(f's_wc: {s_wc[0]}')
        # print(f's_wo: {s_wo[0]}')
        # print(f's_wv: {s_wv[0][0]}')
        print(f'===============================')
        print(f'g_sc : {g_sc}')
        print(f'pr_sc: {pr_sc}')
        print(f'g_sa : {g_sa}')
        print(f'pr_sa: {pr_sa}')
        print(f'g_wn : {g_wn}')
        print(f'pr_wn: {pr_wn}')
        print(f'g_wc : {g_wc}')
        print(f'pr_wc: {pr_wc}')
        print(f'g_wo : {g_wo}')
        print(f'pr_wo: {pr_wo}')
        print(f'g_wv : {g_wv}')
        # print(f'pr_wvi: {pr_wvi}')
        print('g_wv_str:', g_wv_str)
        print('p_wv_str:', pr_wv_str)
        print(f'g_sql_q:  {g_sql_q}')
        print(f'pr_sql_q: {pr_sql_q}')
        print(f'g_ans: {g_ans}')
        print(f'pr_ans: {pr_ans}')
        print(f'--------------------------------')

        print(cnt_list)

        print(f'acc_lx = {cnt_lx/cnt:.3f}, acc_x = {cnt_x/cnt:.3f}\n',
            f'acc_sc = {cnt_sc/cnt:.3f}, acc_sa = {cnt_sa/cnt:.3f}, acc_wn = {cnt_wn/cnt:.3f}\n',
            f'acc_wc = {cnt_wc/cnt:.3f}, acc_wo = {cnt_wo/cnt:.3f}, acc_wv = {cnt_wv/cnt:.3f}')
        print(f'===============================')

    def print_result(self,epoch, acc, dname):
        ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

        print(f'{dname} results ------------')
        print(
            f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
            acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
        )

    def web_real_user_interaction(self, data_loader, data_table, user, agent, tokenizer,
                            max_seq_length, num_target_layers, path_db, save_path):
        dset_name = "test"

        if os.path.isfile(save_path):
            saved_results = json.load(open(save_path, "r"))
            interaction_records = saved_results['interaction_records']
            count_exit = saved_results['count_exit']
            time_spent = datetime.timedelta(seconds=pytimeparse.parse(saved_results['time_spent']))
            st_pos = saved_results['st']
            current_cnt = eval(saved_results['current_cnt'])
            [cnt_tot, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x] = current_cnt
        else:
            cnt_sc = 0
            cnt_sa = 0
            cnt_wn = 0
            cnt_wc = 0
            cnt_wo = 0
            cnt_wv = 0
            cnt_wvi = 0
            cnt_lx = 0
            cnt_x = 0

            interaction_records = {}
            count_exit = 0
            time_spent = datetime.timedelta()
            st_pos = 0
            cnt_tot = 1

        cnt = 0
        engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
        iB=0
        t = next(iter(data_loader))
        msgs = {}
        assert len(t) == 1
        if cnt < st_pos:
            cnt += 1
            return
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        g_sql_q = generate_sql_q(sql_i, tb)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(agent.world_model.bert_config, agent.world_model.model_bert, tokenizer,
                            nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        os.system('clear')  # clear screen
        print_header(len(data_loader.dataset) - cnt)  # interface header
        print(bcolors.BOLD + "Suppose you are given a table with the following " +
            bcolors.BLUE + "header" + bcolors.ENDC +
            bcolors.BOLD + ":" + bcolors.ENDC)
        user.show_table(t[0]['table_id'])  # print table
        table_id = t[0]['table_id']
        table = user.get_table_details(t[0]['table_id'])
        table['rows'] = table['rows'][: min(3, len(table['rows']))]
        msgs["first_prompt"] = {'table_id': table_id, 'table': table}
        msgs['first_prompt']["question"] = ["And you want to answer the following question based on this table",
        t[0]['question'], "To help you get the answer automatically, the system has the following yes/no questions for you.",
        "(When no question prompts, please continue to the next case)", "Ready?"]

        start_time = datetime.datetime.now()
        # init decode
        if isinstance(agent.error_detector, ErrorDetectorBayesDropout):
            input_item = [tb, nlu_t, nlu, hds]
        else:
            input_item = [wemb_n, l_n, wemb_h, l_hpu, l_hs, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu]
        self.hyp = agent.world_model.decode(input_item, dec_beam_size=1, bool_verbal=False)[0]
        self.user = user
        self.input_item = input_item

        self.g_sql = sql_i[0]
        self.g_sql["g_wvi"] = g_wvi[0]
        self.agent = agent
        assert self.user.user_type == "real"

        # setup
        self.user.update_truth(self.g_sql)
        self.user.update_pred(self.hyp.tag_seq, self.hyp.dec_seq)
        self.user.clear_counter()
        self.user.undo_semantic_units = []
        self.agent.world_model.clear()

        # state tracker
        self.tracker = [] # a list of (hypothesis, starting position in tag_seq)

        # error detection
        self.start_pos = 0
        self.err_su_pointer_pairs = self.agent.error_detector.detection(self.hyp.tag_seq, start_pos=self.start_pos, bool_return_first=True)

        return msgs
    
    def undo_execution(self, questioned_su, avoid_items, confirmed_items):
            assert len(tracker) >= 1, "Invalid undo!"
            hyp, start_pos = tracker.pop()

            # reset user states
            user.update_pred(hyp.tag_seq, hyp.dec_seq)

            # clear feedback after start_pos
            _tag_item_lists, _seg_pointers = semantic_unit_segment(hyp.tag_seq)
            clear_start_pointer = 0
            for clear_start_pointer in _seg_pointers:
                if clear_start_pointer >= start_pos:
                    break
            clear_start_dec_idx = _tag_item_lists[_seg_pointers.index(clear_start_pointer)][-1]
            poped_keys = [k for k in avoid_items.keys() if k >= clear_start_dec_idx]
            for k in poped_keys:
                avoid_items.pop(k)
            poped_keys = [k for k in confirmed_items.keys() if k >= clear_start_dec_idx]
            for k in poped_keys:
                confirmed_items.pop(k)

            # clear the last user feedback records
            last_record = user.feedback_records[-1]
            if last_record == (questioned_su, 'undo'):
                _ = user.feedback_records.pop()
                rm_su = user.feedback_records.pop()[0]
                rm_dec_idx = rm_su[-1]
            else:
                rm_su = user.feedback_records.pop()[0]
                rm_dec_idx = rm_su[-1]
                assert rm_dec_idx == questioned_su[-1]

            rm_start_idx = len(user.feedback_records) - 1
            while rm_start_idx >= 0 and user.feedback_records[rm_start_idx][0][-1] == rm_dec_idx:
                rm_start_idx -= 1
            user.feedback_records = user.feedback_records[:rm_start_idx + 1]

            return hyp, start_pos, avoid_items, confirmed_items
    
    def verified_qa(self):
        self.quesType = 'qa'
        if len(self.err_su_pointer_pairs)>0:
            print("inside verified qs len of err_su_pointer_pairs".format(len(self.err_su_pointer_pairs)))
            self.su, self.pointer = self.err_su_pointer_pairs[0]
            semantic_tag = self.su[0]
            print("\nSemantic Unit: {}".format(self.su))

            # question generation
            self.question, self.cheat_sheet = self.agent.q_gen.question_generation(self.su, self.hyp.tag_seq, self.pointer)
            self.user.questioned_pointers.append(self.pointer)
            return {"response":["Semantic Unit: {}".format(self.su), self.question, "Please enter yes(y)/no(n)/undo/exit: "]}
        else:
            return {"response":["The system has finished SQL synthesis. This is the predicted SQL: {}".format(self.hyp.sql)]}
   
    def use_feedback(self, user_feedback):
        self.quesType = 'use_fbk'
        self.user.record_user_feedback(self.hyp.tag_seq[self.pointer], user_feedback, bool_qa=True)
        if user_feedback == "exit":
            return self.hyp, True

        if user_feedback == "undo":
            self.user.undo_semantic_units.append((self.su, "Step1"))
            self.hyp, self.start_pos, self.agent.world_model.avoid_items, self.agent.world_model.confirmed_items = self.undo_execution(
                self.su, self.world_model.avoid_items, self.world_model.confirmed_items)

            # error detection in the next turn
            self.err_su_pointer_pairs = self.error_detector.detection(
                self.hyp.tag_seq, start_pos=self.start_pos, bool_return_first=True)
            return

        self.tracker.append((self.hyp, self.start_pos))

        if self.cheat_sheet[user_feedback][0]: # user affirms the decision
            print("using positive feedback")
            self.agent.world_model.apply_pos_feedback(self.su, self.hyp.dec_seq, self.hyp.dec_seq[:self.su[-1]])
            self.start_pos = self.pointer + 1

        else: # user negates the decision
            if self.cheat_sheet[user_feedback][1] == 0:
                dec_seq_idx = self.su[-1]
                self.dec_prefix = self.hyp.dec_seq[:dec_seq_idx]

                # update negated items
                self.dec_prefix = self.agent.world_model.apply_neg_feedback(self.su, self.hyp.dec_seq, self.dec_prefix)

                # perform one-step beam search to generate options
                self.cand_hypotheses = self.agent.world_model.decode(
                    self.input_item, dec_beam_size=self.agent.world_model.num_options,
                    dec_prefix=self.dec_prefix,
                    avoid_items=self.agent.world_model.avoid_items,
                    confirmed_items=self.agent.world_model.confirmed_items,
                    stop_step=dec_seq_idx, bool_collect_choices=True,
                    bool_verbal=False)

                # prepare options
                self.cand_semantic_units = []
                for cand_hyp in self.cand_hypotheses:
                    cand_units, cand_pointers = semantic_unit_segment(cand_hyp.tag_seq)
                    assert cand_units[-1][0] == self.su[0]
                    self.cand_semantic_units.append(cand_units[-1])

                # present options
                self.opt_question, self.opt_answer_sheet, self.sel_none_of_above = self.agent.q_gen.option_generation(
                    self.cand_semantic_units, self.hyp.tag_seq, self.pointer)

                if self.user.bool_undo:
                    self.undo_opt = self.sel_none_of_above + (2 if self.agent.bool_structure_question else 1)
                    self.opt_question = self.opt_question[:-1] + ";\n" + \
                                    "(%d) I want to undo my last choice!" % self.undo_opt
                opts = self.opt_question.split("\n")
                opts.append("Please enter the option id(s) delimited by comma ', ': ")
                return {"response": opts}

            else: # type 1 unit: for decisions with only yes/no choices, we "flip" the current decision
                assert self.cheat_sheet[user_feedback][1] == 1
                dec_seq_idx = self.su[-1]

                self.dec_prefix = self.agent.world_model.apply_neg_feedback(
                    self.su, self.hyp.dec_seq, self.hyp.dec_seq[:dec_seq_idx])
                try:
                    self.hyp = self.agent.world_model.decode(self.input_item, dec_prefix=self.dec_prefix,
                                                    avoid_items=self.agent.world_model.avoid_items,
                                                    confirmed_items=self.agent.world_model.confirmed_items,
                                                    bool_verbal=False)[0]
                except:
                    pass
                self.user.update_pred(self.hyp.tag_seq, self.hyp.dec_seq)
                self.start_pos = self.pointer + 1
        
        self.err_su_pointer_pairs = self.agent.error_detector.detection(
        self.hyp.tag_seq, start_pos=self.start_pos, bool_return_first=True)
        return self.verified_qa()



    def user_selection(self, user_selections):
        # user selection
        def answer_parsing(answer_str):
            selections = answer_str.split(", ")
            try:
                selections = [int(sel) for sel in selections]
            except:
                return None
            else:
                assert len(selections)
                if self.sel_none_of_above in selections:
                    assert len(selections) == 1 # mutual exclusive "none of the above"
                return selections
        user_selections = answer_parsing(user_selections)
        print("inside user selection", user_selections)
                
        self.user.option_selections.append((self.su[0], self.opt_question, user_selections))
        

        if self.user.bool_undo and user_selections == [self.undo_opt]:
            self.user.undo_semantic_units.append((self.su, "Step2"))
            self.hyp, self.start_pos, self.agent.world_model.avoid_items, self.agent.world_model.confirmed_items = self.undo_execution(
                self.su, self.agent.world_model.avoid_items, self.agent.world_model.confirmed_items)

            # error detection in the next turn
            self.err_su_pointer_pairs = self.agent.error_detector.detection(
                self.hyp.tag_seq, start_pos=self.start_pos, bool_return_first=True)
            return

        for idx in range(len(self.opt_answer_sheet)): # user selection feedback incorporation
            if idx + 1 in user_selections:
                # update dec_prefix for components whose only choice is selected
                self.dec_prefix = self.agent.world_model.apply_pos_feedback(
                    self.cand_semantic_units[idx], self.cand_hypotheses[idx].dec_seq, self.dec_prefix)
            else:
                self.dec_prefix = self.agent.world_model.apply_neg_feedback(
                    self.cand_semantic_units[idx], self.cand_hypotheses[idx].dec_seq, self.dec_prefix)

        # refresh decoding
        self.start_pos, self.hyp = self.agent.world_model.refresh_decoding(
            self.input_item, self.dec_prefix, self.hyp, self.su, self.pointer,
            self.sel_none_of_above, user_selections,
            bool_verbal=False)
        self.user.update_pred(self.hyp.tag_seq, self.hyp.dec_seq)

        # a friendly agent will not ask for further feedback if any wrong decision is not resolved.
        if self.agent.bool_mistake_exit and (self.sel_none_of_above in user_selections or
                                        self.sel_none_of_above + 1 in user_selections):
            return self.hyp, False
        
        self.err_su_pointer_pairs = self.agent.error_detector.detection(
                self.hyp.tag_seq, start_pos=self.start_pos, bool_return_first=True)
        return self.verified_qa()
                
        

    def setup(self):
        print("setting up model")
        SETTING="online_pretrain_1p" # online_pretrain_Xp, full_train
        NUM_OP="3"
        ED="prob=0.95"
        DATA="dev" # dev or test set
        BATCH_SIZE=16
        # path setting
        LOG_DIR="SQLova_model/logs" # save training logs
        MODEL_DIR="SQLova_model/checkpoints_{}/".format(SETTING) # model dir
        OUTPUT_PATH= "{}/records_{}.json".format(LOG_DIR, "web_interaction")  

        # 1. Hyper parameters

        parser = argparse.ArgumentParser()
        print("parser", parser)
        args = self.construct_hyper_param(parser)
        print("arguments", args)
        args.model_dir = MODEL_DIR
        args.data = DATA
        args.num_options = NUM_OP
        args.err_detector = ED
        args.friendly_agent = 0
        args.user = "real"
        args.lr_bert = 5e-5
        args.setting = SETTING
        args.friendly_agent = 0
        args.user = 'real'
        args.bs = BATCH_SIZE
        args.ask_structure = 0
        args.output_path = OUTPUT_PATH 


        ## 2. Paths
        path_wikisql = 'SQLova_model/download/data/'
        BERT_PT_PATH = 'SQLova_model/download/bert/'
        model_dir = args.model_dir
        
        print("## job: {}".format(args.job))
        print("## setting: {}".format(args.setting))
        print("## model_dir: {}".format(args.model_dir))
        if args.auto_iter:
            print("## auto_iter is on.")
            print("\targs.start_iter=%d, args.end_iter=%d." % (args.start_iter, args.end_iter))

        path_model_bert = os.path.join(model_dir, "model_bert_best.pt")
        path_model = os.path.join(model_dir, "model_best.pt")

        ## 3. Load data
        if args.job == 'online_learning':
            dev_data, dev_table = load_processed_wikisql_data(path_wikisql, 'dev')
            test_data, test_table = load_processed_wikisql_data(path_wikisql, 'test')
            test_data = [item for item in test_data if item is not None]
        else:
            if args.data == "user_study":
                test_data, test_table = load_wikisql_data(path_wikisql, mode="test", toy_model=args.toy_model,
                                                        toy_size=args.toy_size, no_hs_tok=True)
                sampled_ids = json.load(open("SQLova_model/download/data/user_study_ids.json", "r"))
                test_data = [test_data[idx] for idx in sampled_ids]
            else:
                # args.data in ["dev", "test"]
                test_data, test_table = load_wikisql_data(
                    path_wikisql, mode=args.data,
                    toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)

        # 4. Build & Load models
        model, model_bert, tokenizer, bert_config = self.get_models(args, BERT_PT_PATH, trained=True,
                                                            path_model_bert=path_model_bert,
                                                            path_model=path_model)
        model.eval()
        model_bert.eval()

        ## 5. Create ISQL agent
        print("Creating MISP agent...")
        question_generator = QuestionGenerator()
        error_evaluator = ErrorEvaluator()
        print("## user: {}".format(args.user))
        if args.user == "real":
            user = RealUser(error_evaluator, test_table)
        elif args.user == "gold_sim":
            user = GoldUserSim(error_evaluator, bool_structure_question=args.ask_structure)
        else:
            assert not args.ask_structure, "UserSim with ask_struct=1 is not supported!"
            user = UserSim(error_evaluator)

        if args.err_detector == 'any':
            error_detector = ErrorDetectorProbability(1.1)  # ask any SU
        elif args.err_detector.startswith('prob='):
            prob = float(args.err_detector[5:])
            error_detector = ErrorDetectorProbability(prob)
            print("Error Detector: probability threshold = %.3f" % prob)
            assert args.passes == 1, "Error: For prob-based evaluation, set --passes 1."
        elif args.err_detector.startswith('stddev='):
            stddev = float(args.err_detector[7:])
            error_detector = ErrorDetectorBayesDropout(stddev)
            print("Error Detector: Bayesian Dropout Stddev threshold = %.3f" % stddev)
            print("num passes: %d, dropout rate: %.3f" % (args.passes, args.dropout))
            assert args.passes > 1, "Error: For dropout-based evaluation, set --passes 10."
        elif args.err_detector == "perfect":
            error_detector = ErrorDetectorSim()
            print("Error Detector: using a simulated perfect detector.")
        else:
            raise Exception("Invalid error detector setup %s!" % args.err_detector)

        if args.num_options == 'inf':
            print("WARNING: Unlimited options!")
            num_options = np.inf
        else:
            num_options = int(args.num_options)
            print("num_options: {}".format(num_options))

        print("ask_structure: {}".format(args.ask_structure))
        world_model = WorldModel((bert_config, model_bert, tokenizer, args.max_seq_length, args.num_target_layers),
                                model, num_options, num_passes=args.passes, dropout_rate=args.dropout,
                                bool_structure_question=args.ask_structure)

        print("friendly_agent: {}".format(args.friendly_agent))
        agent = Agent(world_model, error_detector, question_generator, bool_mistake_exit=args.friendly_agent,
                    bool_structure_question=args.ask_structure)

        ## 6. Test
        if not os.path.exists(os.path.dirname(args.output_path)):
            os.mkdir(os.path.dirname(args.output_path))

        if args.job == 'online_learning':
            assert args.data == "online"
            print("## supervision: {}".format(args.supervision))
            print("## update_iter: {}".format(args.update_iter))

            if args.setting == "online_pretrain_1p":
                online_setup_indices = json.load(open(path_wikisql + "online_setup_1p.json"))
            elif args.setting == "online_pretrain_5p":
                online_setup_indices = json.load(open(path_wikisql + "online_setup_5p.json"))
            elif args.setting == "online_pretrain_10p":
                online_setup_indices = json.load(open(path_wikisql + "online_setup_10p.json"))
            else:
                raise Exception("Invalid args.setting={}".format(args.setting))

            if args.supervision == 'full_expert':
                train_data, train_table = load_processed_wikisql_data(path_wikisql, "train")  # processed data
            else:
                train_data, train_table = load_wikisql_data(path_wikisql, mode="train", toy_model=args.toy_model,
                                                            toy_size=args.toy_size, no_hs_tok=True) # raw data

            init_train_indices = set(online_setup_indices["train"])
            init_train_data = [train_data[idx] for idx in init_train_indices if train_data[idx] is not None]
            print("## Update init train size %d " % len(init_train_data))

            online_train_indices = online_setup_indices["online_seed%d" % args.data_seed]
            online_train_data = [train_data[idx] for idx in online_train_indices if train_data[idx] is not None]

            print("## Update online train size %d " % len(online_train_data))
            online_data_loader = torch.utils.data.DataLoader(
                batch_size=1, # must be 1
                dataset=online_train_data,
                shuffle=False,
                num_workers=1, # 4
                collate_fn=lambda x: x  # now dictionary values are not merged!
            )

            def create_new_model(model, model_bert):
                # parser
                def param_reset(m):
                    if type(m) in {nn.LSTM, nn.Linear}:
                        m.reset_parameters()
                model.apply(param_reset)
                model.eval()

                # bert
                init_checkpoint = os.path.join(BERT_PT_PATH, 'pytorch_model_{}.bin'.format(args.bert_type))
                model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
                print("Reload pre-trained BERT parameters.")
                model_bert.to(device)
                model_bert.eval()

            if args.supervision in ("misp_neil", "misp_neil_pos"):
                subdir = "%s_OP%s_ED%s_SETTING%s_ITER%d_DATASEED%d%s%s/" % (
                    args.supervision, args.num_options, args.err_detector, args.setting,
                    args.update_iter, args.data_seed,
                    ("_FRIENDLY" if args.friendly_agent else ""),
                    ("_GoldUser" if args.user == "gold_sim" else ""))
                if not os.path.isdir(os.path.join(model_dir, subdir)):
                    os.mkdir(os.path.join(model_dir, subdir))

                if args.auto_iter and args.start_iter == 0 and os.path.exists(args.output_path):
                    record_save_path = args.output_path
                    print("Loading interaction records from %s..." % record_save_path)
                    interaction_records_dict = json.load(open(record_save_path, 'r'))
                    args.start_iter = interaction_records_dict['start_iter']
                    print("AUTO start_iter = %d." % args.start_iter)

                if args.start_iter > 0:
                    print("Loading previous checkpoints at iter {}...".format(args.start_iter))
                    start_path_model = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_best.pt')
                    start_path_model_bert = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_bert_best.pt')
                    if torch.cuda.is_available():
                        res = torch.load(start_path_model_bert)
                    else:
                        res = torch.load(start_path_model_bert, map_location='cpu')
                    agent.world_model.model_bert.load_state_dict(res['model_bert'])
                    agent.world_model.model_bert.to(device)

                    if torch.cuda.is_available():
                        res = torch.load(start_path_model)
                    else:
                        res = torch.load(start_path_model, map_location='cpu')
                    agent.world_model.semparser.load_state_dict(res['model'])

                online_learning(args.supervision, user, agent, init_train_data, online_data_loader,
                                train_table, dev_data, dev_table, test_data, test_table, args.update_iter,
                                os.path.join(model_dir, subdir), args.output_path, create_new_model,
                                max_seq_length=222, num_target_layers=2, detail=False,
                                st_pos=args.start_iter, end_pos=args.end_iter,
                                cnt_tot=1, path_db=path_wikisql, batch_size=args.bS)

            elif args.supervision.startswith('self_train'):
                subdir = "%s_SETTING%s_ITER%d_DATASEED%d/" % (
                    args.supervision, args.setting, args.update_iter,
                    args.data_seed)
                if not os.path.isdir(os.path.join(model_dir, subdir)):
                    os.mkdir(os.path.join(model_dir, subdir))

                if args.auto_iter and args.start_iter == 0 and os.path.exists(args.output_path):
                    record_save_path = args.output_path
                    print("Loading interaction records from %s..." % record_save_path)
                    interaction_records_dict = json.load(open(record_save_path, 'r'))
                    args.start_iter = interaction_records_dict['start_iter']
                    print("AUTO start_iter = %d." % args.start_iter)

                if args.start_iter > 0:
                    print("Loading previous checkpoints at iter {}...".format(args.start_iter))
                    start_path_model = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_best.pt')
                    start_path_model_bert = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_bert_best.pt')
                    if torch.cuda.is_available():
                        res = torch.load(start_path_model_bert)
                    else:
                        res = torch.load(start_path_model_bert, map_location='cpu')
                    agent.world_model.model_bert.load_state_dict(res['model_bert'])
                    agent.world_model.model_bert.to(device)

                    if torch.cuda.is_available():
                        res = torch.load(start_path_model)
                    else:
                        res = torch.load(start_path_model, map_location='cpu')
                    agent.world_model.semparser.load_state_dict(res['model'])

                online_learning_self_train(args.supervision, agent, init_train_data, online_data_loader, train_table,
                                        dev_data, dev_table, test_data, test_table, args.update_iter,
                                        os.path.join(model_dir, subdir), args.output_path,
                                        create_new_model, max_seq_length=222, num_target_layers=2, detail=False,
                                        st_pos=args.start_iter, end_pos=args.end_iter,
                                        cnt_tot=1, path_db=path_wikisql, batch_size=args.bS)

            elif args.supervision == "full_expert":
                subdir = "full_expert_SETTING%s_ITER%d_DATASEED%d/" % (
                    args.setting, args.update_iter, args.data_seed)
                if not os.path.isdir(os.path.join(model_dir, subdir)):
                    os.mkdir(os.path.join(model_dir, subdir))

                assert not args.auto_iter, "--auto_iter is not allowed for Full Expert experiments!"

                online_learning_full_expert(agent, init_train_data, online_train_data, train_table,
                                            dev_data, dev_table, test_data, test_table,
                                            path_wikisql, os.path.join(model_dir, subdir), args.update_iter,
                                            create_new_model, start_idx=args.start_iter, end_idx=args.end_iter,
                                            batch_size=args.bS)

            elif args.supervision in {"bin_feedback", "bin_feedback_expert"}:
                subdir = "%s_SETTING%s_ITER%d_DATASEED%d/" % (
                    args.supervision, args.setting, args.update_iter, args.data_seed)
                if not os.path.isdir(os.path.join(model_dir, subdir)):
                    os.mkdir(os.path.join(model_dir, subdir))

                if args.auto_iter and args.start_iter == 0 and os.path.exists(args.output_path):
                    record_save_path = args.output_path
                    print("Loading interaction records from %s..." % record_save_path)
                    interaction_records_dict = json.load(open(record_save_path, 'r'))
                    args.start_iter = interaction_records_dict['start_iter']
                    print("AUTO start_iter = %d." % args.start_iter)

                if args.start_iter > 0:
                    print("Loading previous checkpoints at iter {}...".format(args.start_iter))
                    start_path_model = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_best.pt')
                    start_path_model_bert = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_bert_best.pt')
                    if torch.cuda.is_available():
                        res = torch.load(start_path_model_bert)
                    else:
                        res = torch.load(start_path_model_bert, map_location='cpu')
                    agent.world_model.model_bert.load_state_dict(res['model_bert'])
                    agent.world_model.model_bert.to(device)

                    if torch.cuda.is_available():
                        res = torch.load(start_path_model)
                    else:
                        res = torch.load(start_path_model, map_location='cpu')
                    agent.world_model.semparser.load_state_dict(res['model'])

                online_learning_bin_feedback(args.supervision, agent, init_train_data, online_data_loader, train_table,
                                            dev_data, dev_table, test_data, test_table,
                                            os.path.join(model_dir, subdir), args.output_path,
                                            path_wikisql, args.update_iter, create_new_model,
                                            start_idx=args.start_iter, end_idx=args.end_iter, batch_size=args.bS)

            else:
                assert args.supervision == "misp_neil_perfect"
                subdir = "full_expert_SETTING%s_ITER%d_DATASEED%d/" % (
                    args.setting, args.update_iter, args.data_seed)

                if args.start_iter > 0:
                    print("Loading previous checkpoints at iter {}...".format(args.start_iter))
                    start_path_model = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_best.pt')
                    start_path_model_bert = os.path.join(model_dir, subdir, '%d' % args.start_iter, 'model_bert_best.pt')
                    if torch.cuda.is_available():
                        res = torch.load(start_path_model_bert)
                    else:
                        res = torch.load(start_path_model_bert, map_location='cpu')
                    agent.world_model.model_bert.load_state_dict(res['model_bert'])
                    agent.world_model.model_bert.to(device)

                    if torch.cuda.is_available():
                        res = torch.load(start_path_model)
                    else:
                        res = torch.load(start_path_model, map_location='cpu')
                    agent.world_model.semparser.load_state_dict(res['model'])

                online_learning_misp_perfect(user, agent, online_data_loader, train_table,
                                            args.update_iter, os.path.join(model_dir, subdir),
                                            args.output_path, st_pos=args.start_iter, end_pos=args.end_iter)

        else:
            # test_w_interaction
            test_loader = torch.utils.data.DataLoader(
                batch_size=1,  # must be 1
                dataset=test_data,
                shuffle=False,
                num_workers=1,  # 4
                collate_fn=lambda x: x  # now dictionary values are not merged!
            )

            if args.user == "real":
                with torch.no_grad():
                    ##make available to class 
                    return self.web_real_user_interaction(test_loader, test_table, user, agent, tokenizer, args.max_seq_length,
                                        args.num_target_layers, path_wikisql, args.output_path)

            else:
                with torch.no_grad():
                    acc_test, results_test, cnt_list, interaction_records = interaction(
                        test_loader, test_table, user, agent, tokenizer, args.max_seq_length, args.num_target_layers,
                        detail=True, path_db=path_wikisql, st_pos=0,
                        dset_name="test" if args.data == "user_study" else args.data)
                print(acc_test)

                # save results for the official evaluation
                path_save_for_evaluation = os.path.dirname(args.output_path)
                save_for_evaluation(path_save_for_evaluation, results_test, args.output_path[args.output_path.index('records_'):])
                json.dump(interaction_records, open(args.output_path, "w"), indent=4)

