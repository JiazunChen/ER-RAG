
import torch

from transformers import  LlamaForCausalLM


import models.Retriever as Retriever
from transformers import  GenerationConfig
from models.Parse import parse_answer, finance_parse_answer, music_parse_answer, sports_parse_answer, open_parse_answer
from models.prompt_api import template_map

import time
from peft import PeftModel
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAGModel:
    def __init__(self):

        self.Task = 3
         

        print("-------------------------Loading LLM--------------------------")

        t1 = time.time()
        model = "models/Llama-3-8B-instruct"

        num_gpus = torch.cuda.device_count()

        if num_gpus <= 2:
            self.m = LlamaForCausalLM.from_pretrained(model, device_map="balanced",
                                                      max_memory={0: "44000MiB", 1: 0, "cpu": 0})
            self.used1 = "cuda:1"
            self.used2 = "cuda:1"
            self.used = 'cuda:1'
        else:
            self.m = LlamaForCausalLM.from_pretrained(model, device_map="balanced", )
            self.used1 = "cuda:1"
            self.used2 = "cuda:2"
            self.used = "cuda:1"

        self.m = PeftModel.from_pretrained(self.m, "models/pretrain_models/llama3-52-peft/checkpoint-480", adapter_name="480").eval()
        self.m.load_adapter("models/pretrain_models/tran_619_apioutput/checkpoint-310", adapter_name="api_answer")
        self.m.load_adapter("models/pretrain_models/train_618api_up/checkpoint-580", adapter_name="generate_api")

        self.m.load_adapter("models/pretrain_models/llama3-52-peft/checkpoint-500", adapter_name="old_generate_api")
        self.m.load_adapter("models/pretrain_models/llama3-52-peft/checkpoint-580", adapter_name="old_api_answer")
        self.m.set_adapter("480")

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        print("finish loading LLM", time.time() - t1)

        print("-------------------------Loading RET----------------------")

        t1 = time.time()
        
        self.k = 5
        self.r = Retriever.Retriever2(batch_size=64, device1=self.used1, device2=self.used2,
                                      hf_path="models/all-Mini-L6-v2", parent_chunk_size=2000, parent_chunk_overlap=400,
                                      child_chunk_size=200, child_chunk_overlap=50,
                                      )

        print("finish loading RET", time.time() - t1)

        print("-------------------------Loading LM---------------------")

        t1 = time.time()

        print('finish loading auxilary LM', time.time() - t1)

        self.r.clear()

    def llama3_domain(self, query):
        messages = [
            {"role": "system", "content": f"You are an assistant expert in movie, sports, finance and music fields."},
            {"role": "user",
             "content": "Please judge which category the query belongs to, without answering the query. you can only and must output one word in (movie, sports, finance, music) If the question doesn't belong to movie, sports,finance, music, please answer other. \n Query:" + query + '\n Category:'},
        ]
        domain, _, _ = self.llam3_output(messages, maxtoken=3, disable_adapter=True)
        for key in ['finance', 'music', 'sports', 'movie']:
            if key in domain:
                return key
        return 'open'

    def llam3_output(self, messages, maxtoken=75, disable_adapter=False):
        self.m.eval()
        if time.time() - self.all_st >= self.all_time:
            return "i don't know", 0, 0
        with torch.no_grad():
            t1 = time.time()
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.m.device)
            print('input_ids shape', input_ids.shape)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generation_config = GenerationConfig(
                max_new_tokens=maxtoken, do_sample=False,
                max_time=32 - (time.time() - self.t_s), eos_token_id=terminators)
            if disable_adapter:
                with self.m.disable_adapter():
                    outputs = self.m.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        eos_token_id=terminators,
                        return_dict_in_generate=True,
                        output_scores=False)
            else:
                outputs = self.m.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    eos_token_id=terminators,
                    return_dict_in_generate=True,
                    output_scores=False)

            output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).lower().split("assistant")[
                -1].strip()
            print("end gen:", time.time() - t1)
            print("output:")
            print(output)
        return output, 0, 0

    def get_batch_size(self) -> int:
        return 16

    def batch_generate_answer(self, batch):
        self.all_st = time.time()
        self.all_time = 16 * 29
        answer = []
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        for a, b, c in zip(queries, batch_search_results, query_times):
            if time.time() - self.all_st >= self.all_time:
                answer.append("i don't know")
            else:
                #try:
                answer.append(self.generate_answer(a, b, c))
                #except:
                    #answer.append("i don't know")
        return answer

    def process_api(self, domain, query, query_time):
        print('api prompt')
        t1 = time.time()
        if domain in ['finance']:
            from models.prompt_api import finance_prompt
            filled_template = finance_prompt.format(query_str=query, time_str=query_time)
        elif domain in ['movie']:
            from models.prompt_api import movie_prompt
            filled_template = movie_prompt.format(query_str=query)
        elif domain in ['music']:
            from models.prompt_api import music_prompt
            filled_template = music_prompt.format(query_str=query)
        elif domain in ['sports']:
            from models.prompt_api import sports_prompt
            filled_template = sports_prompt.format(query_str=query, query_time=query_time)
        elif domain in ['open']:
            from models.prompt_api import open_prompt
            filled_template = open_prompt.format(query_str=query)
        messages = [
            {"role": "system",
             "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 50 words or less. Now is {query_time}"},
            {"role": "user", "content": filled_template},
        ]
        if ("domain" in ["sports"]):
            self.m.set_adapter("generate_api")
        else:
            self.m.set_adapter("old_generate_api")
        output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=75)
        # self.m.set_adapter("480")
        print("edn api prompt", output, time.time() - t1)
        if domain in ['finance']:
            res, res_str = finance_parse_answer(output, query_time)
        elif domain in ['movie']:
            res, res_str = parse_answer(output)
        elif domain in ['music']:
            res_str = music_parse_answer(output)
        elif domain in ['sports']:
            res_str = sports_parse_answer(output, query_time)
            if len(res_str) == 1 and 'There is no match' in res_str[0]:
                if 'update' in query or 'at the moment' in query or 'week' in query or 'last' in query or 'yesterday' in query or 'previous' in query or 'late' in query or 'today' in query:
                    print('no record')
                    return 'invalid question'
        elif domain in ['open']:
            res_str = open_parse_answer(output)
        print("end parse_answer", res_str, time.time() - t1)
        if res_str != []:
            context_str = ""
            for snippet in res_str[:]:
                context_str += "<DOC>\n" + snippet + "\n</DOC>\n"
            context_str = self.tokenizer.encode(context_str, max_length=4000, add_special_tokens=False)
            print('len context_str', len(context_str))
            if len(context_str) >= 4000:
                context_str = self.tokenizer.decode(context_str) + "\n</DOC>\n"
            else:
                context_str = self.tokenizer.decode(context_str)
            if domain in ["sports"]:
                filled_template = template_map['template_output_answer'].format(context_str=context_str,
                                                                                query_str=query)
                messages = [
                    {"role": "system",
                     "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 70 words or less. Now is {query_time}"},
                    {"role": "user", "content": filled_template},
                ]
                self.m.set_adapter("api_answer")
            else:
                filled_template = template_map['output_answer_api'].format(context_str=context_str, query_str=query)
                messages = [
                    {"role": "system",
                     "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 50 words or less. If you are not sure about the query, answer i don't know. Now is {query_time}"},
                    {"role": "user", "content": filled_template},
                ]
                self.m.set_adapter("old_api_answer")
            output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=75)
            # self.m.set_adapter("480")
            print("edn api", time.time() - t1)
            if "i don't know" not in output:
                if 'invalid' in output:
                    output = "i don't know"
                return output
        return "i don't know"

    def process_task1(self, domain, query, query_time):
        context_str = ""
        output = ""
        if domain in ['movie']:
            context_str = self.r.get_movie_oscar(query)
            if context_str is not None:
                t1 = time.time()
                filled_template = template_map['output_answer_nofalse'].format(context_str=context_str,
                                                                               query_str=query)
                messages = [
                    {"role": "system",
                     "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"},
                    {"role": "user", "content": filled_template},
                ]
                output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=70, disable_adapter=True)
                print("end oscar", time.time() - t1)
                if "i don't know" not in output and "invalid" not in output:
                    return output, context_str
            else:
                context_str = ""
                t1 = time.time()
                filled_template = template_map['ask_name'].format(query_str=query)
                messages = [
                    {"role": "system",
                     "content": f" You will be asked a lot of questions, but you don't need to answer them, just point out the name of the movie involved."},
                    {"role": "user", "content": filled_template},
                ]
                output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=70, disable_adapter=True)
                print("end ask movie name", time.time() - t1)
                if "i don't know" not in output:
                    try:
                        for tmpoutput in output.split(' && '):
                            tmpoutput = tmpoutput.replace('"', '').strip()
                            context_str += self.r.get_movie_context(tmpoutput)
                        print(context_str)
                    except:
                        context_str = ""
                else:
                    context_str = ""
        elif domain in ['music']:
            context_str = self.r.get_music_grammy(query)
            print("get_music_grammy", context_str)
            if context_str is None:
                context_str = ""
            else:
                t1 = time.time()
                filled_template = template_map['output_answer_nofalse'].format(context_str=context_str,
                                                                               query_str=query)
                messages = [
                    {"role": "system",
                     "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"},
                    {"role": "user", "content": filled_template},
                ]
                output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=70, disable_adapter=True)
                print("edn music", output, time.time() - t1)
                if "i don't know" not in output and "invalid" not in output:
                    return output, context_str
                context_str = ""
        elif domain in ['finance']:
            if 'share' in query or 'pe' in query or 'eps' in query or 'ratio' in query or 'capitalization' in query or 'earnings' in query or 'market' in query:
                context_str = ""
                t1 = time.time()
                filled_template = template_map['ask_name_finance'].format(query_str=query)
                messages = [
                    {"role": "system",
                     "content": f" You will be asked a lot of questions, but you don't need to answer them, just point out the specific stock ticker or company name involved."},
                    {"role": "user", "content": filled_template},
                ]
                output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=70, disable_adapter=True)
                print("edn ask name", output, time.time() - t1)
                if "i don't know" not in output and 'none' not in output:
                    try:
                        for tmpoutput in output.split(' && '):
                            tmpoutput = tmpoutput.replace('"', '').strip()
                            context_str += self.r.get_finance_context(tmpoutput)
                        print(context_str)
                        t1 = time.time()
                        filled_template = template_map['output_answer_nofalse'].format(context_str=context_str,
                                                                                       query_str=query)
                        messages = [
                            {"role": "system",
                             "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"},
                            {"role": "user", "content": filled_template},
                        ]
                        output, minn_logit, mean_logit = self.llam3_output(messages, maxtoken=70,
                                                                           disable_adapter=True)
                        print("edn finance", time.time() - t1)
                        if "i don't know" not in output and "invalid" not in output:
                            return output, context_str
                        context_str = ""
                    except:
                        context_str = ""
                else:
                    context_str = ""
        return "", context_str

    def generate_answer(self, query, search_results, query_time=None) -> str:
        print("-------------Now Querying----------------")

        print(query)

        self.t_s = time.time()
        self.r.clear()

        ###Whether Compare/Multihop

        print("determine compare")

        t1 = time.time()

        domain = self.llama3_domain(query)  # self.determine_domain(query)
        print("judge domain", domain)
        context_str = ""
        if self.Task >= 2:
            apioutput = self.process_api(domain, query, query_time)
            if ("i don't know" not in apioutput):
                return apioutput
        elif self.Task == 1:
            output, context_str = self.process_task1(domain, query, query_time)
            if output!="":
                return output
        self.m.set_adapter("480")
        t1 = time.time()
        if self.r.init_retriever(search_results, query=query, task3=(self.Task == 3)):
            search_empty = 0
        else:
            search_empty = 1
        print("build retriever time:", time.time() - t1)
        print("start query")
        t1 = time.time()
        if (search_empty):
            res = [""]
        else:
            res = self.r.get_result(query, k=self.k)
        for snippet in res[:]:
            context_str += "<DOC>\n" + snippet + "\n</DOC>\n"
        context_str = self.tokenizer.encode(context_str, max_length=4000, add_special_tokens=False)
        print('len context_str', len(context_str))
        if len(context_str) >= 4000:
            context_str = self.tokenizer.decode(context_str) + "\n</DOC>\n"
        else:
            context_str = self.tokenizer.decode(context_str)
        print("query time:", time.time() - t1)
        filled_template = template_map['output_answer_nofalse'].format(context_str=context_str, query_str=query)

        messages = [
            {"role": "system",
             "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 70 words or less. Now is {query_time}"},
            {"role": "user", "content": filled_template},
        ]

        output, minn_logit, mean_logit = self.llam3_output(messages)

        if "i don't know" not in output and output not in ['i' "i don't"]:
            return output
        else:
            return "i don't know"

        return "i don't know"




