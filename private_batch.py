#!/usr/bin/env python3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptation of privateGPT.py to enable batch processing

Created on Tue Jul  4 20:43:18 2023

@author: 
"""

import os       # class _Environ
import sys      # exit()
# import time     # time()
import types    # class SimpleNamespace
import argparse # class ArgumentParser
import datetime # class datetime

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp

from constants import CHROMA_SETTINGS

from word_set import gen_word_set, distance_ws

def field_count(obj : object) -> int:
    return len(obj.__dict__)

def time_at_length(td : datetime.timedelta) -> str:
    result : str = ''
    secs   : int = td.seconds
    
    if td.days > 0: 
        result += str(td.days) +' day'
        
        if td.days > 1: 
            result += 's'
        
        result += ' '
        
    hours : int = secs // 3600
    if hours != 0:
        result += str(hours) + ' h '
        
    secs = secs % 3600
    minutes : int = secs // 60
    # print(f'minutes {minutes}')
    if minutes != 0: 
        result += str(minutes) + ' min '
        
    seconds : int = secs % 60
    micro   : int = int(round(td.microseconds / 10000, 0))
    
    sec_frac = seconds + (micro / 100.0)
        
    if sec_frac != 0:
        result += f'{sec_frac:.2f}' + ' s'
        
    result = result.strip()
    
    # Normal function termination
    return result


def setup(params : types.SimpleNamespace()) -> RetrievalQA:
    load_dotenv()
    
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    
    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

    print('*** Will HuggingFaceEmbeddings ***') 
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    print('*** Done HuggingFaceEmbeddings ***') 
    
    db = Chroma(persist_directory=persist_directory, 
                embedding_function=embeddings, 
                client_settings=CHROMA_SETTINGS)

    print('*** Opened db ***') 
    
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    print('*** Retrieved db ***') 
    
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if params.mute_stream else [StreamingStdOutCallbackHandler()]

    print('*** Preparing the LLM ***') 
    
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, 
                           n_ctx=model_n_ctx, 
                          backend='gptj', 
                           n_batch=model_n_batch, 
                           callbacks=callbacks, 
                           verbose=params.verbose)
        case "GPT4All":
            llm = GPT4All(model=model_path, 
                          n_ctx=model_n_ctx, 
                          backend='gptj', 
                          n_batch=model_n_batch, 
                          callbacks=callbacks, 
                          verbose=params.verbose)
        case _:
            # raise exception if model_type is not supported
            msg = f"Model type {model_type} is not supported. " +\
                "Please choose one of the following: LlamaCpp, GPT4All"
            raise Exception(msg)

    print('*** Will retrieve ***') 
        
    result : RetrievalQA = \
        RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents= not params.hide_source)

    print('*** Before leaving setup() ***') 
    
    # Normal function termination
    return result


def process_query(qa : RetrievalQA, query : str, hide_source : bool):
    """
    Process a query using `langchain.chains`

    Parameters
    ----------
    qa : RetrievalQA
        A `langchain` chain.
    query : str
        The question to be used.
    hide_source : bool
        Should the source be hidden ?

    Returns
    -------
    None.

    """
    ws_q = gen_word_set(query)

    # Print the query
    dt_start = datetime.datetime.now()
    s_dt_start = dt_start.strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n\n> {s_dt_start} Question: ')
    print(query)
    
    # Get the answer from the chain
    res = qa(query)
    answer, docs = res['result'], [] if hide_source else res['source_documents']

    # Print the result
    dt_end = datetime.datetime.now()
    s_dt_end = dt_end.strftime('%Y-%m-%d %H:%M:%S')
    delta = dt_end - dt_start
    print(delta)
    s_delta = time_at_length(delta)
    print(f"\n> {s_dt_end} Answer (took {s_delta}):")
    print(answer)

    # Print the relevant sources used for the answer
    best_inter : int = -1
    inter_len : int = -1
    best_ind : int = -1
    ind : int = -1
    
    for document in docs:
        ind += 1
        
        ws_answer = gen_word_set(document.page_content)
        inter_len = distance_ws(ws_q, ws_answer)
        
        if inter_len > best_inter: 
            best_inter = inter_len
            best_ind = ind
        
        msg = f'\n> {ind} Inter {inter_len} ' +\
            document.metadata["source"] + ":"
        print(msg)
        print(document.page_content)
    
    print(f'\nBest answer {best_ind}, with inter {best_inter}')
    
    # Normal function termination
    return
    

def parse_cli() -> argparse.Namespace:
    desc = 'privateGPT: Ask questions to your documents without an ' + \
        'internet connection, using the power of LLMs.'
    parser = argparse.ArgumentParser(description=desc)
    
    help_str = 'Use this flag to disable printing of source documents ' +\
        'used for answers.'
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help=help_str)

    help_str = 'Use this flag to disable the streaming StdOut callback ' +\
        'for LLMs.'
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help=help_str)
    
    help_str = 'Use this flag to enable the verbose tracing.'
    parser.add_argument("--verbose", "-v", action='store_true',
                        help=help_str)

    help_str = 'Text of single query'    
    parser.add_argument('-q', '--single_query', type=str, 
                        required=False, help=help_str)

    help_str = 'File name for queries'    
    parser.add_argument('-n', '--queries_name', type=str, 
                        required=False, help=help_str)
    
    result = parser.parse_args()

    if (result.single_query is not None) and (result.queries_name is not None):
        print('ERROR Can\'t chose both --single_query and --queries_name')
        
        # Return to indicate failure
        return argparse.Namespace()
    
    # Normal function termination
    return result


def interp_args( args : argparse.Namespace) -> types.SimpleNamespace:
    result : types.SimpleNamespace = types.SimpleNamespace()
    
    result.hide_source = args.hide_source    
    result.mute_stream = args.mute_stream
    result.verbose     = args.verbose
    
    result.single_query = args.single_query
    result.queries_name = args.queries_name
    
    if result.single_query is not None:
        # Normal function termination
        return result
    
    try: 
        queries_f = open(result.queries_name, 'r')
        
    except Exception as exc:
        msg = f'ERROR Could not open file \'{result.queries_name}\' '
        msg += 'for input\n'
        msg += '\t' + type(exc) + ': ' + str(exc)
        print(msg)
        
        # Return to indicate failure
        return types.SimpleNamespace()
    
    result.queries_file = queries_f
    
    queries = []
    for line in result.queries_file:
        queries.append(line.strip())
        
    result.queries = queries
        
    # Normal function termination
    return result


def main() -> int:
    # Parse the command line arguments
    args = parse_cli()
    print(args)
    
    if field_count(args) == 0:
        print('*** The program will be terminated. ***')
        
        # Return to indicate failure
        return 1
    
    params = interp_args(args)    
    if field_count(params) == 0:
        print('*** The program will be terminated. ***')
        
        # Return to indicate failure
        return 2

    print(params)
    
    dt_start = datetime.datetime.now()
    s_dt_start = dt_start.strftime('%Y-%m-%d %H:%M:%S')
    print(f'{s_dt_start} Started setup')
    qa = setup(params)
    dt_end = datetime.datetime.now()
    s_dt_end = dt_end.strftime('%Y-%m-%d %H:%M:%S')
    delta = dt_end - dt_start
    print(f'delta {delta}')
    s_delta = time_at_length(dt_end - dt_start)
    print(f"\n> {s_dt_end} Finished setup (took {s_delta}):")
    
    # Question and answer

    try:
        if params.single_query is not None:
            process_query(qa, params.single_query, params.hide_source)
    
        else:
            for query in params.queries:
                process_query(qa, query, params.hide_source)
        
    except Exception as exc:
        print('\n*** The program will be terminated ***')
        print(f'\t{type(exc).__name__} : {str(exc)}')
        
        # Return to indicate error
        return 3
        
    # Normal function termination
    return 0

if __name__ == "__main__":
    sys.exit(main())
