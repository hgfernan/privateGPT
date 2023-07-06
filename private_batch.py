#!/usr/bin/env python3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptation of privateGPT.py to enable batch processing

Created on Tue Jul  4 20:43:18 2023

@author: 
"""

import os       # class _Environ
import time     # time()
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


def field_count(obj : object) ->dict:
    return len(obj.__dict__)

def time_at_length(dt : datetime.timedelta) -> str:
    result : str = ''
    secs   : int = dt.seconds 
    
    hours : int = secs // 3600
    if hours != 0:
        result += str(hours) + ' h '
        
    secs = secs % 3600
    minutes : int = secs // 60
    if minutes != 0: 
        result += str(minutes) + ' min '
        
    seconds : int = secs % 60
    micro   : int = int(round(dt.microseconds / 10000, 0))
    
    result = str(seconds)
        
    if micro != 0:
        result += '.' + str(micro)
    
    result += ' s'
    
    # Normal function termination
    return result


def setup(params : types.SimpleNamespace()) -> RetrievalQA:
    result : RetrievalQA() = None
    
    load_dotenv()
    
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    
    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

    # # Parse the command line arguments
    # args = parse_cli()
    # print(args)
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, 
                embedding_function=embeddings, 
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if params.mute_stream else [StreamingStdOutCallbackHandler()]
    
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, 
                           n_ctx=model_n_ctx, 
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
        
    result = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="stuff", 
                                     retriever=retriever, 
                                     return_source_documents= not params.hide_source)
    
    
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

    dt_start = datetime.datetime.now()
    s_dt_start = dt_start.strftime('%Y-%m-%d %H:%M:%S')
    print(f'{s_dt_start} Started setup')
    qa = setup(params)
    dt_end = datetime.datetime.now()
    s_dt_end = dt_end.strftime('%Y-%m-%d %H:%M:%S')
    s_delta = time_at_length(dt_end - dt_start)
    print(f"\n> {s_dt_end} Finished setup (took {s_delta}):")
    
    # Question and answer

    query = params.single_query
    
    # Print the query
    dt_start = datetime.datetime.now()
    s_dt_start = dt_start.strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n\n> {s_dt_start} Question: ')
    print(query)
    
    # Get the answer from the chain
    res = qa(query)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']

    # Print the result
    dt_end = datetime.datetime.now()
    s_dt_end = dt_end.strftime('%Y-%m-%d %H:%M:%S')
    s_delta = time_at_length(dt_end - dt_start)
    print(f"\n> {s_dt_end} Answer (took {s_delta}):")
    print(answer)

    # Print the relevant sources used for the answer
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
        
    # Normal function termination
    return 0


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

if __name__ == "__main__":
    main()
