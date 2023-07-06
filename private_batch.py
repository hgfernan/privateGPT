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
import argparse # class ArgumentParser
import datetime # class datetime

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp

from constants import CHROMA_SETTINGS

def setup(args : argparse.Namespace) -> RetrievalQA:
    result : RetrievalQA() = None
    
    load_dotenv()
    
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    
    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

    # Parse the command line arguments
    args = parse_arguments()
    print(args)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, 
                embedding_function=embeddings, 
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, 
                           n_ctx=model_n_ctx, 
                           n_batch=model_n_batch, 
                           callbacks=callbacks, 
                           verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, 
                          n_ctx=model_n_ctx, 
                          backend='gptj', 
                          n_batch=model_n_batch, 
                          callbacks=callbacks, 
                          verbose=False)
        case _:
            # raise exception if model_type is not supported
            msg = f"Model type {model_type} is not supported. " +\
                "Please choose one of the following: LlamaCpp, GPT4All"
            raise Exception(msg)
        
    result = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="stuff", 
                                     retriever=retriever, 
                                     return_source_documents= not args.hide_source)
    
    
    # Normal function termination
    return result

def main():

    # load_dotenv()
    
    # embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    # persist_directory = os.environ.get('PERSIST_DIRECTORY')
    
    # model_type = os.environ.get('MODEL_TYPE')
    # model_path = os.environ.get('MODEL_PATH')
    # model_n_ctx = os.environ.get('MODEL_N_CTX')
    # model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
    # target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

    # Parse the command line arguments
    args = parse_arguments()
    print(args)
    
    # embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # db = Chroma(persist_directory=persist_directory, 
    #             embedding_function=embeddings, 
    #             client_settings=CHROMA_SETTINGS)
    # retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # # activate/deactivate the streaming StdOut callback for LLMs
    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    # # Prepare the LLM
    # match model_type:
    #     case "LlamaCpp":
    #         llm = LlamaCpp(model_path=model_path, 
    #                        n_ctx=model_n_ctx, 
    #                        n_batch=model_n_batch, 
    #                        callbacks=callbacks, 
    #                        verbose=False)
    #     case "GPT4All":
    #         llm = GPT4All(model=model_path, 
    #                       n_ctx=model_n_ctx, 
    #                       backend='gptj', 
    #                       n_batch=model_n_batch, 
    #                       callbacks=callbacks, 
    #                       verbose=False)
    #     case _:
    #         # raise exception if model_type is not supported
    #         msg = f"Model type {model_type} is not supported. " +\
    #             "Please choose one of the following: LlamaCpp, GPT4All"
    #         raise Exception(msg)
        
    # qa = RetrievalQA.from_chain_type(llm=llm, 
    #                                  chain_type="stuff", 
    #                                  retriever=retriever, 
    #                                  return_source_documents= not args.hide_source)
    
    qa = setup(args)
    print(type(qa))
    
    # Question and answer

    query = args.query
    
    # Get the answer from the chain
    start = time.time()

    # Print the query
    dt_start = datetime.datetime.fromtimestamp(start)
    s_dt_start = dt_start.strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n\n> {s_dt_start} Question: ')
    print(query)
    res = qa(query)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']
    end = time.time()

    # Print the result
    end = time.time()
    dt_end = datetime.datetime.fromtimestamp(end)
    s_dt_end = dt_end.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n> {s_dt_end} Answer (took {round(end - start, 2)} s.):")
    print(answer)

    # Print the relevant sources used for the answer
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)

def parse_arguments():
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

    help_str = 'Text to query'    
    parser.add_argument('-q', '--query', type=str, 
                        required=True, help=help_str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
