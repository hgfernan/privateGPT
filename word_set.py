#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elimination of punctiation and grammatical words

Created on Mon Jul 17 13:47:48 2023

@author: hilton
"""

import re  #
import sys # argv, exit()

from typing import List, Set 

# TODO lower case the question
# TODO replace all punctuation from question with space
# TODO create a question word set 

# TODO lower case the answer
# TODO replace all punctuation from answer with space
# TODO create an answer word set 

# TODO compute the cardinality of the intersection set

def gen_word_set(s : str) -> Set[str]:
    result : Set[str]
    
    sl : str = s.lower()
    r : str = re.sub('[\n\t,.;:\-?!]', ' ', sl)
    
    fields : List[str] = r.split()
    
    # result : Set[str] = set(fields)
    result : Set[List[str]] = set(fields)
    
    # Normal function termination
    return result

def distance_ws(ws1 : Set[str], ws2 : Set[str]) -> int:
    result : int = 0
    
    result = len(ws1.intersection(ws2))
    
    # Normal function termination 
    return result

def main(argv : List[str]) -> int:
    s1 : str = """
        E a José chamou Faraó de Zafenate-Paneia e lhe deu por mulher a 
        Asenate, filha de Potífera, sacerdote de Om; e percorreu José toda 
        a terra do Egito.    
        """
    ws1 = gen_word_set(s1)
    
    print(ws1)
    
    s2 = """
        Respondeu-lhe Efrom:
        """
    
    ws2 = gen_word_set(s2)
    
    print(ws2)
    
    ws_inter : Set[str] = ws1.intersection(ws2)
    print(ws_inter)
    
    inter_len : int = len(ws_inter)
    print(inter_len)
    
    inter_len = distance_ws(ws1, ws2)
    
    print(f'distance between the two sets: {inter_len}')
    
    # Normal function termination
    return 0 

if __name__ == '__main__': 
    sys.exit(main(sys.argv))
