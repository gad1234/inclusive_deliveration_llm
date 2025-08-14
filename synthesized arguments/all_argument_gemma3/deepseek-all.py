import asyncio

import ollama
import time
import functools

import pathlib
import pandas as pd
from datetime import datetime

import json
import re

#model configuration params
model_name = "deepseek-v2:16b"
model_config = "deepseek"

CONFIG = {
    "llama": {
            "num_gpu": 1,
            "temperature": 0.1,    # Low value to reduce creativity and hallucinations
            "top_p": 0.9,          # Limit diversity of considered tokens
            "top_k": 40,           # Token filtering for better responses
            "seed": 42,            # Set a consistent seed
            "repeat_penalty": 1.1, # Penalizes repetitions (Llama responds well to 1.1)
            "num_ctx": 8192,       # Llama 3.1 supports larger context windows
            "num_predict": 2048,   # Maximum tokens to generate
            "num_thread": 8        # Optimize threading for larger model
    },
    "gemma": {
        "num_gpu": 1,
        "temperature": 0.1,  # Low value to reduce creativity and allucinations
        "top_p": 0.9,        # Limit diversity of considered tokens
        "seed": 42,          # Set a consistent seed
        "repeat_penalty": 1.2 # penalizes restrictions to aid consistency
        #"num_ctx": 4096      # Match context window size
    },
    "deepseek": {
        "num_gpu": 1,
        "temperature": 0.1,    # Keep low for factual responses
        "top_p": 0.9,          # Good value for DeepSeek-V2
        "top_k": 50,           # DeepSeek-V2 benefits from slightly higher top_k
        "seed": 42,            # Keep for consistency
        "repeat_penalty": 1.1, # Slightly lower - DeepSeek-V2 handles repetition well
        "num_ctx": 32768,      # DeepSeek-V2 supports much larger context (32K tokens)
        "num_predict": 4096    # Can generate longer responses with V2
    }
}

context_polarized = ''
context_favor = ''
context_against = ''

###

##Time function
def timer(func):
    """
    A decorator that measures and prints the execution time of a function.
    
    Args:
        func: The function to be timed
        
    Returns:
        wrapper: The wrapped function that includes timing functionality
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record start time
        start_time = time.perf_counter()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Record end time
        end_time = time.perf_counter()
        
        # Calculate duration
        duration = end_time - start_time
        
        return result,duration
    return wrapper


async def generate_response(question):
    start_time = time.time()
    client = ollama.AsyncClient()
    
    response = await client.generate(model_name, question,options= CONFIG.get(model_config))
    end_time = time.time()
    duration = end_time - start_time
    
    return response['response'], duration

async def generate_argument_forlist(topic,index, filename):          
        op1 = df_opinions.at[index,'0'].replace('"', '\\"')
        op2 = df_opinions.at[index,'1'].replace('"', '\\"')
        op3 = df_opinions.at[index,'2'].replace('"', '\\"')

        #construct the prompt
        stance_type = df_opinions.at[index,'stance']
        
        #print(stance_type)
    
        if stance_type == 'favor':
            context = context_favor
        elif stance_type == 'against':
            context = context_against
        else:
            context = context_polarized

        #print(context)
                
        opinions1 = f'"{op1}",\n"{op2}",\n"{op3}"'
        
        prompt = context + opinions1  + '''] }''' 
        
        response,duration = await generate_response(prompt)
                                
        return response,duration,prompt,op1,op2,op3

async def generate_argument(topic,index, filename):     
    response,duration,prompt,op1,op2,op3 = await generate_argument_forlist(topic,index, filename) 

    return response,duration,prompt,op1,op2,op3  

async def background_task():   
    global file_name_complete
    global df_opinions
    global repres_docs_aux
    
    try:
        version='_v1'
        df_opinions = pd.read_csv('7repres-docs-topic-color-stance.csv') 
 
        filename = model_config + '_1shot_all'
        
        column = ["topic", "response", "prompt"]
        repres_docs_aux = pd.DataFrame(columns=column)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name_complete = timestr + filename +version+'.csv'
        for index, row in df_opinions.iterrows():
            topic = df_opinions.at[index, 'topic']
            print(topic)
            response,duration,prompt,op1,op2,op3 = await generate_argument(topic,index,file_name_complete)
            # Wait a bit between requests to avoid overwhelming the API
            await asyncio.sleep(2)
            # Create a new row for the DataFrame
            new_row = {
                        'timestamp': datetime.now(),
                        'topic': topic,
                        'prompt': prompt,
                        'response': response,
                        'op1': op1,
                        'op2': op2,
                        'op3': op3,
                        'duration': duration
            }
            repres_docs_aux = pd.concat([repres_docs_aux, pd.DataFrame([new_row])], ignore_index=True)
            repres_docs_aux.to_csv(file_name_complete, index=False)


        print('all responses processed.')
        repres_docs_aux.to_csv(file_name_complete, index=False)
    except Exception as e:
        print(e)
        print (topic)
        new_row = {"topic": topic, "response": '', "prompt": '', "duration":0}
        repres_docs_aux = pd.concat([repres_docs_aux, pd.DataFrame([new_row])], ignore_index=True)
        repres_docs_aux.to_csv(file_name_complete, index=False)

def read_configuration():
    global context_polarized
    global context_favor
    global context_against
    try:
        with open('context_favor.txt', 'r', encoding='utf-8') as filef:
            context_favor = filef.read()
        with open('context_against.txt', 'r', encoding='utf-8') as filea:
            context_against = filea.read()
        with open('context_polarized.txt', 'r', encoding='utf-8') as filep:
            context_polarized = filep.read()
        print("Files successfully read")
    except FileNotFoundError:
        print("Error: File not found")
      
    except Exception as e:
        print(f"Error reading file: {e}")
        
    
async def main():
    

    # Print info message
    print("Starting background Ollama processing...")
    print("Responses will be saved to csv file")
    read_configuration()
    
    # Run the background task
    await background_task()
    
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nSaving final data and exiting...')
        # Ensure the DataFrame is saved before exit
        repres_docs_aux.to_csv(file_name_complete, index=False)
        print(f'Data saved to ' + file_name_complete)
        print('Goodbye!')