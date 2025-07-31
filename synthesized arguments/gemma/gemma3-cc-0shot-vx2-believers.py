# 15/04 last prompt change. 

import asyncio

import ollama
import time
import functools

import pathlib
import pandas as pd
from datetime import datetime

import json
import re

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
        
        # Print execution time
        #print(f"Function '{func.__name__}' took {duration:.4f} seconds to execute")
        
        return result,duration
    return wrapper


async def generate_response(question):
    start_time = time.time()
    client = ollama.AsyncClient()
    response = await client.generate('Gemma3:4b', question,
        options={
        "num_gpu": 1,
        "temperature": 0.1,  # Low value to reduce creativity and allucinations
        "top_p": 0.9,        # Limit diversity of considered tokens
        "seed": 42,          # Set a consistent seed
        "repeat_penalty": 1.2 # penalizes restrictions to aid consistency
        #"num_ctx": 4096      # Match context window size
    })
    end_time = time.time()
    duration = end_time - start_time
    
    return response['response'], duration

async def generate_argument_forlist(topic,index, filename):   
        #print(df_opinions['topic'])
        #print(topics_list)
        
        op1 = df_opinions.at[index,'0'].replace('"', '\\"')
        op2 = df_opinions.at[index,'1'].replace('"', '\\"')
        op3 = df_opinions.at[index,'2'].replace('"', '\\"')

        opinions1 = f'"{op1}",\n"{op2}",\n"{op3}"'
 
        context = ''' 
Your task is to construct a compelling global argument supporting the widely held belief that human activity is the dominant driver of current climate change. You are not to explicitly state that humans cause climate change, but rather to present a perspective that recognizes the extent of human influence. To do this, don't add new information; only use the information provided by the opinions given.
Task: Constructing a Global Short Argument in Passive Voice Challenging that Climate Change is Caused by Humans. 
Objective: To synthesize multiple opinions into a single, coherent global short argument that reflects the shared concerns or themes, using passive voice and using only opinions provided without adding any new information.
Instructions:
You are given a list of opinions that address a common issue or theme, possibly from different perspectives. Your task is to:
1. Identify the main concerns expressed in the opinions.
2. Integrate these concerns into a single, unified global argument.
3. Write the global argument entirely in passive voice, presenting a cohesive perspective that reflects the central tension or goal shared among the individual opinions. Using only opinnions provided without adding any new information.
4. Return your output in JSON format using the following structure:
{
"global_argument": "[Your global argument written in passive voice]",
"comments":"[if you want to add comments or notes in reference to the output]"
}
**Input:**
{"opinions": [
        '''   

        prompt = context + opinions1  + '''] }''' 
        #print(prompt)
        response,duration = await generate_response(prompt)
        #print(response)
                                
        return response,duration,prompt,op1,op2,op3

async def generate_argument(topic,index, filename):     
    response,duration,prompt,op1,op2,op3 = await generate_argument_forlist(topic,index, filename) 

    return response,duration,prompt,op1,op2,op3  

def process_json(file_path=None, json_str=None):
    """
    Process a JSON file or string and extract the 'global_argument' and 'comments' fields.
    Handles JSON strings that might be wrapped in code blocks or have escape characters.
    
    Args:
        file_path (str, optional): Path to the JSON file
        json_str (str, optional): JSON string to process
    
    Returns:
        tuple: (global_argument, comments)
    """
    # Load JSON data
    if file_path:
        with open(file_path, 'r') as f:
            text = f.read()
    elif json_str:
        text = json_str
    else:
        raise ValueError("Either file_path or json_str must be provided")
    
    # Clean the input text
    # Remove code block markers if present
    text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
    # Fix trailing commas in objects
    fixed = re.sub(r',\s*}', r'}', text)
    # Fix trailing commas in arrays
    text = re.sub(r',\s*]', r']', fixed)

    # Try to parse the JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract the JSON portion
        # Look for patterns that might indicate JSON content
        json_pattern = r'{\s*"global_argument"\s*:.*?"comments"\s*:.*?}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                # Try to parse the extracted JSON
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                # If this still fails, we might need to handle escaped characters
                clean_text = match.group(0).replace('\\"', '"').replace('\\n', '\n')
                # Remove any remaining escapes that might interfere
                clean_text = re.sub(r'\\(.)', r'\1', clean_text)
                data = json.loads(clean_text)
        else:
            raise ValueError("Could not extract valid JSON from the input")
    
    # Extract the required fields
    global_argument = data.get("global_argument", "")
    comments = data.get("comments", "")
    
    return global_argument, comments

async def background_task():   
    #global repres_docs
    #global df_hier
    global file_name_complete
    global df_opinions
    global repres_docs_aux
    
    try:
        version='_bert2025_vx2'
        df_opinions = pd.read_csv('BERT2025-repres_docs_sample_favor.csv') 
 #pd.read_csv('../climate_change/one_representative_doc.csv')
        #df_hier = pd.read_csv('../climate_change/df_nodes_3_child.csv')

        filename = 'gemma3_0shot_favor'
        
        colum = ["topic", "response", "prompt"]
        repres_docs_aux = pd.DataFrame(columns=colum)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        # Resetear el índice  
        #df_hier = df_hier.reset_index(drop=True)
        file_name_complete = timestr + filename +version+'.csv'
        #la primer corrida se detuvo en la fila 56.
        for index, row in df_opinions.iterrows():
            topic = df_opinions.at[index, 'Unnamed: 0']
            print(topic)
            #topics_list = eval(df_hier.at[index, 'Topics'])
            response,duration,prompt,op1,op2,op3 = await generate_argument(topic,index,file_name_complete)
            # Wait a bit between requests to avoid overwhelming the API
            await asyncio.sleep(2)
            #Process response
            
            global_argument, comments = process_json(json_str=response)
            
            # Create a new row for the DataFrame
            new_row = {
                        'timestamp': datetime.now(),
                        'topic': topic,
                        'prompt': prompt,
                        'response': response,
                        'global_argument':global_argument,
                        'op1': op1,
                        'op2': op2,
                        'op3': op3,
                        'comments' : comments,
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

async def main():
    

    # Print info message
    print("Starting background Ollama processing...")
    print("Responses will be saved to csv file")

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