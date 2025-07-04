import os
import torch
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, Gemma3ForCausalLM

import numpy as np
from flask import request, Flask, render_template

## os :- Interacts with the operating system
## torch:- PyTorch library, used for deep learning
## huggingface_hub.login:- For accessing private models/datasets from Hugging Face
## flask:- Used to build a simple web app

## Load the Gemma model and its tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-27b-it")

## Ideal values for temperature, top_p, repetition_penalty
params = [1.0, 0.95, 1.5]
#print("Params ", params)

## pipeline function :- [from the Hugging Face Transformers library] Sets up a text-generation interface using the loaded Gemma model and its tokenizer
code_explain = pipeline(
"text-generation",
model=model,
tokenizer=tokenizer,
device=0,             # -1 for CPU, 0 for GPU
max_new_tokens=600,   
temperature=params[0],      #adjusts the randomness of predictions
top_p=params[1],            #allows for more diversity in the output
repetition_penalty=params[2],   #higher value decreases repitition
do_sample=True       #creativity
)


##First check whether the code is Python or not
def chk_python(code):
    prompt = f"""
Is this a Python code? Reply Yes or No:
    {code}
    
    """

    reply = code_explain(prompt)[0]["generated_text"]
    generated_reply = reply[len(prompt):].strip()
    return generated_reply


## Function: First calls the chk_python() function. 
## If reply is "Yes", it will provide the explanation 
## If reply is "No", it will prompt the user to enter a python code 
def gemma_ask(code):
    prompt = f"""
Explain the following code in a few lines:
    {code}
    
    """

    reply = chk_python(code)
    if (reply == "Yes"):
        output = code_explain(prompt)[0]["generated_text"]
        generated_output = output[len(prompt):].strip()
        return generated_output
            
    
    else:
        return """
This does not look like a Python code, please enter a Python code.
"""



app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/result', methods=['GET'])
def disp_res():
    #print(request.form.values())
    inp = request.args.get("inp")
    #print(inp)
    res = gemma_ask(inp)
    return(res)


if __name__ == '__main__':
    app.run()
