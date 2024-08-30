"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from itertools import product
import pandas as pd
import pickle
import numpy as np
from collections import Counter
# -----------------------------------------------------------------------------

init_from      = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir        = '' # ignored if init_from is not 'resume'
start          = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples    = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature    = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k          = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed           = 1337
device         = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype          = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'

genomic        = True
mode           = '' 
name           = ''
input_type     = 'test_data'
input_file     = ''
block_size     = 0
with_all_outputs   = False
with_results       = False
with_embeddings    = False
with_attention     = False 
with_probabilities = False




compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file  #add --genomic here

if not genomic: mode='autoreg'
else:
    assert block_size!=0 and name!='' and mode!='', 'Please provide block_size, mode and name as input parameter'

if genomic:
    if input_file == '': input_file=None
    if with_all_outputs:
        with_results=with_embeddings=with_attention=with_probabilities=True
    max_new_tokens = 1
    if not with_probabilities: 
        top_k = 1 
        print('ATTENTION: Taking only most likely prediction (top_k=1)')
    else:
        print('ATTENTION: Taking NOT only most likely prediction (top_k!=1), accuracy will be worse!')
    
    # to do: have only one of those!!
    context_size=context_length= block_size
    if input_type in ['val_data','train_data']: 
        print('\n\n ATTENTION: MODEL IS EVALUATED ON VALIDATION OR TRAINING SET! FOR MOST ANALYSIS CONSIDER USING THE TESTING SET INSTEAD! \n\n ')

    
    if   mode=='autoreg' : architecture=0
    elif mode=='basic'   : architecture=1
    elif mode=='multi'   : architecture=2
    elif mode=='advanced': architecture=3
        
    if input_type=='all_combinations' or input_type=='start': with_results = False  # cause we don't know correct results then 
    if input_type=='whole_chr'                              : with_results = True   # will want to store results always here 
    if input_type=='start' and start == "\n"                : print("Attention: Please provide valid start sequence!s")
    if start== "\n" : start='A' #genomic encoding doesnt know \n character 

# -----------------------------------------------------------------------------
### function definitions ###

def save_results_file(file_path,df):
    if os.path.exists(file_path):
        base, ext = os.path.splitext(file_path)
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"): i += 1
        file_path = f"{base}_{i}{ext}"
    df.to_csv(file_path, index=False)
    


def model_input(input_type,input_file,  s ):
    global device,contexts, context_size,mode, architecture, num_samples
    global first_letter, middle_letter, last_letter

    y_correct= None
    context= None
    position=None
    
    if  input_type == 'all_combinations' : 
        context = contexts[s]
        x = torch.tensor(encode(context), dtype=torch.long, device=device)[None, ...]
        
    elif input_type in ['test_data','val_data','train_data', 'whole_chr']:
        
        if input_file is not None:
            # THIS ALREADY ASSUMES ENCODED DATA!!!!!!!
            input_data = np.memmap(input_file, dtype=np.uint16, mode='r', shape=(os.path.getsize(input_file) // np.dtype(np.uint16).itemsize,))
        else: 
            if mode=='autoreg':    
                if input_type in ['test_data','whole_chr']:
                    input_data = np.memmap('data/genomic_char/architecture_nr_0_context_size_255/small_dataset_train_chr1/test_Chr22.bin',dtype=np.uint16, mode='r')
                elif input_type=='val_data':
                    input_data = np.memmap('data/genomic_char/architecture_nr_0_context_size_255/small_dataset_train_chr1/val_Chr21.bin',dtype=np.uint16, mode='r')
                elif input_type=='train_data':
                    input_data = np.memmap('data/genomic_char/architecture_nr_0_context_size_255/small_dataset_train_chr1/train_Chr1.bin',dtype=np.uint16, mode='r')
                    
            else: # mutation prediction
                test_file = f'data/genomic_char/architecture_nr_{architecture}_context_size_{context_size}/test.bin'
                val_file  = f'data/genomic_char/architecture_nr_{architecture}_context_size_{context_size}/val.bin'
                train_file  = f'data/genomic_char/architecture_nr_{architecture}_context_size_{context_size}/train.bin'

                if input_type =='test_data':
                    input_data = np.memmap(test_file, dtype=np.uint16, mode='r')
                    if s==0: print(f'Input data is {test_file} \n')                
                elif input_type=='val_data':
                    input_data = np.memmap(val_file, dtype=np.uint16, mode='r')
                    if s==0: print(f'Input data is {val_file} \n')                
                elif input_type=='train_data':
                    input_data = np.memmap(train_file, dtype=np.uint16, mode='r')
                    if s==0: print(f'Input data is {train_file} \n')                
                   
                """
                if os.path.isfile(test_file):
                    input_data = np.memmap(test_file, dtype=np.uint16, mode='r')
                    if s==0: print(f'Input data is {test_file} \n')
                else: 
                    input_data = np.memmap(val_file, dtype=np.uint16, mode='r')
                    if s==0: print(f"No Test data found! Proceeding with validation data ({val_file}) for testing and results will be useless :)")
                """

            
                
    
        # Get x and y depending on prediction mode
        if mode=='basic':
            ix = torch.randint(int((len(input_data) - 1) // (context_length + 2) + 1), (1,)) * (context_length + 2)
            x  = torch.stack([torch.tensor(input_data[i: i + context_length], dtype=torch.long, device=device) for i in ix])

            y_correct_encoded = torch.stack(
                                [torch.from_numpy(
                                        (input_data[i+context_length:i+context_length+1]).astype(np.int64))  for i in ix])
            y_correct =  decode(y_correct_encoded[0].tolist())       
            #context = decode(x[0].tolist())
            if context_size==3:
                first_letter = decode(x[0].tolist())[-3]
                middle_letter= decode(x[0].tolist())[-2]
                last_letter  = decode(x[0].tolist())[-1]
                
        elif mode=='autoreg':
            if input_type in ['test_data','val_data','train_data']:
                if input_file is None:
                    ix = torch.randint(len(input_data) - block_size, (1,))     
                else:
                    #print(test_data[i*(context_size+1)]
                    ix = torch.tensor([s*(context_size+1)])
                    print(ix)

            elif input_type == 'whole_chr':    

                #if False: ## Zooming in on telomeres
                        #input_data = input_data[:10000]
                        #input_data = input_data[len(input_data)-10000:]
                # for general whole chr
                """"   
                n_per_region = 50
                n_regions=num_samples//n_per_region
                current_region = s//n_per_region
                current_point_region = s - (current_region*n_per_region)
                chr_length = len(input_data) - block_size
                chr_region_spacing=chr_length//n_regions
                
                if s==0:
                    print(f'Taking {n_regions} point with {n_per_region} examples each')
                    print(f'chr_length is {chr_length}, thus with {n_regions} points spacing is {chr_region_spacing}')
                
                ix=torch.tensor([current_region*chr_region_spacing + current_point_region])
                print(f's is {s}, region {current_region}, point in region {current_point_region} \
                      and ix {current_region*chr_region_spacing + current_point_region}')
                
                """
                # for telomers 
                ix=torch.tensor([s])
                #position = ix+block_size 
                #position = position.item()q
                position=s


            x = torch.stack([torch.tensor(input_data[i  :i+  block_size], dtype=torch.long, device=device) for i in ix])
            #print(x)
            y_correct_encoded = torch.stack([torch.tensor(input_data[i+1:i+1+block_size], dtype=torch.long, device=device) for i in ix])
            #print('y ',decode(y_correct_encoded[0].tolist()))
            y_correct =  decode(y_correct_encoded[0].tolist())[-1]       
            #print(y_correct)

                
        elif mode=='multi' or mode=='advanced':   
            muts_per_input = 255 // (block_size + 3)
            #print('muts_per_input is', muts_per_input)
            ix = torch.randint(int((len(input_data) - (muts_per_input * (block_size + 3) +3 ) ) // (block_size+3)) , (1,)) * (block_size+3)
            x = torch.stack([torch.tensor(input_data[i:i+ muts_per_input*(block_size+3) -2], 
                            dtype=torch.long, device=device) for i in ix])
            if context_size==3:
                first_letter = decode(x[0].tolist())[-4]
                middle_letter= decode(x[0].tolist())[-3]
                last_letter  = decode(x[0].tolist())[-2]
                            
            y_correct_encoded = torch.stack([torch.from_numpy(np.array([
                                input_data[i + j * (block_size + 3) - 2] for j in range(1, muts_per_input + 1)
                                ]).astype(np.int64)) for i in ix ])
            y_last_correct= decode(y_correct_encoded[0].tolist())[-1]
            y_correct= y_last_correct
    
    context = decode(x[0].tolist())
    #if input_type=='test_data': print('ix:', ix)
    #print('context:', context)
    #print('x:',x)
    if input_type in ['test_data','val_data','train_data']: print('y_correct:',y_correct)
        
    """ # not sure what i need this part for
    if context == None: 
        context = ''.join(chr(c.item()) for c in x[0]) 
        print('context:', context)"""
    
    return x, y_correct, context, position



def save_attention(pkl_file_path, filename, context): # ,attention_maps):
    global context_size,name , base_results_path
    attention_maps = []
                   
    with open(pkl_file_path, 'rb') as f:
        while True:
            try:
                dat_singleLayer = pickle.load(f)
                attention_maps.append(dat_singleLayer)
            except EOFError:
                break
    n_layers= len(attention_maps)
    
    context_path = os.path.join(base_results_path, f'attention_analysis/attention_example_{filename}')
    os.makedirs(context_path, exist_ok=True)
    
    for layer in range(n_layers):      # Iterate over layers
        attention_layer = attention_maps[layer] 
        n_heads= attention_layer.shape[0]
        for head in range(n_heads):    # Iterate over heads
            att_head = attention_layer[head,:,:]  # Shape (context_size, context_size)
            att_df = pd.DataFrame(att_head, columns=list(context), index=list(context))
            save_results_file(os.path.join(context_path, f'layer{layer}_head_{head}.csv'),att_df)


def output(input_type='test_data', input_file=None, 
           with_results=False, with_embeddings=False, with_attention=False, with_probabilities=False):
    
    global num_samples, temperature, top_k , contexts, context_size, name, architecture, base_results_path, start, mode
    global first_letter, middle_letter, last_letter

    if input_type=='all_combinations' and context_length<11 and input_file is None:
        #Generate all possible combinations of 3-letter contexts
        contexts = [''.join(context) for context in product(['A', 'C', 'G', 'T'], repeat=context_length)] 
        num_samples= len(contexts)
        print(f'Generating all possible combinations of {block_size}-letter contexts, thus overriding num_samples to {num_samples} \n')

    if with_probabilities:  data = []
    if with_embeddings:     representations = [] 
    
    with torch.no_grad(), ctx:            
        for s in range(num_samples):
            ### clear old attention pkl  
            if with_attention:
                pkl_file_path = (
                    f'OUT_GENOMIC_MODELS/out_Architecture_nr_{architecture}_Context_size_{context_size}_Model_{name}/'
                    'attention_tensors.pkl')
                if os.path.exists(pkl_file_path): os.remove(pkl_file_path)

            ### define input x that goes into model 
            if not input_type=='start':
                x, y_correct , context, position = model_input(input_type,input_file, s)
            else:
                start_ids = encode(start)
                x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                y= None
                context=start
                print('context:', context)
                print('x:',x)

            ### model prediction 
            y, probs, representation = model.generate(x, 1, temperature=temperature, top_k=top_k)
            predicted_mutation= decode(y[0].tolist())[-1]
            print('--> The prediction is', y[-1, -1].item(), 'decoded as', predicted_mutation , '\n\n')

            ### store additional results
            if with_results:
                correct_positive_stats.append(predicted_mutation == y_correct)
                if input_type=='whole_chr':
                    position_stats.append({'prediction': predicted_mutation == y_correct,
                                          'position'   : position})

                if context_size==3 and mode!='autoreg':
                    results.append({'sample' : s, 
                                    'first_letter' : first_letter,
                                    'middle_letter': middle_letter,
                                    'last_letter'  : last_letter,
                                    'predicted_mutation': predicted_mutation, 
                                    'actual_mutation'   : y_correct
                                   })
            if with_embeddings:  representations.append([context] + representation[0].tolist())
            if with_attention:    
                save_attention(pkl_file_path, s, context)# attention_maps) # or: ,context,..)
            if with_probabilities:
                data.append([context] + probs[0].tolist() + [y_correct] )
                #print('appended [context] + probs[0].tolist() for context a c g t =', [context] + probs[0].tolist(), '\n\n')
   
    ### store more additional results
    if with_results:
        print('generating results...')
        correct_positive_rate = (sum(correct_positive_stats) / len(correct_positive_stats)) * 100
        result_text = f"Prediction accuracy of model {name}: {correct_positive_rate:.2f}% tested on {num_samples} examples."
        print(result_text)
        with open(os.path.join(base_results_path, f'accuracy_model_{name}_on_{input_type}_summary.txt'), 'a') as f:
            f.write('\n'+result_text + '\n')
        if context_size==3 and mode!='autoreg':
            results_df = pd.DataFrame(results)
            save_results_file(os.path.join(base_results_path, f'results_model_{name}.csv'),results_df)

        elif mode=='autoreg' and input_type=='whole_chr':
            position_stats_df = pd.DataFrame(position_stats)
            save_results_file(os.path.join(base_results_path, f'position_stats_model_{name}_telomere_p_test2.csv'),position_stats_df)
    
    if with_probabilities:                
        # Create DataFrames for probabilities and representations
        probs_df = pd.DataFrame(data, columns=['Context', 'A', 'C', 'G', 'T', 'Target']).round(5)
        save_results_file(os.path.join(base_results_path, f'prediction_probabilities_model_{name}.csv'),probs_df)
    if with_embeddings:
        representations_df = pd.DataFrame(representations).round(5)
        save_results_file(os.path.join(base_results_path, f'output_representations_model_{name}.csv'),representations_df)


### end of function definitions ###
# -----------------------------------------------------------------------------


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint['model_args']['attention_path'] = out_dir if with_attention else None
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f" Loading meta from {meta_path} \n")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    if genomic:
        meta_vocab_size = meta['vocab_size'] +1 
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
if genomic:

    data = []
    contexts=[]
    representations = []
    correct_positive_stats = []
    position_stats =[]
    results = []
    base_results_path = f'../results/all_results_by_model/model_{name}___with_architecture{architecture}_context_{context_size}'
    os.makedirs(base_results_path, exist_ok=True)
    
    """if input_file is not None and mode == 'autoreg':
        with open(input_file, 'r') as f:
            input_data = f.read()
            print('leninput',len(input_data))
            num_samples = int(len(input_data) / (2*(context_size + 1)))-1
            print('num_samples',num_samples)"""

    output(input_type=input_type, input_file=input_file, \
           with_results=with_results, with_embeddings=with_embeddings, with_attention=with_attention, with_probabilities=with_probabilities)
    

else: 
    with torch.no_grad(), ctx:
        for k in range(num_samples):
            y, _, _ = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()), '---------------')



"""
base, ext = os.path.splitext(filepath)
filepath  = next(f"{base}_{i}{ext}" for i in range(1,100) if not os.path.exists(f"{base}_{i}{ext}"))
"""
