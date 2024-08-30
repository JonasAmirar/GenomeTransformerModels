import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def calculate_next_letter_percentages(file_path, n, output_path=None, accuracy_prediction=True, verbose=False):
    with open(file_path, 'r') as file:
        sequence = file.readline().strip().upper()

    subseq_counts = defaultdict(Counter)

    # Traverse the sequence to populate subseq_counts
    for i in tqdm(range(len(sequence) - n), desc="Processing", leave=False):
    #for i in range(len(sequence) - n):
        subseq = sequence[i:i+n]
        next_letter = sequence[i+n]
        subseq_counts[subseq][next_letter] += 1

    # Calculate relative percentages
    subseq_percentages = {}
    for subseq, counts in subseq_counts.items():
        total_count = sum(counts.values())
        subseq_percentages[subseq] = {letter: round((count / total_count) * 100, 10) for letter, count in counts.items()}

    # Create a DataFrame with all possible next letters as columns
    all_letters = sorted(set(sequence))  # Ensure all letters are columns
    data = []
    for subseq, percentages in subseq_percentages.items():
        row = [subseq] + [percentages.get(letter, 0.0) for letter in all_letters]
        data.append(row)

    columns = ["Context"] + all_letters
    df = pd.DataFrame(data, columns=columns)
    df.sort_values(by="Context", inplace=True)

    if output_path is not None:
        df.to_csv(output_path, index=False)

    if accuracy_prediction:
        # Count the occurrences of each context
        context_counts = pd.Series(Counter([sequence[i:i+n] for i in range(len(sequence) - n)])).sort_index()
        total_contexts = len(sequence) - n
        context_relative_freq = round((context_counts / total_contexts * 100),10)
        
        # Identify the most likely nucleotide for each context
        most_likely_nucleotide = df.iloc[:, 1:].idxmax(axis=1)
        
        # Combine all the information into a single DataFrame
        df['Context_Count_Absolute'] = context_counts.values
        df['Context_Count_Relative'] = context_relative_freq.values
        df['Most_Likely_Nucleotide'] = most_likely_nucleotide
        df['accuracy_most_likely'] = df.apply(lambda row: row[row['Most_Likely_Nucleotide']], axis=1)
        
        # Reset index to make 'Context' a column
        df.reset_index(drop=True, inplace=True)
        best_average_accuracy = ((df['accuracy_most_likely'].sum()) / (4 ** int(n))).round(3)
        #print(f"The average best accuracy achievable with a {n} base context per base is: {best_average_accuracy}% \n")

        # Calculate the weighted sum over the accuracy_most_likely column using Context_Count_Relative
        best_accuracy = ((df['accuracy_most_likely'] * df['Context_Count_Relative']).sum() / 100).round(3)
        
        if verbose: print(f"The best accuracy achievable with a {n} base context is: {best_accuracy}% \n\n")

        total_combinations = 4 ** n  # Assuming 4 possible nucleotides (A, T, C, G)
        present_combinations = len(df)
        percentage_present = (present_combinations / total_combinations) * 100
        if verbose: print(f"Only {present_combinations} combinations present of {total_combinations}, i.e. {percentage_present:.2f}%")
        
        if output_path is not None:
            df.to_csv(output_path, index=False)

    return df, best_accuracy



def filter_by_absolute_count(df, threshold):
    filtered_df = df[df['Context_Count_Absolute'] >= threshold]
    return filtered_df

def merge_dataframes_on_context(dfs, labels, n, threshold=10, output_path=None):
    
    if len(dfs) != len(labels):
        raise ValueError("The number of DataFrames and labels must be the same.")
    
    # Prepare DataFrames by filtering low counts, dropping 'A', 'C', 'G', 'T' columns and renaming columns
    processed_dfs = []
    for df, label in zip(dfs, labels):
        df = filter_by_absolute_count(df, threshold)
        df = df.drop(columns=['A', 'C', 'G', 'T'])
        df = df.rename(columns=lambda x: f"{x}_{label}" if x != 'Context' else x)
        processed_dfs.append(df)
    
    # Merge all DataFrames on 'Context'
    merged_df = processed_dfs[0]
    for df in processed_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Context', how='inner')  # Use inner join to ensure no NaN values
    
    # Filter rows where all 'Most_Likely_Nucleotide' columns contain the same value
    most_likely_cols = [col for col in merged_df.columns if col.startswith('Most_Likely_Nucleotide')]
    merged_df = merged_df[merged_df[most_likely_cols].nunique(axis=1) == 1]

    # Calculate the weighted sum over the accuracy_most_likely column using Context_Count_Relative
    best_accuracy_on_test = ((merged_df['accuracy_most_likely_test'] * merged_df['Context_Count_Relative_test']).sum() / 100).round(3)
    print(f"The best generalized accuracy achievable on the Test set with a {n} base context is: {best_accuracy_on_test}% \n\n")

    total_combinations = 4 ** n  #  4 possible nucleotides
    present_combinations = len(merged_df)
    percentage_present = (present_combinations / total_combinations) * 100
    print(f"Only {present_combinations} contexts of {total_combinations} generalise sufficiently between the sets, i.e. {percentage_present:.2f}%")
    
    if output_path is not None:
        merged_df.to_csv(output_path, index=False)

    return merged_df, best_accuracy_on_test, percentage_present
    


#t2t_traindata = '../genomic_data/t2t_genome/chr1.txt'
#t2t_valdata   = '../genomic_data/t2t_genome/chr21.txt'
t2t_testdata  = '../genomic_data/t2t_genome/chr22.txt'

for i in range(20):
    print(i,end =': ')
    #print('Processing train...', end ='')
    #result_train,_= calculate_next_letter_percentages(t2t_traindata,i, accuracy_prediction=True)
    #print('val...', end ='')
    #result_val,_  = calculate_next_letter_percentages(t2t_valdata,  i, accuracy_prediction=True)
    print('test')
    result_test,_ = calculate_next_letter_percentages(t2t_testdata ,i, accuracy_prediction=True)
     
    dfs = [result_test]
    labels = ['test']
    merged_df,best_accuracy_on_test, percentage_present= merge_dataframes_on_context(dfs, labels, i,output_path=f'autoregressive_best_model_calc/OnlyTest_Context_{i}.csv')
    with open('autoregressive_best_model_calc/OnlyTest_Summary.txt', 'a') as f:
            f.write('\n'+ f'Context {i} has accuracy {best_accuracy_on_test} on {percentage_present}%' + '\n')